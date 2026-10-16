"""
scratch_specialist.py

Binary Scratch Specialist Network targeting bit 6 (Scratch):
1. Directional Line-Kernel Morphology Bank (8 orientations: 0, 30, 45, 60, 90, 120, 135, 150 deg)
2. Laplacian Filter Bank (2 scales)
3. Native-resolution feature extraction with FastEPCA (max + avg spatial attention)
4. Mixed precision training on canonical 80/20 train split (27,370 wafers)
5. Validation-calibrated confidence on 3,042 val wafers
6. Strict VRAM cap: 55% (~3.3 GB limit)
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Thermal & VRAM safety configuration
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)
    torch.backends.cudnn.benchmark = True
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "conformal_wafer"))
DATA_PATH = os.path.join(BASE_DIR, "conformal_wafer", "data", "Wafer_Map_Datasets.npz")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "conformal_wafer", "experiments_sota")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Directional Line-Kernel & Laplacian Filter Bank
# -------------------------------------------------------------------------
class LineAndLaplacianFilterBank(nn.Module):
    def __init__(self, L=7):
        super().__init__()
        r = L // 2
        kernels = []
        # 8 directional line kernels (7x7)
        angles = [0, 30, 45, 60, 90, 120, 135, 150]
        for a in angles:
            th = math.radians(a)
            k = torch.zeros(L, L, dtype=torch.float32)
            for y in range(-r, r + 1):
                for x in range(-r, r + 1):
                    d = abs(x * math.sin(th) - y * math.cos(th))
                    along = abs(x * math.cos(th) + y * math.sin(th))
                    if d <= 0.6 and along <= r:
                        k[y + r, x + r] = 1.0
            k_sum = k.sum()
            if k_sum > 0:
                k = k / k_sum
            k = k - k.mean()
            kernels.append(k)

        # 2 Laplacian filters (3x3 padded to 7x7)
        lap1 = torch.zeros(L, L, dtype=torch.float32)
        lap1[r - 1:r + 2, r - 1:r + 2] = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]) / 4.0

        lap2 = torch.zeros(L, L, dtype=torch.float32)
        lap2[r - 1:r + 2, r - 1:r + 2] = torch.tensor([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]]) / 8.0

        kernels.extend([lap1, lap2])
        # Stack 10 kernels of shape (10, 1, 7, 7)
        weight = torch.stack(kernels, dim=0).unsqueeze(1)
        self.register_buffer("weight", weight)
        self.pad = r

    def forward(self, defect_channel):
        # defect_channel: (B, 1, H, W)
        edges = F.conv2d(defect_channel, self.weight, padding=self.pad)
        return torch.abs(edges) # (B, 10, H, W)

# -------------------------------------------------------------------------
# Fast Edge-Preserving Coordinate Attention
# -------------------------------------------------------------------------
class FastEPCA(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid_channels = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = x.mean(dim=3, keepdim=True) + x.amax(dim=3, keepdim=True)
        x_w = (x.mean(dim=2, keepdim=True) + x.amax(dim=2, keepdim=True)).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))
        return identity * a_h * a_w

# -------------------------------------------------------------------------
# Scratch Specialist Architecture
# -------------------------------------------------------------------------
class ScratchSpecialistBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.stride = stride
        mid_c = out_c
        self.pw1 = nn.Sequential(nn.Conv2d(in_c, mid_c, 1, bias=False), nn.BatchNorm2d(mid_c), nn.GELU())
        self.dw = nn.Conv2d(mid_c, mid_c, 5, stride=stride, padding=2, groups=mid_c, bias=False)
        self.dw_bn = nn.BatchNorm2d(mid_c)
        self.dw_act = nn.GELU()
        self.ca = FastEPCA(mid_c)
        self.pw2 = nn.Sequential(nn.Conv2d(mid_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm2d(out_c))

    def forward(self, x):
        res = self.shortcut(x)
        f = self.pw1(x)
        f = self.dw_act(self.dw_bn(self.dw(f)))
        f = self.ca(f)
        out = self.pw2(f)
        return F.gelu(out + res)

class ScratchSpecialistNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input channels: 1 defect channel + 10 line/Laplacian edge maps = 11 channels
        self.filter_bank = LineAndLaplacianFilterBank(L=7)
        self.stem = nn.Sequential(
            nn.Conv2d(11, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU()
        )
        self.stage1 = nn.Sequential(ScratchSpecialistBlock(48, 80, stride=1), ScratchSpecialistBlock(80, 80, stride=1))
        self.stage2 = nn.Sequential(ScratchSpecialistBlock(80, 128, stride=2), ScratchSpecialistBlock(128, 128, stride=1))
        self.stage3 = nn.Sequential(ScratchSpecialistBlock(128, 192, stride=2), ScratchSpecialistBlock(192, 192, stride=1))

        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(192 * 2, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x is (B, 3, 52, 52); channel 2 is defect die
        defect_ch = x[:, 2:3, :, :]
        edges = self.filter_bank(defect_ch) # (B, 10, 52, 52)
        f_in = torch.cat([defect_ch, edges], dim=1) # (B, 11, 52, 52)
        
        f = self.stem(f_in)
        f = self.stage1(f)
        f = self.stage2(f)
        f = self.stage3(f)

        emb = torch.cat([self.pool_max(f).flatten(1), self.pool_avg(f).flatten(1)], dim=1)
        logits = self.head(emb).squeeze(1)
        return logits

# -------------------------------------------------------------------------
# Dataset with Binary Scratch Labels
# -------------------------------------------------------------------------
class BinaryScratchDataset(Dataset):
    def __init__(self, images, scratch_labels, is_train=False):
        self.images = images
        self.labels = scratch_labels
        self.is_train = is_train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        y = float(self.labels[idx])

        ch0 = (img == 0).astype(np.float32)
        ch1 = (img == 1).astype(np.float32)
        ch2 = (img >= 2).astype(np.float32)
        tensor_img = np.stack([ch0, ch1, ch2], axis=0)

        if self.is_train:
            if np.random.rand() > 0.5:
                tensor_img = np.flip(tensor_img, axis=2).copy()
            if np.random.rand() > 0.5:
                tensor_img = np.flip(tensor_img, axis=1).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                tensor_img = np.rot90(tensor_img, k=k, axes=(1, 2)).copy()

        return {
            'image': torch.from_numpy(tensor_img),
            'label': torch.tensor(y, dtype=torch.float32)
        }

def get_scratch_loaders():
    raw = np.load(DATA_PATH, allow_pickle=True)
    x = raw['arr_0']
    y_vecs = raw['arr_1']
    y_scratch = (y_vecs[:, 6] == 1).astype(np.float32)

    from conformal_wafer.src.dataset import CLASS_DEFINITIONS
    vec_map = {tuple(vec): idx for idx, (cid, name, vec, depth) in enumerate(CLASS_DEFINITIONS)}
    y_comp = np.array([vec_map[tuple(v.tolist())] for v in y_vecs])

    n_total = len(x)
    train_full_idx, test_idx = train_test_split(np.arange(n_total), test_size=0.20, stratify=y_comp, random_state=42)
    train_idx, val_idx = train_test_split(train_full_idx, test_size=0.10, stratify=y_comp[train_full_idx], random_state=42)

    train_ds = BinaryScratchDataset(x[train_idx], y_scratch[train_idx], is_train=True)
    val_ds = BinaryScratchDataset(x[val_idx], y_scratch[val_idx], is_train=False)
    test_ds = BinaryScratchDataset(x[test_idx], y_scratch[test_idx], is_train=False)

    loaders = {
        'train': DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=True),
        'val': DataLoader(val_ds, batch_size=128, shuffle=False, pin_memory=True),
        'test': DataLoader(test_ds, batch_size=128, shuffle=False, pin_memory=True)
    }
    return loaders, train_idx, val_idx, test_idx

# -------------------------------------------------------------------------
# Training Routine
# -------------------------------------------------------------------------
def train_scratch_specialist(epochs=18, lr=1e-3):
    print("=" * 80, flush=True)
    print("TRAINING BINARY SCRATCH SPECIALIST (DIRECTIONAL LINE-KERNEL STREAM)", flush=True)
    print("=" * 80, flush=True)

    loaders, _, _, _ = get_scratch_loaders()
    model = ScratchSpecialistNet().to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ScratchSpecialistNet Parameters: {param_count:,} (Ultra-Compact)", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.1], device=DEVICE))

    best_val_auc = 0.0
    save_path = os.path.join(EXPERIMENTS_DIR, "scratch_specialist_best.pt")

    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total_loss, n_samples = 0.0, 0

        for batch in loaders['train']:
            images = batch['image'].to(DEVICE, non_blocking=True)
            labels = batch['label'].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=True):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            n_samples += images.size(0)

        scheduler.step()
        train_loss = total_loss / n_samples

        if ep % 2 == 0 or ep >= epochs - 2:
            model.eval()
            val_probs, val_targets = [], []
            with torch.no_grad():
                for batch in loaders['val']:
                    images = batch['image'].to(DEVICE)
                    labels = batch['label'].to(DEVICE)
                    with autocast('cuda', enabled=True):
                        logits = model(images)
                    probs = torch.sigmoid(logits)
                    val_probs.append(probs.cpu().numpy())
                    val_targets.append(labels.cpu().numpy())
            y_val_t = np.concatenate(val_targets)
            y_val_p = np.concatenate(val_probs)
            auc = roc_auc_score(y_val_t, y_val_p)
            acc = accuracy_score(y_val_t, (y_val_p >= 0.5).astype(int))
            rec = recall_score(y_val_t, (y_val_p >= 0.5).astype(int))
            prec = precision_score(y_val_t, (y_val_p >= 0.5).astype(int))
            print(f"  Epoch {ep:02d}/{epochs:02d} | Loss: {train_loss:.4f} | Val Acc: {acc*100:.2f}% | Recall: {rec*100:.2f}% | Prec: {prec*100:.2f}% | ROC-AUC: {auc:.4f}", flush=True)

            if auc > best_val_auc:
                best_val_auc = auc
                torch.save({'model_state_dict': model.state_dict(), 'val_auc': auc}, save_path)
                print(f"    -> Best Scratch Specialist saved! (ROC-AUC: {auc:.4f})", flush=True)

    dt = time.time() - t_start
    print(f"Training completed in {dt:.1f}s ({dt/60:.1f} mins)!", flush=True)
    return save_path

# -------------------------------------------------------------------------
# D4-TTA Evaluator for Binary Scratch Specialist
# -------------------------------------------------------------------------
def evaluate_scratch_d4(model, loader):
    model.eval()
    all_probs, targets = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            sum_p = 0.0
            for k in range(4):
                xk = torch.rot90(images, k, [2, 3])
                for flip in [False, True]:
                    x_v = torch.flip(xk, [3]) if flip else xk
                    with autocast('cuda', enabled=True):
                        logits = model(x_v)
                    sum_p += torch.sigmoid(logits)
            all_probs.append((sum_p / 8.0).cpu().numpy())
            targets.append(labels.cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(targets, axis=0)
    return probs, y_true

if __name__ == '__main__':
    save_path = train_scratch_specialist(epochs=18, lr=1e-3)
    
    # Evaluate with D4-TTA on Val and Test
    loaders, _, _, _ = get_scratch_loaders()
    model = ScratchSpecialistNet().to(DEVICE)
    ckpt = torch.load(save_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    print("\nComputing D4-TTA Scratch Probabilities on Validation Set...", flush=True)
    val_probs, y_val = evaluate_scratch_d4(model, loaders['val'])
    print(f"Val ROC-AUC: {roc_auc_score(y_val, val_probs):.4f} | Acc: {accuracy_score(y_val, (val_probs>=0.5).astype(int))*100:.2f}% | Recall: {recall_score(y_val, (val_probs>=0.5).astype(int))*100:.2f}%", flush=True)

    print("\nComputing D4-TTA Scratch Probabilities on Held-Out Test Set...", flush=True)
    test_probs, y_test = evaluate_scratch_d4(model, loaders['test'])
    print(f"Test ROC-AUC: {roc_auc_score(y_test, test_probs):.4f} | Acc: {accuracy_score(y_test, (test_probs>=0.5).astype(int))*100:.2f}% | Recall: {recall_score(y_test, (test_probs>=0.5).astype(int))*100:.2f}%", flush=True)

    # Save test probabilities for ensemble rescoring
    out_prob_path = os.path.join(EXPERIMENTS_DIR, "scratch_specialist_probs.npz")
    np.savez_compressed(out_prob_path, val_probs=val_probs, y_val=y_val, test_probs=test_probs, y_test=y_test)
    print(f"Saved scratch specialist probabilities to: {out_prob_path}", flush=True)
