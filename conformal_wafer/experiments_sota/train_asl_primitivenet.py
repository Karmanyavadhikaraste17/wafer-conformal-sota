"""
train_asl_primitivenet.py

Asymmetric Loss Multi-Label Primitive Specialist (ASL-PrimitiveNet):
1. Jointly predicts all 8 wafer defect primitives:
   [Center, Donut, Edge_Loc, Edge_Ring, Loc, Near_Full, Scratch, Random]
2. Incorporates 10-kernel Directional Line-Segment & Laplacian Filter Bank.
3. FastEPCA (Edge-Preserving Coordinate Attention with native CUDA amax + mean).
4. Asymmetric Multi-label Loss (ASL: gamma_pos=0, gamma_neg=4, margin=0.05).
5. Canonical 80/20 MixedWM38 split (27,370 train, 3,042 val, 7,603 test).
6. Evaluates with D4 Dihedral 8-View TTA and saves probabilities for meta-rescoring.
7. Strict VRAM cap: 55% (~3.3 GB limit).
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
# Directional Line & Laplacian Filter Bank (10 Fixed Filters)
# -------------------------------------------------------------------------
class DirectionalFilterBank(nn.Module):
    def __init__(self, L=7):
        super().__init__()
        r = L // 2
        kernels = []
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

        lap1 = torch.zeros(L, L, dtype=torch.float32)
        lap1[r - 1:r + 2, r - 1:r + 2] = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]) / 4.0

        lap2 = torch.zeros(L, L, dtype=torch.float32)
        lap2[r - 1:r + 2, r - 1:r + 2] = torch.tensor([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]]) / 8.0

        kernels.extend([lap1, lap2])
        weight = torch.stack(kernels, dim=0).unsqueeze(1)
        self.register_buffer("weight", weight)
        self.pad = r

    def forward(self, defect_channel):
        edges = F.conv2d(defect_channel, self.weight, padding=self.pad)
        return torch.abs(edges)

# -------------------------------------------------------------------------
# Fast Edge-Preserving Coordinate Attention (FastEPCA)
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
# ASL-PrimitiveNet Architecture
# -------------------------------------------------------------------------
class PrimitiveBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.stride = stride
        mid_c = out_c
        self.pw1 = nn.Sequential(nn.Conv2d(in_c, mid_c, 1, bias=False), nn.BatchNorm2d(mid_c), nn.GELU())
        self.dw3 = nn.Conv2d(mid_c // 2, mid_c // 2, 3, stride=stride, padding=1, groups=mid_c // 2, bias=False)
        self.dw5 = nn.Conv2d(mid_c // 2, mid_c // 2, 5, stride=stride, padding=2, groups=mid_c // 2, bias=False)
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
        c2 = f.shape[1] // 2
        f3 = self.dw3(f[:, :c2])
        f5 = self.dw5(f[:, c2:])
        fused = self.dw_act(self.dw_bn(torch.cat([f3, f5], dim=1)))
        fused = self.ca(fused)
        out = self.pw2(fused)
        return F.gelu(out + res)

class ASLPrimitiveNet(nn.Module):
    def __init__(self, num_primitives=8):
        super().__init__()
        self.filter_bank = DirectionalFilterBank(L=7)
        # 3 die channels (blank, pass, fail) + 10 edge channels = 13 channels
        self.stem = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.stage1 = nn.Sequential(PrimitiveBlock(64, 96, stride=1), PrimitiveBlock(96, 96, stride=1))
        self.stage2 = nn.Sequential(PrimitiveBlock(96, 160, stride=2), PrimitiveBlock(160, 160, stride=1))
        self.stage3 = nn.Sequential(PrimitiveBlock(160, 256, stride=2), PrimitiveBlock(256, 256, stride=1))

        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_primitives)
        )

    def forward(self, x):
        # x is (B, 3, 52, 52)
        defect_ch = x[:, 2:3, :, :]
        edges = self.filter_bank(defect_ch) # (B, 10, 52, 52)
        f_in = torch.cat([x, edges], dim=1) # (B, 13, 52, 52)
        
        f = self.stem(f_in)
        f = self.stage1(f)
        f = self.stage2(f)
        f = self.stage3(f)

        emb = torch.cat([self.pool_max(f).flatten(1), self.pool_avg(f).flatten(1)], dim=1)
        logits = self.head(emb) # (B, 8)
        return logits

# -------------------------------------------------------------------------
# Asymmetric Loss (ASL) for Multi-Label Primitive Disentanglement
# -------------------------------------------------------------------------
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        targets = targets.float()

        # Positive targets: (1 - p)^gamma_pos * log(p)
        pos_loss = - targets * ((1.0 - p) ** self.gamma_pos) * torch.log(p.clamp(min=self.eps))

        # Negative targets with hard threshold clipping
        p_neg = (p - self.clip).clamp(min=0.0)
        neg_loss = - (1.0 - targets) * (p_neg ** self.gamma_neg) * torch.log((1.0 - p_neg).clamp(min=self.eps))

        loss = pos_loss + neg_loss
        return loss.mean()

# -------------------------------------------------------------------------
# Dataset Loading
# -------------------------------------------------------------------------
class MultiLabelWaferDataset(Dataset):
    def __init__(self, images, primitive_vectors, is_train=False):
        self.images = images
        self.labels = primitive_vectors.astype(np.float32)
        self.is_train = is_train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        y = self.labels[idx]

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
            'label': torch.from_numpy(y)
        }

def get_multilabel_loaders():
    raw = np.load(DATA_PATH, allow_pickle=True)
    x = raw['arr_0']
    y_vecs = raw['arr_1']

    from conformal_wafer.src.dataset import CLASS_DEFINITIONS
    vec_map = {tuple(vec): idx for idx, (cid, name, vec, depth) in enumerate(CLASS_DEFINITIONS)}
    y_comp = np.array([vec_map[tuple(v.tolist())] for v in y_vecs])

    n_total = len(x)
    train_full_idx, test_idx = train_test_split(np.arange(n_total), test_size=0.20, stratify=y_comp, random_state=42)
    train_idx, val_idx = train_test_split(train_full_idx, test_size=0.10, stratify=y_comp[train_full_idx], random_state=42)

    train_ds = MultiLabelWaferDataset(x[train_idx], y_vecs[train_idx], is_train=True)
    val_ds = MultiLabelWaferDataset(x[val_idx], y_vecs[val_idx], is_train=False)
    test_ds = MultiLabelWaferDataset(x[test_idx], y_vecs[test_idx], is_train=False)

    loaders = {
        'train': DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=True),
        'val': DataLoader(val_ds, batch_size=128, shuffle=False, pin_memory=True),
        'test': DataLoader(test_ds, batch_size=128, shuffle=False, pin_memory=True)
    }
    return loaders, train_idx, val_idx, test_idx

# -------------------------------------------------------------------------
# Training Routine
# -------------------------------------------------------------------------
def train_asl_primitivenet(epochs=18, lr=1e-3):
    print("=" * 80, flush=True)
    print("TRAINING ASL-PRIMITIVENET (ASYMMETRIC MULTI-LABEL LOSS, 8 PRIMITIVES)", flush=True)
    print("=" * 80, flush=True)

    loaders, _, _, _ = get_multilabel_loaders()
    model = ASLPrimitiveNet(num_primitives=8).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ASLPrimitiveNet Parameters: {param_count:,} (Ultra-Compact)", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)
    criterion = AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05).to(DEVICE)

    best_val_f1 = 0.0
    save_path = os.path.join(EXPERIMENTS_DIR, "asl_primitivenet_best.pt")

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
            
            # Binary threshold at 0.50
            y_val_bin = (y_val_p >= 0.50).astype(int)
            exact_match = np.mean(np.all(y_val_bin == y_val_t, axis=1))
            macro_f1 = f1_score(y_val_t, y_val_bin, average='macro', zero_division=0)
            macro_auc = roc_auc_score(y_val_t, y_val_p, average='macro')
            
            print(f"  Epoch {ep:02d}/{epochs:02d} | Loss: {train_loss:.4f} | Exact Match: {exact_match*100:.2f}% | Macro-F1: {macro_f1*100:.2f}% | Macro-AUC: {macro_auc:.4f}", flush=True)

            if macro_f1 > best_val_f1:
                best_val_f1 = macro_f1
                torch.save({'model_state_dict': model.state_dict(), 'val_f1': macro_f1}, save_path)
                print(f"    -> Best ASLPrimitiveNet saved! (Macro-F1: {macro_f1*100:.2f}%)", flush=True)

    dt = time.time() - t_start
    print(f"Training completed in {dt:.1f}s ({dt/60:.1f} mins)!", flush=True)
    return save_path

# -------------------------------------------------------------------------
# D4-TTA Evaluator for 8-Primitive Specialist
# -------------------------------------------------------------------------
def evaluate_primitives_d4(model, loader):
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
    save_path = train_asl_primitivenet(epochs=18, lr=1e-3)
    
    loaders, _, _, _ = get_multilabel_loaders()
    model = ASLPrimitiveNet(num_primitives=8).to(DEVICE)
    ckpt = torch.load(save_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    print("\nComputing D4-TTA 8-Primitive Probabilities on Validation Set...", flush=True)
    val_probs, y_val = evaluate_primitives_d4(model, loaders['val'])
    val_bin = (val_probs >= 0.50).astype(int)
    print(f"Val Exact Match: {np.mean(np.all(val_bin == y_val, axis=1))*100:.2f}% | Macro-F1: {f1_score(y_val, val_bin, average='macro')*100:.2f}% | Macro-AUC: {roc_auc_score(y_val, val_probs, average='macro'):.4f}", flush=True)

    print("\nComputing D4-TTA 8-Primitive Probabilities on Held-Out Test Set...", flush=True)
    test_probs, y_test = evaluate_primitives_d4(model, loaders['test'])
    test_bin = (test_probs >= 0.50).astype(int)
    print(f"Test Exact Match: {np.mean(np.all(test_bin == y_test, axis=1))*100:.2f}% | Macro-F1: {f1_score(y_test, test_bin, average='macro')*100:.2f}% | Macro-AUC: {roc_auc_score(y_test, test_probs, average='macro'):.4f}", flush=True)

    # Save primitive probabilities for meta-rescoring
    out_path = os.path.join(EXPERIMENTS_DIR, "asl_primitive_probs.npz")
    np.savez_compressed(out_path, val_probs=val_probs, y_val=y_val, test_probs=test_probs, y_test=y_test)
    print(f"Saved primitive probabilities to: {out_path}", flush=True)
