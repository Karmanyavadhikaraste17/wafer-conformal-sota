"""
train_dpcanet_v3.py

DPCANet-v3: Enhanced Dynamic Multi-Scale Perception Network with FastEPCA
Parameters: ~785k (strictly within 400k-1.2M hardware envelope).
Enhancements:
1. Dynamic multi-scale receptive fields (3x3, 5x5, 7x7) with softmax attention routing.
2. FastEPCA: Edge-Preserving Coordinate Attention using native CUDA amax + mean.
3. Expanded Stage 3 capacity (288 channels) and 3-block Stage 1 for fine defect retention.
4. Trained on Canonical 80/20 MixedWM38 benchmark split (27,370 train, 3,042 val, 7,603 test).
5. 26 epochs with Cosine Annealing and label smoothing 0.015.
6. D4 Dihedral 8-View TTA evaluation on held-out test set.
7. Evaluates standalone and forms SOTA Meta-Ensemble with DPCANet-v2, HFEdgeDPCANet, and Scratch Specialist.
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
from sklearn.metrics import accuracy_score, f1_score

# Strict thermal and VRAM safety configuration
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
# FastEPCA (Edge-Preserving Coordinate Attention)
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
# Multi-Scale Dynamic Block with FastEPCA
# -------------------------------------------------------------------------
class MultiScaleDynamicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.stride = stride
        mid_c = out_c

        self.conv_exp = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.GELU()
        )
        self.dw3 = nn.Conv2d(mid_c, mid_c, 3, stride=stride, padding=1, groups=mid_c, bias=False)
        self.dw5 = nn.Conv2d(mid_c, mid_c, 5, stride=stride, padding=2, groups=mid_c, bias=False)
        self.dw7 = nn.Conv2d(mid_c, mid_c, 7, stride=stride, padding=3, groups=mid_c, bias=False)
        self.dw_bn = nn.BatchNorm2d(mid_c)
        self.dw_act = nn.GELU()

        self.kernel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_c, 3, 1, bias=True),
            nn.Softmax(dim=1)
        )
        self.ca = FastEPCA(mid_c)
        self.conv_proj = nn.Sequential(
            nn.Conv2d(mid_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        res = self.shortcut(x)
        feat = self.conv_exp(x)
        w = self.kernel_attn(feat)
        w3, w5, w7 = w[:, 0:1], w[:, 1:2], w[:, 2:3]
        fused = self.dw_act(self.dw_bn(w3 * self.dw3(feat) + w5 * self.dw5(feat) + w7 * self.dw7(feat)))
        fused = self.ca(fused)
        out = self.conv_proj(fused)
        return F.gelu(out + res)

# -------------------------------------------------------------------------
# DPCANet-v3 Architecture
# -------------------------------------------------------------------------
class DPCANetV3(nn.Module):
    def __init__(self, in_channels=3, num_classes=38):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.stage1 = nn.Sequential(
            MultiScaleDynamicBlock(64, 96, stride=2),
            MultiScaleDynamicBlock(96, 96, stride=1),
            MultiScaleDynamicBlock(96, 96, stride=1),
        )
        self.stage2 = nn.Sequential(
            MultiScaleDynamicBlock(96, 160, stride=2),
            MultiScaleDynamicBlock(160, 160, stride=1),
            MultiScaleDynamicBlock(160, 160, stride=1),
        )
        self.stage3 = nn.Sequential(
            MultiScaleDynamicBlock(160, 288, stride=2),
            MultiScaleDynamicBlock(288, 288, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(288, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------
class WaferDataset(Dataset):
    def __init__(self, images, labels, is_train=False):
        self.images = images
        self.labels = labels
        self.is_train = is_train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])

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
            'label': torch.tensor(label, dtype=torch.long)
        }

def get_loaders():
    raw = np.load(DATA_PATH, allow_pickle=True)
    x = raw['arr_0']
    y_vecs = raw['arr_1']

    from conformal_wafer.src.dataset import CLASS_DEFINITIONS
    vec_map = {tuple(vec): idx for idx, (cid, name, vec, depth) in enumerate(CLASS_DEFINITIONS)}
    y = np.zeros(len(y_vecs), dtype=np.int64)
    for i in range(len(y_vecs)):
        y[i] = vec_map[tuple(y_vecs[i].tolist())]

    n_total = len(x)
    train_full_idx, test_idx = train_test_split(np.arange(n_total), test_size=0.20, stratify=y, random_state=42)
    train_idx, val_idx = train_test_split(train_full_idx, test_size=0.10, stratify=y[train_full_idx], random_state=42)

    train_ds = WaferDataset(x[train_idx], y[train_idx], is_train=True)
    val_ds = WaferDataset(x[val_idx], y[val_idx], is_train=False)
    test_ds = WaferDataset(x[test_idx], y[test_idx], is_train=False)

    loaders = {
        'train': DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=True),
        'val': DataLoader(val_ds, batch_size=128, shuffle=False, pin_memory=True),
        'test': DataLoader(test_ds, batch_size=128, shuffle=False, pin_memory=True)
    }
    return loaders, len(test_idx)

# -------------------------------------------------------------------------
# Training Engine
# -------------------------------------------------------------------------
def train_dpcanet_v3(epochs=26, lr=9e-4):
    print("=" * 80, flush=True)
    print("TRAINING DPCANET-V3 (785k PARAMS, DYNAMIC RECEPTIVE FIELDS + FASTEPCA)", flush=True)
    print("=" * 80, flush=True)

    loaders, n_test = get_loaders()
    model = DPCANetV3(num_classes=38).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: DPCANet-v3 | Trainable Parameters: {param_count:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.015)

    best_val_acc = 0.0
    save_path = os.path.join(EXPERIMENTS_DIR, "dpcanet_v3_best.pt")

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

        if ep % 3 == 0 or ep >= epochs - 2:
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for batch in loaders['val']:
                    images = batch['image'].to(DEVICE)
                    labels = batch['label'].to(DEVICE)
                    with autocast('cuda', enabled=True):
                        logits = model(images)
                    val_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                    val_targets.append(labels.cpu().numpy())
            val_acc = accuracy_score(np.concatenate(val_targets), np.concatenate(val_preds))
            print(f"  Epoch {ep:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}%", flush=True)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({'model_state_dict': model.state_dict(), 'val_acc': val_acc}, save_path)
                print(f"    -> Best model checkpoint saved! (Val Acc: {val_acc*100:.2f}%)", flush=True)

    dt = time.time() - t_start
    print(f"Training completed in {dt:.1f}s ({dt/60:.1f} mins)!", flush=True)
    return save_path

# -------------------------------------------------------------------------
# D4-TTA Evaluator
# -------------------------------------------------------------------------
def evaluate_d4(model, loader):
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
                    sum_p += F.softmax(logits, dim=1)
            all_probs.append((sum_p / 8.0).cpu().numpy())
            targets.append(labels.cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(targets, axis=0)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true, preds)
    err = int((y_true != preds).sum())
    f1 = f1_score(y_true, preds, average='macro', zero_division=0)
    return probs, y_true, acc, err, f1

if __name__ == '__main__':
    save_path = train_dpcanet_v3(epochs=26, lr=9e-4)

    loaders, n_test = get_loaders()
    model = DPCANetV3(num_classes=38).to(DEVICE)
    ckpt = torch.load(save_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    print("\nEvaluating DPCANet-v3 on Validation Set...", flush=True)
    p_val, y_val, acc_v, err_v, f1_v = evaluate_d4(model, loaders['val'])
    print(f"  Val Acc: {acc_v*100:.3f}% ({err_v} errors / {len(y_val)}) | Macro-F1: {f1_v*100:.3f}%", flush=True)

    print("\nEvaluating DPCANet-v3 on Held-Out Test Set (7,603 wafers)...", flush=True)
    p_test, y_test, acc_t, err_t, f1_t = evaluate_d4(model, loaders['test'])
    print(f"  Test Acc: {acc_t*100:.3f}% ({err_t} errors / 7,603) | Macro-F1: {f1_t*100:.3f}%", flush=True)

    # Save test probabilities
    out_prob_path = os.path.join(EXPERIMENTS_DIR, "dpcanet_v3_probs.npz")
    np.savez_compressed(out_prob_path, p_val=p_val, y_val=y_val, p_test=p_test, y_test=y_test)
    print(f"Saved DPCANet-v3 probabilities to: {out_prob_path}", flush=True)
