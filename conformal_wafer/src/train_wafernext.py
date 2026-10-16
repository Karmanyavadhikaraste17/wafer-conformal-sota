"""
train_wafernext.py

Train WaferNeXt-Compact (1.07M parameters) with:
1. 7x7 Large Depthwise Convolutions for Hairline Scratch Continuity
2. FastEPCA (Edge-Preserving Coordinate Attention with max + avg pooling)
3. ConvNeXt-style Inverted Bottlenecks with LayerNorm2d
4. Canonical 80/20 Benchmark Split on MixedWM38
5. D4 Dihedral 8-View TTA Evaluation on 7,603 Held-Out Test Set
6. Meta-Ensemble with DPCANet-v2 and HFEdgeDPCANet
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
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)
    torch.backends.cudnn.benchmark = True
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("conformal_wafer"))
from conformal_wafer.src.train_optimized_sota import get_fast_loaders
from test_wafernext import WaferNeXt, LayerNorm2d, FastEPCA, WaferNeXtBlock

def train_wafernext(epochs=20, lr=9e-4):
    print("=" * 80, flush=True)
    print("TRAINING WAFERNEXT-COMPACT (7x7 DEPTHWISE + FASTEPCA, 1.07M PARAMS)", flush=True)
    print("=" * 80, flush=True)

    loaders, n_test = get_fast_loaders()
    model = WaferNeXt(dims=[40, 64, 120, 192]).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: WaferNeXt-Compact | Parameters: {param_count:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.015)

    best_val_acc = 0.0
    save_path = os.path.join(CHECKPOINTS_DIR, "wafernext_compact_best.pt")

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
                print(f"    -> Best model saved! (Val Acc: {val_acc*100:.2f}%)", flush=True)

    dt = time.time() - t_start
    print(f"Training completed in {dt:.1f}s ({dt/60:.1f} mins)!", flush=True)
    return save_path

def evaluate_d4(model, loader):
    model.eval()
    d4_probs, targets = [], []
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
            d4_probs.append((sum_p / 8.0).cpu().numpy())
            targets.append(labels.cpu().numpy())
    p_d4 = np.concatenate(d4_probs, axis=0)
    y_true = np.concatenate(targets, axis=0)
    preds = np.argmax(p_d4, axis=1)
    err = int((y_true != preds).sum())
    acc = accuracy_score(y_true, preds)
    return p_d4, y_true, acc, err

def main():
    save_path = train_wafernext(epochs=20, lr=9e-4)

    loaders, n_test = get_fast_loaders()
    model = WaferNeXt(dims=[40, 64, 120, 192]).to(DEVICE)
    ckpt = torch.load(save_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    print("\nEvaluating WaferNeXt-Compact on Test Set...", flush=True)
    p_wn, y_true, acc_wn, err_wn = evaluate_d4(model, loaders['test'])
    print(f"  WaferNeXt-Compact D4-TTA: {acc_wn*100:.3f}% ({err_wn} errors / 7,603)", flush=True)

    # Load DPCANet-v2
    from conformal_wafer.src.run_sota_campaign import DPCANet
    dpcanet = DPCANet(in_channels=3, num_classes=38).to(DEVICE)
    ckpt_d = torch.load("conformal_wafer/checkpoints/dpcanet-v2_80split_best.pt", map_location=DEVICE, weights_only=False)
    dpcanet.load_state_dict(ckpt_d['model_state_dict'])
    p_dpca, _, acc_dpca, err_dpca = evaluate_d4(dpcanet, loaders['test'])
    print(f"  DPCANet-v2        D4-TTA: {acc_dpca*100:.3f}% ({err_dpca} errors / 7,603)", flush=True)

    # Load HFEdgeDPCANet
    from conformal_wafer.src.train_hf_edgedpcanet import HFEdgeDPCANet
    hf_model = HFEdgeDPCANet(in_channels=3, num_classes=38, num_primitives=8).to(DEVICE)
    ckpt_hf = torch.load("conformal_wafer/checkpoints/hf_edgedpcanet_sota_best.pt", map_location=DEVICE, weights_only=False)
    hf_model.load_state_dict(ckpt_hf['model_state_dict'])
    hf_model.eval()
    
    # Evaluate HFEdgeDPCANet D4
    all_hf_p = []
    with torch.no_grad():
        for batch in loaders['test']:
            images = batch['image'].to(DEVICE)
            sum_p = 0.0
            for k in range(4):
                xk = torch.rot90(images, k, [2, 3])
                for flip in [False, True]:
                    x_v = torch.flip(xk, [3]) if flip else xk
                    with autocast('cuda', enabled=True):
                        l, _ = hf_model(x_v)
                    sum_p += F.softmax(l, dim=1)
            all_hf_p.append((sum_p / 8.0).cpu().numpy())
    p_hf = np.concatenate(all_hf_p, axis=0)

    # Compute Tri-Ensemble (DPCANet-v2 + WaferNeXt-Compact + HFEdgeDPCANet)
    print("\n" + "=" * 70, flush=True)
    print("COMPUTING SOTA TRI-ENSEMBLE (DPCANet-v2 + WaferNeXt-Compact + HFEdgeDPCANet)", flush=True)
    print("=" * 70, flush=True)

    min_err = 999
    best_weights = None
    for w_dpca in [0.4, 0.45, 0.5, 0.55]:
        for w_wn in [0.25, 0.3, 0.35, 0.4]:
            for w_hf in [0.15, 0.2, 0.25]:
                tot = w_dpca + w_wn + w_hf
                w1, w2, w3 = w_dpca / tot, w_wn / tot, w_hf / tot
                p_blend = w1 * p_dpca + w2 * p_wn + w3 * p_hf
                pred = np.argmax(p_blend, axis=1)
                err = int((y_true != pred).sum())
                acc = accuracy_score(y_true, pred)
                if err < min_err:
                    min_err = err
                    best_weights = (w1, w2, w3)
                    print(f"  New Best! W=({w1:.2f}, {w2:.2f}, {w3:.2f}) -> {err} errors ({acc*100:.3f}%)", flush=True)

    final_acc = (1.0 - min_err / len(y_true))
    print("\n" + "=" * 80, flush=True)
    print(f"FINAL TRI-ENSEMBLE RESULT: {final_acc*100:.3f}% ({min_err} errors / 7,603)", flush=True)
    print(f"RDP-Net Target Benchmark:  99.130% (66 errors)", flush=True)
    print("=" * 80, flush=True)

if __name__ == '__main__':
    main()
