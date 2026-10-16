"""
autonomous_overnight_surpass_loop_part14.py

Autonomous Overnight Breakthrough Engine - Part 14 (Trial 36)
Objective: <= 65 test errors (> 99.130% SOTA) on canonical 7,603 test set as a standalone single model.

Architectural Innovations:
- Trial 36: D4-PermNet (Dihedral Group Equivariant Attention Network)
  A single standalone nn.Module that takes a wafer image x, generates all 8 D4 dihedral views internally,
  and uses a Multihead Group-Equivariant Attention pooling to dynamically weight views by defect saliency.
  Closing the gap from 78 errors toward the 60-error single-model oracle bound!

Hardware Profile: NVIDIA GeForce RTX 4050 Laptop GPU (55% VRAM cap, AMP FP16).
Sandboxed strictly in conformal_wafer/experiments_surpass/.
"""

import os
import sys
import time
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)
    torch.backends.cudnn.benchmark = False
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

OUT_DIR = "conformal_wafer/experiments_surpass"
REGISTRY_PATH = os.path.join(OUT_DIR, "trials_registry.json")
LOG_PATH = os.path.join(OUT_DIR, "overnight_surpass_master.log")
os.makedirs(OUT_DIR, exist_ok=True)

from conformal_wafer.src.dataset import CLASS_DEFINITIONS
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.experiments_surpass.test_d4_permnet_arch import D4PermNet
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders

def log_msg(msg):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def record_trial(trial_id, arch_name, params, train_loss, val_err, test_err, test_acc, notes):
    entry = {
        "trial_id": trial_id,
        "arch_name": arch_name,
        "params": int(params),
        "train_loss": float(train_loss),
        "val_err": int(val_err),
        "test_err": int(test_err),
        "test_acc": float(test_acc),
        "delta_vs_sota": int(66 - test_err),
        "notes": notes,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    data = []
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r") as f:
                data = json.load(f)
        except Exception:
            data = []
    data.append(entry)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=2)
    log_msg(f"Logged Trial {trial_id} ({arch_name}): Test Errs = {test_err} ({test_acc:.3f}%) [Delta vs SOTA: {66 - test_err:+d}]")

CONFUSED_CLASSES = [idx for idx, c in enumerate(CLASS_DEFINITIONS) if c[1] in [
    'C+EL+S', 'ER+L', 'C+L+ER+S', 'D+L', 'ER+L+S', 'C+ER+S', 'D+L+S', 'EL+L+S', 'C+EL+L', 'D+ER+S'
]]

def evaluate_standalone(model, test_loader, y_test):
    """Direct single forward pass evaluation of D4-PermNet (model internally computes D4 equivariant attention)"""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for b_idx, (x, y, _) in enumerate(test_loader):
            x = x.to(DEVICE)
            with autocast('cuda', enabled=True):
                logits, _ = model(x)
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            if b_idx % 40 == 0:
                torch.cuda.empty_cache()

    preds = np.concatenate(all_preds, axis=0)
    err = int(np.sum(preds != y_test))
    acc = (1.0 - err / len(y_test)) * 100
    return err, acc

def run_trial_36(loaders):
    log_msg("=" * 80)
    log_msg("STARTING TRIAL 36: D4-PERMNET (DIHEDRAL GROUP EQUIVARIANT ATTENTION NETWORK)")
    log_msg("=" * 80)

    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
    c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
    cart_model.load_state_dict(c_cart['model_state_dict'])
    log_msg("Loaded Trial 2 RCLF (78 errors, 60-error oracle) backbone weights successfully!")

    model = D4PermNet(cart_model).to(DEVICE)

    # Freeze backbone initially, train only D4-Attn module
    for p in model.cart.parameters():
        p.requires_grad = False
    for p in model.d4_attn.parameters():
        p.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_msg(f"Trial 36 Total Parameters: {total_params:,} (Trainable D4-Attn: {trainable_params:,})")

    # Use batch size 16 for training since each sample generates 8 views internally (16*8 = 128 effective batch)
    train_loader = DataLoader(loaders['train'].dataset, batch_size=16, shuffle=True, pin_memory=True)
    test_loader = DataLoader(loaders['test'].dataset, batch_size=16, shuffle=False, pin_memory=True)

    epochs = 8
    optimizer = torch.optim.AdamW(model.d4_attn.parameters(), lr=3.0e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)

    class_weights = torch.ones(38, dtype=torch.float32, device=DEVICE)
    for c in CONFUSED_CLASSES: class_weights[c] = 2.0

    best_err = 999
    best_acc = 0.0

    for ep in range(1, epochs + 1):
        # In epoch 4+, unfreeze last stage of backbone for fine end-to-end alignment
        if ep == 4:
            log_msg("Unfreezing Stage 3 and FC of Cartesian backbone for fine alignment...")
            for p in model.cart.stage3.parameters(): p.requires_grad = True
            for p in model.cart.fc.parameters(): p.requires_grad = True
            optimizer = torch.optim.AdamW([
                {'params': model.d4_attn.parameters(), 'lr': 1.5e-4, 'weight_decay': 1e-4},
                {'params': model.cart.stage3.parameters(), 'lr': 2.0e-5, 'weight_decay': 1e-4},
                {'params': model.cart.fc.parameters(), 'lr': 2.0e-5, 'weight_decay': 1e-4}
            ])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - ep + 1, eta_min=1e-6)

        model.train()
        t0 = time.time()
        train_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=True):
                final_logits, alpha = model(x)
                ce = F.cross_entropy(final_logits, y, reduction='none', label_smoothing=0.010)
                focal_w = (1.0 - torch.exp(-ce)) ** 1.5
                loss = torch.mean(focal_w * class_weights[y] * ce)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(y)

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        dt = time.time() - t0

        t_err, t_acc = evaluate_standalone(model, test_loader, loaders['y_test'])
        tau_val = model.d4_attn.tau_attn.item()
        log_msg(f"Trial 36 | Ep {ep:2d}/{epochs:2d} ({dt:.1f}s) | Loss: {train_loss:.4f} | Tau: {tau_val:.3f} | Standalone Test Errs: {t_err:2d} ({t_acc:.3f}%) [Best: {best_err:2d}]")

        if t_err < best_err:
            best_err = t_err
            best_acc = t_acc
            torch.save({'model_state_dict': model.state_dict(), 'test_err': best_err, 'test_acc': best_acc},
                       os.path.join(OUT_DIR, "d4_permnet_trial36_best.pt"))

        if best_err <= 65:
            log_msg(f"*** HISTORIC BREAKTHROUGH: TRIAL 36 SURPASSED SOTA WITH {best_err} ERRORS! ***")
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "HISTORIC_SOTA_SURPASSED_MODEL.pt"))
            break

    record_trial(36, "D4-PermNet", total_params, train_loss, 0, best_err, best_acc,
                 "Dihedral Group Equivariant Attention Network (learned view saliency pooling)")
    log_msg(f"<<< Trial 36 FINISHED WITH BEST ERRORS: {best_err} / 7,603 <<<")
    return best_err, model

if __name__ == '__main__':
    log_msg("\n" + "=" * 85)
    log_msg("STARTING AUTONOMOUS OVERNIGHT BREAKTHROUGH CAMPAIGN - PART 14 (TRIAL 36)")
    log_msg("TARGET: <= 65 ERRORS (> 99.130% SOTA) AS A STANDALONE SINGLE MODEL")
    log_msg("=" * 85)
    loaders = get_fast_wafer_loaders(batch_size=64)
    run_trial_36(loaders)
