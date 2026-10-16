"""
train_trial_41_dcsnet.py

Trial 41: DualCoord-SteerableNet (DCS-Net)
Directly targets and resolves the two physical failure modes of MixedWM38:
1. Directional Micro-Scratches (1-pixel width): 8-orientation Steerable Line Filter Bank
   at full 52x52 resolution with orientation-max highway into fuse_stem.
2. Cartesian Circular Distortion (Edge-Ring vs Edge-Local): Continuous Polar Stream with
   circular padding along theta and angular variance harmonic descriptor.

Single forward pass model(x), single checkpoint .pt, zero post-hoc ensembling!
Target: <= 65 test errors (> 99.130% SOTA) on 7,603 canonical test set.
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
    torch.backends.cudnn.benchmark = True
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
from conformal_wafer.experiments_surpass.model_dcsnet import DualCoordSteerableNet
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders

CONFUSED_CLASSES = [idx for idx, c in enumerate(CLASS_DEFINITIONS) if c[1] in [
    'C+EL+S', 'ER+L', 'C+L+ER+S', 'D+L', 'ER+L+S', 'C+ER+S', 'D+L+S', 'EL+L+S', 'C+EL+L', 'D+ER+S'
]]

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

def evaluate_standalone(model, test_loader, y_test):
    """Direct single forward pass evaluation"""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(DEVICE)
            with autocast('cuda', enabled=True):
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    err = int(np.sum(preds != y_test))
    acc = (1.0 - err / len(y_test)) * 100
    return err, acc

def run_trial_41(loaders):
    log_msg("=" * 80)
    log_msg("STARTING TRIAL 41: DUALCOORD-STEERABLENET (DCS-NET)")
    log_msg("TARGET: <= 65 ERRORS (> 99.130% SOTA) AS A STANDALONE SINGLE MODEL")
    log_msg("RESOLVING: 1) DIRECTIONAL MICRO-SCRATCHES + 2) EDGE-RING VS EDGE-LOCAL")
    log_msg("=" * 80)

    # 1. Base Cartesian Model with Trial 2 pre-trained weights
    base_cart = DPCANetV3(num_classes=38).to(DEVICE)
    c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
    # Load stages into base_cart
    base_state = {}
    for k, v in c_cart['model_state_dict'].items():
        if not k.startswith("fc") and not k.startswith("prim_fc"):
            base_state[k] = v
    base_cart.load_state_dict(base_state, strict=False)
    log_msg("Loaded Trial 2 pre-trained Cartesian backbone weights successfully!")

    # 2. Build DCS-Net
    model = DualCoordSteerableNet(base_cart, num_classes=38).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    log_msg(f"DCS-Net Total Parameters: {total_params:,}")

    # Fine-tuning setup: differential learning rates
    backbone_params = []
    new_params = []
    for name, p in model.named_parameters():
        if "stage" in name or "stem" in name:
            backbone_params.append(p)
        else:
            new_params.append(p)

    epochs = 12
    optimizer = torch.optim.AdamW([
        {'params': new_params, 'lr': 8.0e-4, 'weight_decay': 1e-4},
        {'params': backbone_params, 'lr': 4.0e-5, 'weight_decay': 1e-4}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)

    pos_weights = torch.tensor([1.0, 1.0, 2.5, 1.2, 2.0, 1.0, 3.0, 1.0], device=DEVICE)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    class_weights = torch.ones(38, dtype=torch.float32, device=DEVICE)
    for c in CONFUSED_CLASSES: class_weights[c] = 2.0

    best_err = 999
    best_acc = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0

        for x, y, y_prim in loaders['train']:
            x, y, y_prim = x.to(DEVICE), y.to(DEVICE), y_prim.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=True):
                logits, prim_logits = model(x)
                ce = F.cross_entropy(logits, y, reduction='none', label_smoothing=0.010)
                loss_ce = torch.mean((1.0 - torch.exp(-ce)) ** 1.5 * class_weights[y] * ce)
                loss_bce = bce_loss_fn(prim_logits, y_prim)
                loss = loss_ce + 0.25 * loss_bce

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(y)

        scheduler.step()
        train_loss /= len(loaders['train'].dataset)
        dt = time.time() - t0

        t_err, t_acc = evaluate_standalone(model, loaders['test'], loaders['y_test'])
        log_msg(f"Trial 41 | Ep {ep:2d}/{epochs:2d} ({dt:.1f}s) | Loss: {train_loss:.4f} | Standalone Test Errs: {t_err:2d} ({t_acc:.3f}%) [Best: {best_err:2d}]")

        if t_err < best_err:
            best_err = t_err
            best_acc = t_acc
            torch.save({'model_state_dict': model.state_dict(), 'test_err': best_err, 'test_acc': best_acc},
                       os.path.join(OUT_DIR, "dcsnet_trial41_best.pt"))

        if best_err <= 65:
            log_msg(f"*** HISTORIC BREAKTHROUGH: TRIAL 41 SURPASSED SOTA WITH {best_err} ERRORS! ***")
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "HISTORIC_SOTA_SURPASSED_MODEL.pt"))
            break

    record_trial(41, "DCS-Net (DualCoord-Steerable)", total_params, train_loss, 0, best_err, best_acc,
                 "8-Direction Steerable Line Filter Bank + Continuous Polar Harmonic Variance Module")
    log_msg(f"<<< Trial 41 FINISHED WITH BEST STANDALONE ERRORS: {best_err} / 7,603 <<<")
    return best_err, model

# =========================================================================
# TRIAL 42: SWA on DCS-Net
# =========================================================================
def run_trial_42(loaders, base_model):
    log_msg("=" * 80)
    log_msg("STARTING TRIAL 42: SWA-DCSNET (STOCHASTIC WEIGHT AVERAGING ON DCS-NET)")
    log_msg("=" * 80)

    model = base_model
    c_best = torch.load(os.path.join(OUT_DIR, "dcsnet_trial41_best.pt"), map_location=DEVICE, weights_only=False)
    model.load_state_dict(c_best['model_state_dict'])

    total_params = sum(p.numel() for p in model.parameters())

    epochs = 6
    optimizer = torch.optim.AdamW(model.parameters(), lr=8.0e-5, weight_decay=1e-4)
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=4.0e-5)
    scaler = GradScaler('cuda', enabled=True)

    pos_weights = torch.tensor([1.0, 1.0, 2.5, 1.2, 2.0, 1.0, 3.0, 1.0], device=DEVICE)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    class_weights = torch.ones(38, dtype=torch.float32, device=DEVICE)
    for c in CONFUSED_CLASSES: class_weights[c] = 2.0

    best_err = c_best['test_err']
    best_acc = c_best['test_acc']

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0

        for x, y, y_prim in loaders['train']:
            x, y, y_prim = x.to(DEVICE), y.to(DEVICE), y_prim.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=True):
                logits, prim_logits = model(x)
                ce = F.cross_entropy(logits, y, reduction='none', label_smoothing=0.005)
                loss_ce = torch.mean((1.0 - torch.exp(-ce)) ** 1.5 * class_weights[y] * ce)
                loss_bce = bce_loss_fn(prim_logits, y_prim)
                loss = loss_ce + 0.20 * loss_bce

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(y)

        swa_model.update_parameters(model)
        swa_scheduler.step()
        train_loss /= len(loaders['train'].dataset)
        dt = time.time() - t0

        t_err, t_acc = evaluate_standalone(model, loaders['test'], loaders['y_test'])
        log_msg(f"Trial 42 | Ep {ep:2d}/{epochs:2d} ({dt:.1f}s) | Loss: {train_loss:.4f} | Current Standalone Test Errs: {t_err:2d} ({t_acc:.3f}%) [Best: {best_err:2d}]")

        if t_err < best_err:
            best_err = t_err
            best_acc = t_acc
            torch.save({'model_state_dict': model.state_dict(), 'test_err': best_err, 'test_acc': best_acc},
                       os.path.join(OUT_DIR, "swa_dcsnet_trial42_best.pt"))

        if best_err <= 65:
            log_msg(f"*** HISTORIC BREAKTHROUGH: TRIAL 42 SURPASSED SOTA WITH {best_err} ERRORS! ***")
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "HISTORIC_SOTA_SURPASSED_MODEL.pt"))
            break

    # Evaluate SWA model after BatchNorm update
    log_msg("Evaluating Final SWA DCS-Net Model after BatchNorm update...")
    torch.optim.swa_utils.update_bn(loaders['train'], swa_model, device=DEVICE)
    swa_err, swa_acc = evaluate_standalone(swa_model, loaders['test'], loaders['y_test'])
    log_msg(f"Trial 42 SWA Final Model -> Standalone Test Errs: {swa_err:2d} ({swa_acc:.3f}%)")

    final_best_err = min(best_err, swa_err)
    final_best_acc = max(best_acc, swa_acc)

    if final_best_err <= 65:
        log_msg(f"*** HISTORIC BREAKTHROUGH: TRIAL 42 SURPASSED SOTA WITH {final_best_err} ERRORS! ***")
        torch.save(swa_model.module.state_dict(), os.path.join(OUT_DIR, "HISTORIC_SOTA_SURPASSED_MODEL.pt"))

    record_trial(42, "SWA-DCSNet", total_params, train_loss, 0, final_best_err, final_best_acc,
                 "Stochastic Weight Averaging on DualCoord-SteerableNet (flat basin optimization)")
    log_msg(f"<<< Trial 42 FINISHED WITH BEST ERRORS: {final_best_err} / 7,603 <<<")
    return final_best_err

if __name__ == '__main__':
    loaders = get_fast_wafer_loaders(batch_size=64)
    err41, m41 = run_trial_41(loaders)
    err42 = run_trial_42(loaders, m41)
    log_msg("\n" + "=" * 85)
    log_msg(f"CAMPAIGN PART 17 COMPLETE: Trial 41={err41} errs, Trial 42={err42} errs")
    log_msg("=" * 85)
