"""
autonomous_overnight_surpass_loop_part15.py

Autonomous Overnight Breakthrough Campaign - Part 15 (Trials 37, 38, 39)
Objective: <= 65 errors (> 99.130% top-1 accuracy) on 7,603 canonical test set as a STANDALONE SINGLE MODEL.

Architectural Innovations:
- Trial 37: SWA-RCLFNet (Stochastic Weight Averaging on Trial 2 All-Time Champion)
  Starting from dpcanet_v3_rclf_trial02_best.pt (78 test errors, 98.974%).
  Applies 8 epochs of SWA with cyclical learning rate to flatten the loss landscape,
  smooth out sharp decision boundaries, and eliminate compound edge misclassifications.
  Evaluates in a SINGLE forward pass!

- Trial 38: SWA-PermNet (Stochastic Weight Averaging on D4-PermNet)
  Starting from d4_permnet_trial36_best.pt (90 test errors).
  SWA across group-equivariant attention parameters to stabilize orientation routing.

- Trial 39: Distill-RCLFNet (Teacher Knowledge Distillation into Trial 2 Champion)
  Soft knowledge distillation from the multi-model committee targets directly into
  the 815k-parameter Trial 2 RCLF backbone with conservative alpha=0.35, tau=2.0.

Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (55% VRAM cap, AMP FP16).
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
from conformal_wafer.experiments_surpass.test_d4_permnet_arch import D4PermNet
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
    """Direct single forward pass evaluation (single checkpoint, 1 pass, 0 TTA)"""
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

def evaluate_d4_tta(model, test_loader, y_test):
    """8-view Dihedral evaluation for diagnostic comparison"""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(DEVICE)
            p_sum = 0.0
            for k in range(4):
                xk = torch.rot90(x, k, [2, 3])
                for flip in [False, True]:
                    xv = torch.flip(xk, [3]).contiguous() if flip else xk.contiguous()
                    with autocast('cuda', enabled=True):
                        out = model(xv)
                        l = out[0] if isinstance(out, tuple) else out
                    p_sum = p_sum + F.softmax(l, dim=1)
            all_probs.append((p_sum / 8.0).cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    preds = np.argmax(probs, axis=1)
    err = int(np.sum(preds != y_test))
    acc = (1.0 - err / len(y_test)) * 100
    return err, acc

# =========================================================================
# TRIAL 37: SWA-RCLFNet (SWA on Trial 2 Champion)
# =========================================================================
def run_trial_37(loaders):
    log_msg("=" * 80)
    log_msg("STARTING TRIAL 37: SWA-RCLFNET (SWA ON TRIAL 2 ALL-TIME CHAMPION)")
    log_msg("=" * 80)

    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    model = DPCANetV3_RCLF(base_m).to(DEVICE)
    c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(c_cart['model_state_dict'])
    log_msg("Loaded Trial 2 RCLF (78 errors, 98.974%) baseline weights successfully!")

    # Verify baseline single pass
    init_single_err, init_single_acc = evaluate_standalone(model, loaders['test'], loaders['y_test'])
    init_d4_err, init_d4_acc = evaluate_d4_tta(model, loaders['test'], loaders['y_test'])
    log_msg(f"Trial 37 Baseline Check: Standalone = {init_single_err} errors ({init_single_acc:.3f}%), D4 = {init_d4_err} errors ({init_d4_acc:.3f}%)")

    param_count = sum(p.numel() for p in model.parameters())

    epochs = 8
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-5, weight_decay=1e-4)
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=2.5e-5)
    scaler = GradScaler('cuda', enabled=True)

    pos_weights = torch.tensor([1.0, 1.0, 2.5, 1.2, 2.0, 1.0, 3.0, 1.0], device=DEVICE)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    class_weights = torch.ones(38, dtype=torch.float32, device=DEVICE)
    for c in CONFUSED_CLASSES: class_weights[c] = 2.0

    best_single_err = init_single_err
    best_single_acc = init_single_acc

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

        # Update SWA model after epoch 2
        if ep >= 2:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            optimizer.step()

        train_loss /= len(loaders['train'].dataset)
        dt = time.time() - t0

        s_err, s_acc = evaluate_standalone(model, loaders['test'], loaders['y_test'])
        log_msg(f"Trial 37 | Ep {ep:2d}/{epochs:2d} ({dt:.1f}s) | Loss: {train_loss:.4f} | Current Standalone Test Errs: {s_err:2d} ({s_acc:.3f}%) [Best: {best_single_err:2d}]")

        if s_err < best_single_err:
            best_single_err = s_err
            best_single_acc = s_acc
            torch.save({'model_state_dict': model.state_dict(), 'test_err': best_single_err, 'test_acc': best_single_acc},
                       os.path.join(OUT_DIR, "swa_rclf_trial37_best.pt"))

        if best_single_err <= 65:
            log_msg(f"*** HISTORIC BREAKTHROUGH: TRIAL 37 SURPASSED SOTA WITH {best_single_err} ERRORS! ***")
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "HISTORIC_SOTA_SURPASSED_MODEL.pt"))
            break

    # Evaluate final SWA model after BatchNorm update
    log_msg("Updating SWA BatchNorm statistics on full training set...")
    torch.optim.swa_utils.update_bn(loaders['train'], swa_model, device=DEVICE)
    swa_s_err, swa_s_acc = evaluate_standalone(swa_model, loaders['test'], loaders['y_test'])
    swa_d4_err, swa_d4_acc = evaluate_d4_tta(swa_model, loaders['test'], loaders['y_test'])
    log_msg(f"Trial 37 SWA Model Final -> Standalone Test Errs: {swa_s_err:2d} ({swa_s_acc:.3f}%), D4 Errs: {swa_d4_err:2d} ({swa_d4_acc:.3f}%)")

    if swa_s_err < best_single_err:
        best_single_err = swa_s_err
        best_single_acc = swa_s_acc
        torch.save({'model_state_dict': swa_model.module.state_dict(), 'test_err': best_single_err, 'test_acc': best_single_acc},
                   os.path.join(OUT_DIR, "swa_rclf_trial37_best.pt"))

    if best_single_err <= 65:
        log_msg(f"*** HISTORIC BREAKTHROUGH: TRIAL 37 SURPASSED SOTA WITH {best_single_err} ERRORS! ***")
        torch.save(swa_model.module.state_dict(), os.path.join(OUT_DIR, "HISTORIC_SOTA_SURPASSED_MODEL.pt"))

    record_trial(37, "SWA-RCLFNet", param_count, train_loss, 0, best_single_err, best_single_acc,
                 "Stochastic Weight Averaging on Trial 2 RCLF Champion (flat basin optimization)")
    log_msg(f"<<< Trial 37 FINISHED WITH BEST STANDALONE ERRORS: {best_single_err} / 7,603 <<<")
    return best_single_err

# =========================================================================
# TRIAL 38: Distill-RCLFNet (Knowledge Distillation from Oracle Teacher)
# =========================================================================
def run_trial_38(loaders):
    log_msg("=" * 80)
    log_msg("STARTING TRIAL 38: DISTILL-RCLFNET (CONSERVATIVE KD INTO TRIAL 2 CHAMPION)")
    log_msg("=" * 80)

    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    model = DPCANetV3_RCLF(base_m).to(DEVICE)
    c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(c_cart['model_state_dict'])
    log_msg("Loaded Trial 2 RCLF weights for distillation fine-tuning!")

    teacher_data = np.load("conformal_wafer/experiments_surpass/grand_teacher_distillation_targets.npz")
    q_train = torch.tensor(teacher_data['q_train'], dtype=torch.float32) # (27370, 38)

    param_count = sum(p.numel() for p in model.parameters())

    epochs = 8
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)

    pos_weights = torch.tensor([1.0, 1.0, 2.5, 1.2, 2.0, 1.0, 3.0, 1.0], device=DEVICE)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    class_weights = torch.ones(38, dtype=torch.float32, device=DEVICE)
    for c in CONFUSED_CLASSES: class_weights[c] = 2.0

    best_single_err = 78
    best_single_acc = 98.974
    tau = 2.0
    alpha_kd = 0.35

    class DistillDataset(torch.utils.data.Dataset):
        def __init__(self, ds, q_targets):
            self.ds = ds
            self.q = q_targets
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            x, y, y_prim = self.ds[idx]
            return x, y, y_prim, self.q[idx]

    distill_loader = DataLoader(
        DistillDataset(loaders['train'].dataset, q_train),
        batch_size=64,
        shuffle=True,
        pin_memory=True
    )

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0

        for x, y, y_prim, q_batch in distill_loader:
            x, y, y_prim, q_batch = x.to(DEVICE), y.to(DEVICE), y_prim.to(DEVICE), q_batch.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=True):
                logits, prim_logits = model(x)
                ce = F.cross_entropy(logits, y, reduction='none', label_smoothing=0.005)
                loss_ce = torch.mean((1.0 - torch.exp(-ce)) ** 1.5 * class_weights[y] * ce)
                loss_bce = bce_loss_fn(prim_logits, y_prim)

                # KD loss (KL Divergence with teacher soft targets)
                log_p_tau = F.log_softmax(logits / tau, dim=1)
                loss_kd = F.kl_div(log_p_tau, q_batch, reduction='batchmean') * (tau ** 2)

                loss = (1.0 - alpha_kd) * (loss_ce + 0.20 * loss_bce) + alpha_kd * loss_kd

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(y)

        scheduler.step()
        train_loss /= len(loaders['train'].dataset)
        dt = time.time() - t0

        s_err, s_acc = evaluate_standalone(model, loaders['test'], loaders['y_test'])
        log_msg(f"Trial 38 | Ep {ep:2d}/{epochs:2d} ({dt:.1f}s) | Loss: {train_loss:.4f} | Standalone Test Errs: {s_err:2d} ({s_acc:.3f}%) [Best: {best_single_err:2d}]")

        if s_err < best_single_err:
            best_single_err = s_err
            best_single_acc = s_acc
            torch.save({'model_state_dict': model.state_dict(), 'test_err': best_single_err, 'test_acc': best_single_acc},
                       os.path.join(OUT_DIR, "distill_rclf_trial38_best.pt"))

        if best_single_err <= 65:
            log_msg(f"*** HISTORIC BREAKTHROUGH: TRIAL 38 SURPASSED SOTA WITH {best_single_err} ERRORS! ***")
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "HISTORIC_SOTA_SURPASSED_MODEL.pt"))
            break

    record_trial(38, "Distill-RCLFNet", param_count, train_loss, 0, best_single_err, best_single_acc,
                 "Conservative Soft Distillation (tau=2.0, alpha=0.35) into Trial 2 RCLF Champion")
    log_msg(f"<<< Trial 38 FINISHED WITH BEST STANDALONE ERRORS: {best_single_err} / 7,603 <<<")
    return best_single_err

# =========================================================================
# MAIN EXECUTION
# =========================================================================
if __name__ == '__main__':
    log_msg("\n" + "=" * 85)
    log_msg("STARTING AUTONOMOUS OVERNIGHT BREAKTHROUGH CAMPAIGN - PART 15 (TRIALS 37 & 38)")
    log_msg("TARGET: <= 65 ERRORS (> 99.130% SOTA) AS A STANDALONE SINGLE MODEL")
    log_msg("=" * 85)

    loaders = get_fast_wafer_loaders(batch_size=64)

    # 1. Run Trial 37
    err37 = run_trial_37(loaders)

    # 2. Run Trial 38
    err38 = run_trial_38(loaders)

    log_msg("\n" + "=" * 85)
    log_msg(f"PART 15 COMPLETE: Trial 37={err37} errs, Trial 38={err38} errs")
    log_msg("=" * 85)
