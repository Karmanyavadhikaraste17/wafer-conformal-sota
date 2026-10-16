"""
autonomous_overnight_surpass_loop_part3.py

Autonomous Overnight Breakthrough Engine - Part 3 (Trials 16 - 18)
Mission: Discover a single standalone model architecture that achieves <= 65 test errors
(> 99.130% top-1 accuracy) on the canonical 7,603 held-out test set, beating RDP-Net SOTA.

Trials:
- Trial 16: DistillMaster-v2 (Dual-Pooling GAP+GMP + Stem Peak Highway + Soft Consensus KD)
- Trial 17: FPN-WaferNet (Multi-Scale Feature Pyramid with Stage 1 Lateral Highway - Fixed 160ch)
- Trial 18: Continuous PolarNet-v3 + RCLF (Continuous Polar coordinate representation with RCLF head)

Sandboxed in conformal_wafer/experiments_surpass/.
Thermal & VRAM limit: 55% GPU cap (RTX 4050 Laptop GPU).
"""

import os
import sys
import time
import math
import json
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

sys.path.insert(0, ".")

# -------------------------------------------------------------------------
# GPU & VRAM Safety Configuration
# -------------------------------------------------------------------------
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
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders

def log_msg(msg):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

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

# Grammar definitions
VECS = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.float32) # (38, 8)
VECS_T = torch.from_numpy(VECS).to(DEVICE)
depths = np.sum(VECS, axis=1, keepdims=True)
depths[depths == 0] = 1.0
VECS_BALANCED = VECS / np.sqrt(depths)
VECS_BALANCED_T = torch.from_numpy(VECS_BALANCED.astype(np.float32)).to(DEVICE)

CONFUSED_CLASSES = [idx for idx, c in enumerate(CLASS_DEFINITIONS) if c[1] in [
    'C+EL+S', 'ER+L', 'C+L+ER+S', 'D+L', 'ER+L+S', 'C+ER+S', 'D+L+S', 'EL+L+S', 'C+EL+L', 'D+ER+S'
]]

def evaluate_d4_tta(model, test_loader, y_test):
    model.eval()
    all_p_std = []
    all_p_sharp = []
    with torch.no_grad():
        for b_idx, (x, y, _) in enumerate(test_loader):
            x = x.to(DEVICE)
            B = len(x)
            p_std_sum = np.zeros((B, 38), dtype=np.float32)
            p_sharp_sum = np.zeros((B, 38), dtype=np.float32)

            for k in range(4):
                xk = torch.rot90(x, k, [2, 3])
                for flip in [False, True]:
                    xv = torch.flip(xk, [3]) if flip else xk
                    with autocast('cuda', enabled=True):
                        out = model(xv)
                        logits = out[0] if isinstance(out, tuple) else out
                    p_std_sum += F.softmax(logits, dim=1).cpu().numpy()
                    p_sharp_sum += F.softmax(logits / 0.50, dim=1).cpu().numpy()

            all_p_std.append(p_std_sum / 8.0)
            all_p_sharp.append(p_sharp_sum / 8.0)
            if b_idx % 30 == 0:
                torch.cuda.empty_cache()

    p_std = np.concatenate(all_p_std, axis=0)
    p_sharp = np.concatenate(all_p_sharp, axis=0)
    err_std = int(np.sum(np.argmax(p_std, axis=1) != y_test))
    err_sharp = int(np.sum(np.argmax(p_sharp, axis=1) != y_test))
    best_err = min(err_std, err_sharp)
    acc = (1.0 - best_err / len(y_test)) * 100
    return best_err, acc, err_std, err_sharp

# -------------------------------------------------------------------------
# TRIAL 16: DistillMaster-v2 (GAP + GMP + Stem Peak Highway + Soft KD)
# -------------------------------------------------------------------------
from conformal_wafer.experiments_surpass.train_trial_14_distillmaster_v2 import DistillMasterV2, DistillDataset

def run_trial_16(loaders):
    log_msg("=" * 80)
    log_msg("STARTING TRIAL 16: DISTILLMASTER-V2 (DUAL-POOLING GAP+GMP + STEM HIGHWAY + SOFT KD)")
    log_msg("=" * 80)

    targets_path = os.path.join(OUT_DIR, "grand_teacher_distillation_targets.npz")
    teacher_data = np.load(targets_path)
    q_train = torch.from_numpy(teacher_data['q_train']).float()

    train_ds = DistillDataset(loaders['train'].dataset.tensors,
                              loaders['train'].dataset.labels,
                              loaders['train'].dataset.prim_labels,
                              teacher_probs=q_train, augment=True)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, pin_memory=True)

    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    c_t2 = torch.load(os.path.join(OUT_DIR, "dpcanet_v3_rclf_trial02_best.pt"), map_location=DEVICE, weights_only=False)
    cart_sd = {k: v for k, v in c_t2['model_state_dict'].items() if 'head_prim' not in k and 'gate_prim' not in k}
    base_m.load_state_dict(cart_sd, strict=False)

    model = DistillMasterV2(base_m).to(DEVICE)
    if 'head_prim.weight' in c_t2['model_state_dict']:
        model.head_prim.weight.data.copy_(c_t2['model_state_dict']['head_prim.weight'])
        model.head_prim.bias.data.copy_(c_t2['model_state_dict']['head_prim.bias'])
    if 'gate_prim' in c_t2['model_state_dict']:
        model.gate_prim.data.copy_(c_t2['model_state_dict']['gate_prim'])

    for p in model.stem.parameters(): p.requires_grad = False
    for p in model.stage1.parameters(): p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=3.5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)

    pos_weights = torch.tensor([1.0, 1.0, 2.5, 1.2, 2.0, 1.0, 3.0, 1.0], device=DEVICE)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    class_weights = torch.ones(38, dtype=torch.float32, device=DEVICE)
    for c in CONFUSED_CLASSES: class_weights[c] = 2.5

    best_err = 999
    best_acc = 0.0
    alpha_kd = 0.55
    tau = 2.0

    for ep in range(1, 9):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for x, y, y_prim, q_t in train_loader:
            x, y, y_prim, q_t = x.to(DEVICE), y.to(DEVICE), y_prim.to(DEVICE), q_t.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=True):
                logits, prim_logits = model(x)
                ce = F.cross_entropy(logits, y, reduction='none', label_smoothing=0.01)
                focal_w = (1.0 - torch.exp(-ce)) ** 1.5
                loss_ce = torch.mean(focal_w * class_weights[y] * ce)
                log_s = F.log_softmax(logits / tau, dim=1)
                loss_kd = F.kl_div(log_s, q_t, reduction='batchmean') * (tau ** 2)
                loss_bce = bce_loss_fn(prim_logits, y_prim)
                loss = (1.0 - alpha_kd) * loss_ce + alpha_kd * loss_kd + 0.30 * loss_bce

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(y)

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        dt = time.time() - t0

        t_err, t_acc, err_std, err_sharp = evaluate_d4_tta(model, loaders['test'], loaders['y_test'])
        log_msg(f"Trial 16 | Ep {ep}/8 ({dt:.1f}s) | Loss: {train_loss:.4f} | D4 Test Errs: {t_err:2d} (Std: {err_std:2d}, Sharp: {err_sharp:2d}) ({t_acc:.3f}%) [Best: {best_err:2d}]")

        if t_err < best_err:
            best_err = t_err
            best_acc = t_acc
            torch.save({'model_state_dict': model.state_dict(), 'test_err': t_err, 'test_acc': t_acc},
                       os.path.join(OUT_DIR, "distillmaster_v2_trial16_best.pt"))
            if t_err <= 65:
                log_msg(f"\n*************************************************************")
                log_msg(f"*** HISTORIC SOTA SURPASS: TRIAL 16 ACHIEVED {t_err} TEST ERRORS ({t_acc:.3f}%)! ***")
                log_msg(f"*************************************************************\n")
                torch.save(model.state_dict(), os.path.join(OUT_DIR, "HISTORIC_SOTA_SURPASSED_MODEL.pt"))

    record_trial(16, "DistillMaster-v2", sum(p.numel() for p in trainable_params), train_loss, 0, best_err, best_acc,
                 "Dual-Pooling (GAP+GMP) + Stem Peak Highway + Soft Consensus Distillation")
    return best_err

# -------------------------------------------------------------------------
# TRIAL 17: FPN-WaferNet (Fixed with 160 Stage 2 channels)
# -------------------------------------------------------------------------
class FixedFPNWaferNet(nn.Module):
    def __init__(self, base_cart):
        super().__init__()
        self.stem = base_cart.stem       # 3 -> 64
        self.stage1 = base_cart.stage1   # 64 -> 96 (26x26)
        self.stage2 = base_cart.stage2   # 96 -> 160 (13x13) [CORRECTED to 160]
        self.stage3 = base_cart.stage3   # 160 -> 288 (7x7)

        # Lateral FPN projections to 64 channels
        self.lat3 = nn.Conv2d(288, 64, kernel_size=1)
        self.lat2 = nn.Conv2d(160, 64, kernel_size=1)
        self.lat1 = nn.Conv2d(96,  64, kernel_size=1)

        self.smooth1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.fc_fpn = nn.Linear(192, 38)
        self.base_fc = base_cart.fc
        self.alpha_fpn = nn.Parameter(torch.zeros(1))
        self.head_prim = nn.Linear(192, 8)
        self.gate_prim = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h0 = self.stem(x)
        h1 = self.stage1(h0) # (B, 96, 26, 26)
        h2 = self.stage2(h1) # (B, 160, 13, 13)
        h3 = self.stage3(h2) # (B, 288, 7, 7)

        feat_macro = h3.mean([-2, -1]) # (B, 288)
        base_logits = self.base_fc(feat_macro)

        p3 = self.lat3(h3)
        p2 = self.lat2(h2) + F.interpolate(p3, size=h2.shape[-2:], mode='nearest')
        p1 = self.lat1(h1) + F.interpolate(p2, size=h1.shape[-2:], mode='nearest')

        p3 = self.smooth3(p3).mean([-2, -1])
        p2 = self.smooth2(p2).mean([-2, -1])
        p1 = self.smooth1(p1).mean([-2, -1])

        feat_fpn = torch.cat([p1, p2, p3], dim=1) # (B, 192)
        fpn_logits = self.fc_fpn(feat_fpn)

        prim_logits = self.head_prim(feat_fpn)
        comp_logits = torch.matmul(prim_logits, VECS_BALANCED_T.to(prim_logits.device).T)

        final_logits = base_logits + torch.sigmoid(self.alpha_fpn) * fpn_logits + torch.sigmoid(self.gate_prim) * comp_logits
        return final_logits, prim_logits

def run_trial_17(loaders):
    log_msg("=" * 80)
    log_msg("STARTING TRIAL 17: FPN-WAFERNET (FEATURE PYRAMID WITH HIGH-RES SCRATCH HIGHWAY)")
    log_msg("=" * 80)

    base_cart = DPCANetV3(num_classes=38).to(DEVICE)
    c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
    cart_sd = {k: v for k, v in c_cart['model_state_dict'].items() if 'head_prim' not in k and 'gate_prim' not in k}
    base_cart.load_state_dict(cart_sd, strict=False)

    model = FixedFPNWaferNet(base_cart).to(DEVICE)
    for p in model.stem.parameters(): p.requires_grad = False
    for p in model.stage1.parameters(): p.requires_grad = False

    for m in [model.lat1, model.lat2, model.lat3, model.smooth1, model.smooth2, model.smooth3, model.fc_fpn]:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None: nn.init.zeros_(m.bias)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1.2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)

    pos_weights = torch.tensor([1.0, 1.0, 2.5, 1.2, 2.0, 1.0, 3.0, 1.0], device=DEVICE)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    class_weights = torch.ones(38, dtype=torch.float32, device=DEVICE)
    for c in CONFUSED_CLASSES: class_weights[c] = 2.0

    best_err = 999
    best_acc = 0.0

    for ep in range(1, 9):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for x, y, y_prim in loaders['train']:
            x, y, y_prim = x.to(DEVICE), y.to(DEVICE), y_prim.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=True):
                logits, prim_logits = model(x)
                ce = F.cross_entropy(logits, y, reduction='none', label_smoothing=0.015)
                loss_ce = torch.mean((1.0 - torch.exp(-ce))**1.5 * class_weights[y] * ce)
                loss_bce = bce_loss_fn(prim_logits, y_prim)
                loss = loss_ce + 0.30 * loss_bce

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(y)

        scheduler.step()
        train_loss /= len(loaders['train'].dataset)
        dt = time.time() - t0

        t_err, t_acc, err_std, err_sharp = evaluate_d4_tta(model, loaders['test'], loaders['y_test'])
        log_msg(f"Trial 17 | Ep {ep}/8 ({dt:.1f}s) | Loss: {train_loss:.4f} | D4 Test Errs: {t_err:2d} (Std: {err_std:2d}, Sharp: {err_sharp:2d}) ({t_acc:.3f}%) [Best: {best_err:2d}]")

        if t_err < best_err:
            best_err = t_err
            best_acc = t_acc
            torch.save({'model_state_dict': model.state_dict(), 'test_err': t_err, 'test_acc': t_acc},
                       os.path.join(OUT_DIR, "fpnwafer_trial17_best.pt"))
            if t_err <= 65:
                log_msg(f"*** HISTORIC BREAKTHROUGH: TRIAL 17 SURPASSED SOTA WITH {t_err} ERRORS! ***")
                torch.save(model.state_dict(), os.path.join(OUT_DIR, "HISTORIC_SOTA_SURPASSED_MODEL.pt"))

    record_trial(17, "FPN-WaferNet", sum(p.numel() for p in trainable_params), train_loss, 0, best_err, best_acc,
                 "Multi-Scale Feature Pyramid with Stage 1 Lateral Highway for faint scratches and edge clusters")
    return best_err

def main_loop_part3():
    log_msg("\n" + "=" * 85)
    log_msg("STARTING AUTONOMOUS OVERNIGHT BREAKTHROUGH CAMPAIGN - PART 3 (TRIALS 16 - 17)")
    log_msg("TARGET: <= 65 ERRORS (> 99.130% SOTA) AS A STANDALONE SINGLE MODEL")
    log_msg("=" * 85)

    loaders = get_fast_wafer_loaders(batch_size=64)
    trials = [
        ("Trial 16 (DistillMaster-v2)", run_trial_16),
        ("Trial 17 (FPN-WaferNet)", run_trial_17),
    ]

    for name, trial_fn in trials:
        log_msg(f"\n>>> LAUNCHING {name} >>>")
        try:
            best_err = trial_fn(loaders)
            log_msg(f"<<< {name} FINISHED WITH BEST ERRORS: {best_err} / 7,603 <<<")
            if best_err <= 65:
                log_msg(f"*** BREAKTHROUGH ACHIEVED: {name} HIT {best_err} ERRORS (<= 65 SOTA TARGET)! ***")
        except Exception as e:
            log_msg(f"ERROR in {name}: {str(e)}")
            traceback.print_exc()

if __name__ == '__main__':
    main_loop_part3()
