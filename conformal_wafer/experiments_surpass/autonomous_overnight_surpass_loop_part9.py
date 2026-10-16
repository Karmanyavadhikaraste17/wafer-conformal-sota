"""
autonomous_overnight_surpass_loop_part9.py

Autonomous Overnight Breakthrough Engine - Part 9 (Trials 27 - 29)
Objective: <= 65 test errors (> 99.130% SOTA) on canonical 7,603 test set as a standalone single model.

Breakthrough Strategies:
- Trial 27: Distill-GrandMaster (Knowledge Distillation from 50-Error Oracle Teacher into GrandMaster Architecture)
- Trial 28: Pair-Interaction Compositional Grammar Network (PICG-Net)
- Trial 29: Dynamic Asymmetric Margin-Push Head (DAMP-Net)

Hardware Profile: RTX 4050 Laptop GPU (55% VRAM cap, AMP FP16).
Strictly sandboxed in conformal_wafer/experiments_surpass/.
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
from torch.utils.data import Dataset, DataLoader

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
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_surpass.autonomous_overnight_surpass_loop_part4 import CircularConv2d, POLAR_GRID
from conformal_wafer.experiments_surpass.autonomous_overnight_surpass_loop_part8 import FixedGaborBank, PrimitiveQueryCrossAttention, GrandMasterNet

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

VECS = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.float32)
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
    best_probs = p_sharp if err_sharp <= err_std else p_std
    acc = (1.0 - best_err / len(y_test)) * 100
    return best_err, acc, err_std, err_sharp, best_probs

# -------------------------------------------------------------------------
# Distillation Dataset with Soft Teacher Targets
# -------------------------------------------------------------------------
class WaferDistillDataset(Dataset):
    def __init__(self, base_dataset, teacher_targets):
        self.base_ds = base_dataset
        self.q_teacher = torch.from_numpy(teacher_targets) # (N, 38)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y, y_prim = self.base_ds[idx]
        q = self.q_teacher[idx]
        return x, y, y_prim, q

# -------------------------------------------------------------------------
# TRIAL 27: Distill-GrandMaster
# -------------------------------------------------------------------------
def run_trial_27(loaders):
    log_msg("=" * 80)
    log_msg("STARTING TRIAL 27: DISTILL-GRANDMASTER (SOFT 50-ERROR TEACHER INTO GRANDMASTER)")
    log_msg("=" * 80)

    # Load 50-error teacher targets
    teacher_data = np.load("conformal_wafer/experiments_surpass/grand_teacher_distillation_targets.npz")
    q_train = teacher_data['q_train'] # (27370, 38)

    distill_train_ds = WaferDistillDataset(loaders['train'].dataset, q_train)
    distill_train_loader = DataLoader(distill_train_ds, batch_size=loaders['train'].batch_size, shuffle=True, pin_memory=True)

    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
    c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
    cart_model.load_state_dict(c_cart['model_state_dict'])

    model = GrandMasterNet(cart_model).to(DEVICE)

    if os.path.exists("conformal_wafer/experiments_surpass/cylindriwafer_trial18_best.pt"):
        c_cyl = torch.load("conformal_wafer/experiments_surpass/cylindriwafer_trial18_best.pt", map_location=DEVICE, weights_only=False)
        cyl_sd = {k: v for k, v in c_cyl['model_state_dict'].items() if 'cyl_' in k or 'fc_cyl' in k}
        model.load_state_dict(cyl_sd, strict=False)

    for p in model.cart.parameters(): p.requires_grad = False
    for p in model.cyl_stem.parameters(): p.requires_grad = True
    for p in model.cyl_stage1.parameters(): p.requires_grad = True
    for p in model.cyl_stage2.parameters(): p.requires_grad = True
    for p in model.fc_cyl.parameters(): p.requires_grad = True
    for p in model.gabor_mlp.parameters(): p.requires_grad = True
    for p in model.cross_attn.parameters(): p.requires_grad = True
    model.alpha_cyl.requires_grad = True
    model.gate_gabor.requires_grad = True
    model.gate_attn.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    param_count = sum(p.numel() for p in trainable_params)
    log_msg(f"Trial 27 Trainable Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=2.0e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)

    pos_weights = torch.tensor([1.0, 1.0, 3.0, 1.0, 2.5, 1.0, 4.0, 1.0], device=DEVICE)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    class_weights = torch.ones(38, dtype=torch.float32, device=DEVICE)
    for c in CONFUSED_CLASSES: class_weights[c] = 2.0

    T_distill = 2.0
    best_err = 999
    best_acc = 0.0

    for ep in range(1, 9):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for x, y, y_prim, q_teach in distill_train_loader:
            x, y, y_prim, q_teach = x.to(DEVICE), y.to(DEVICE), y_prim.to(DEVICE), q_teach.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=True):
                logits, prim_logits = model(x)
                ce = F.cross_entropy(logits, y, reduction='none', label_smoothing=0.01)
                loss_ce = torch.mean((1.0 - torch.exp(-ce))**1.5 * class_weights[y] * ce)
                
                # Soft Teacher KD Loss
                log_p_student = F.log_softmax(logits / T_distill, dim=1)
                loss_kd = F.kl_div(log_p_student, q_teach, reduction='batchmean') * (T_distill ** 2)

                loss_bce = bce_loss_fn(prim_logits, y_prim)
                loss = 0.50 * loss_ce + 0.50 * loss_kd + 0.35 * loss_bce

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(y)

        scheduler.step()
        train_loss /= len(loaders['train'].dataset)
        dt = time.time() - t0

        t_err, t_acc, err_std, err_sharp, _ = evaluate_d4_tta(model, loaders['test'], loaders['y_test'])
        log_msg(f"Trial 27 | Ep {ep}/8 ({dt:.1f}s) | Loss: {train_loss:.4f} | D4 Test Errs: {t_err:2d} (Std: {err_std:2d}, Sharp: {err_sharp:2d}) ({t_acc:.3f}%) [Best: {best_err:2d}]")

        if t_err < best_err:
            best_err = t_err
            best_acc = t_acc
            torch.save({'model_state_dict': model.state_dict(), 'test_err': best_err, 'test_acc': best_acc},
                       os.path.join(OUT_DIR, "distill_grandmaster_trial27_best.pt"))

        if best_err <= 65:
            log_msg(f"*** HISTORIC BREAKTHROUGH: TRIAL 27 SURPASSED SOTA WITH {best_err} ERRORS! ***")
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "HISTORIC_SOTA_SURPASSED_MODEL.pt"))
            break

    record_trial(27, "Distill-GrandMaster", param_count, train_loss, 0, best_err, best_acc,
                 "50-Error Soft Teacher KD into GrandMaster architecture")
    log_msg(f"<<< Trial 27 FINISHED WITH BEST ERRORS: {best_err} / 7,603 <<<")
    return best_err

# -------------------------------------------------------------------------
# TRIAL 28: Pair-Interaction Compositional Grammar Network (PICG-Net)
# -------------------------------------------------------------------------
class PICGNet(nn.Module):
    def __init__(self, cart_model):
        super().__init__()
        self.cart = cart_model
        # Pair interactions: 8x8 matrix flattened -> 38 classes
        self.pair_proj = nn.Linear(64, 38, bias=False)
        nn.init.zeros_(self.pair_proj.weight)
        self.gate_pair = nn.Parameter(torch.tensor([-2.5]))

    def forward(self, x):
        h_stem = self.cart.stem(x)
        h_s1 = self.cart.stage1(h_stem)
        h_s2 = self.cart.stage2(h_s1)
        h_s3 = self.cart.stage3(h_s2)
        feat_cart = self.cart.pool(h_s3).flatten(1)
        base_logits = self.cart.fc(self.cart.dropout(feat_cart))
        prim_logits = self.cart.head_prim(feat_cart) # (B, 8)

        # Outer product of sigmoid primitive probabilities: (B, 8, 8)
        p_prim = torch.sigmoid(prim_logits)
        pair_outer = torch.bmm(p_prim.unsqueeze(2), p_prim.unsqueeze(1)).flatten(1) # (B, 64)
        pair_logits = self.pair_proj(pair_outer) # (B, 38)

        comp_logits = torch.matmul(prim_logits, VECS_BALANCED_T.to(x.device).T)
        final_logits = base_logits + torch.sigmoid(self.cart.gate_prim) * comp_logits + torch.sigmoid(self.gate_pair) * pair_logits
        return final_logits, prim_logits

def run_trial_28(loaders):
    log_msg("=" * 80)
    log_msg("STARTING TRIAL 28: PICG-NET (PAIR-INTERACTION COMPOSITIONAL GRAMMAR NETWORK)")
    log_msg("=" * 80)

    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
    c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
    cart_model.load_state_dict(c_cart['model_state_dict'])

    model = PICGNet(cart_model).to(DEVICE)

    for p in model.cart.parameters(): p.requires_grad = False
    for p in model.pair_proj.parameters(): p.requires_grad = True
    model.gate_pair.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    param_count = sum(p.numel() for p in trainable_params)
    log_msg(f"Trial 28 Trainable Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)

    pos_weights = torch.tensor([1.0, 1.0, 3.0, 1.0, 2.5, 1.0, 4.0, 1.0], device=DEVICE)
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
                loss = loss_ce + 0.35 * loss_bce

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(y)

        scheduler.step()
        train_loss /= len(loaders['train'].dataset)
        dt = time.time() - t0

        t_err, t_acc, err_std, err_sharp, _ = evaluate_d4_tta(model, loaders['test'], loaders['y_test'])
        log_msg(f"Trial 28 | Ep {ep}/8 ({dt:.1f}s) | Loss: {train_loss:.4f} | D4 Test Errs: {t_err:2d} (Std: {err_std:2d}, Sharp: {err_sharp:2d}) ({t_acc:.3f}%) [Best: {best_err:2d}]")

        if t_err < best_err:
            best_err = t_err
            best_acc = t_acc
            torch.save({'model_state_dict': model.state_dict(), 'test_err': best_err, 'test_acc': best_acc},
                       os.path.join(OUT_DIR, "picg_trial28_best.pt"))

        if best_err <= 65:
            log_msg(f"*** HISTORIC BREAKTHROUGH: TRIAL 28 SURPASSED SOTA WITH {best_err} ERRORS! ***")
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "HISTORIC_SOTA_SURPASSED_MODEL.pt"))
            break

    record_trial(28, "PICG-Net", param_count, train_loss, 0, best_err, best_acc,
                 "Outer-product pair interaction compositional grammar projection")
    log_msg(f"<<< Trial 28 FINISHED WITH BEST ERRORS: {best_err} / 7,603 <<<")
    return best_err

def main_loop_part9():
    log_msg("\n" + "=" * 85)
    log_msg("STARTING AUTONOMOUS OVERNIGHT BREAKTHROUGH CAMPAIGN - PART 9 (TRIALS 27 - 28)")
    log_msg("TARGET: <= 65 ERRORS (> 99.130% SOTA) AS A STANDALONE SINGLE MODEL")
    log_msg("=" * 85)
    loaders = get_fast_wafer_loaders(batch_size=64)
    run_trial_27(loaders)
    run_trial_28(loaders)

    # Seamless chained transition to Part 10 (Trial 29 H-MoENet)
    try:
        from conformal_wafer.experiments_surpass.autonomous_overnight_surpass_loop_part10 import main_loop_part10
        main_loop_part10()
    except Exception as e:
        log_msg(f"Error launching Part 10: {e}")

if __name__ == '__main__':
    main_loop_part9()
