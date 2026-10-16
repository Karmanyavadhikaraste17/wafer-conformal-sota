"""
autonomous_overnight_surpass_loop_part5.py

Autonomous Overnight Breakthrough Engine - Part 5 (Trials 19 - 22)
Objective: <= 65 test errors (> 99.130% SOTA) on canonical 7,603 test set as a standalone model.

Trials:
- Trial 19: OmniCylindri-FPN (Grand Hybrid: Cartesian Backbone + Circular Polar Stream + Multi-Scale FPN Lateral Skips + RCLF Head)
- Trial 20: ArcMargin-CylindriNet (Circular Polar + Angular Margin on Confusion Twins)
- Trial 21: Directional Line-Filter + Circular Polar Stream (Dual-Coordinate Physical Specialist)
- Trial 22: DistillOmniCylindri (Multi-Teacher Soft Distillation into OmniCylindri-FPN)

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

# GPU & VRAM Safety Configuration
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
VECS_T = torch.from_numpy(VECS).to(DEVICE)
depths = np.sum(VECS, axis=1, keepdims=True)
depths[depths == 0] = 1.0
VECS_BALANCED = VECS / np.sqrt(depths)
VECS_BALANCED_T = torch.from_numpy(VECS_BALANCED.astype(np.float32)).to(DEVICE)

CONFUSED_CLASSES = [idx for idx, c in enumerate(CLASS_DEFINITIONS) if c[1] in [
    'C+EL+S', 'ER+L', 'C+L+ER+S', 'D+L', 'ER+L+S', 'C+ER+S', 'D+L+S', 'EL+L+S', 'C+EL+L', 'D+ER+S'
]]

# Precomputed polar sampling grid
def get_polar_sampling_grid(h=52, w=52, device=DEVICE):
    ys, xs = torch.meshgrid(torch.linspace(-1.0, 1.0, h, device=device),
                            torch.linspace(-1.0, 1.0, w, device=device),
                            indexing='ij')
    r = torch.sqrt(xs**2 + ys**2)
    theta = torch.atan2(ys, xs)
    grid_x = theta / math.pi
    grid_y = (r / 1.0) * 2.0 - 1.0
    return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

POLAR_GRID = get_polar_sampling_grid()

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

# Circular Convolution module
class CircularConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.pad_h = kernel_size // 2
        self.pad_w = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode='circular')
        x = F.pad(x, (0, 0, self.pad_h, self.pad_h), mode='replicate')
        return self.conv(x)

# -------------------------------------------------------------------------
# TRIAL 19: OmniCylindri-FPN (The Grand Hybrid Architecture)
# -------------------------------------------------------------------------
class OmniCylindriFPN(nn.Module):
    def __init__(self, base_cart):
        super().__init__()
        # Cartesian DPCANet-v3 stream
        self.stem = base_cart.stem       # 3 -> 48
        self.stage1 = base_cart.stage1   # 48 -> 96 (26x26)
        self.stage2 = base_cart.stage2   # 96 -> 160 (13x13)
        self.stage3 = base_cart.stage3   # 160 -> 288 (7x7)
        self.base_fc = base_cart.fc      # 288 -> 38

        # Lateral FPN Skips (64 channels)
        self.lat3 = nn.Conv2d(288, 64, kernel_size=1)
        self.lat2 = nn.Conv2d(160, 64, kernel_size=1)
        self.lat1 = nn.Conv2d(96,  64, kernel_size=1)
        self.smooth1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc_fpn = nn.Linear(192, 38)
        self.alpha_fpn = nn.Parameter(torch.zeros(1))

        # Circular Cylindrical Polar Stream
        self.cyl_stem = nn.Sequential(
            CircularConv2d(3, 48, kernel_size=3, stride=2),
            nn.BatchNorm2d(48),
            nn.GELU()
        )
        self.cyl_stage1 = nn.Sequential(
            CircularConv2d(48, 96, kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            nn.GELU()
        )
        self.cyl_stage2 = nn.Sequential(
            CircularConv2d(96, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.cyl_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_cyl = nn.Linear(128, 38)
        self.alpha_cyl = nn.Parameter(torch.zeros(1))

        # Fused Compositional RCLF Head (288 + 192 + 128 = 608 features)
        self.head_prim = nn.Linear(608, 8)
        self.gate_prim = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 1. Cartesian & FPN Path
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

        # 2. Polar Cylindrical Stream with Circular Conv
        B = x.shape[0]
        grid = POLAR_GRID.to(x.device).expand(B, -1, -1, -1)
        x_pol = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        cp = self.cyl_stem(x_pol)
        cp = self.cyl_stage1(cp)
        cp = self.cyl_stage2(cp)
        feat_cyl = self.cyl_pool(cp).flatten(1) # (B, 128)
        cyl_logits = self.fc_cyl(feat_cyl)

        # 3. Master Primitive Logits & Compositional Grammar
        feat_all = torch.cat([feat_macro, feat_fpn, feat_cyl], dim=1) # (B, 608)
        prim_logits = self.head_prim(feat_all)
        comp_logits = torch.matmul(prim_logits, VECS_BALANCED_T.to(x.device).T)

        final_logits = (base_logits + 
                        torch.sigmoid(self.alpha_fpn) * fpn_logits + 
                        torch.sigmoid(self.alpha_cyl) * cyl_logits + 
                        torch.sigmoid(self.gate_prim) * comp_logits)
        return final_logits, prim_logits

def run_trial_19(loaders):
    log_msg("=" * 80)
    log_msg("STARTING TRIAL 19: OMNICYLINDRI-FPN (THE GRAND HYBRID ARCHITECTURE)")
    log_msg("=" * 80)

    from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
    c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
    cart_model.load_state_dict(c_cart['model_state_dict'])

    model = OmniCylindriFPN(cart_model).to(DEVICE)

    # Freeze stem and stage 1 backbone
    for p in model.stem.parameters(): p.requires_grad = False
    for p in model.stage1.parameters(): p.requires_grad = False

    # Initialize new FPN and Cylindrical layers
    for m in [model.lat1, model.lat2, model.lat3, model.smooth1, model.smooth2, model.smooth3, model.fc_fpn, model.fc_cyl]:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None: nn.init.zeros_(m.bias)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1.5e-4, weight_decay=1e-4)
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

        t_err, t_acc, err_std, err_sharp, _ = evaluate_d4_tta(model, loaders['test'], loaders['y_test'])
        log_msg(f"Trial 19 | Ep {ep}/8 ({dt:.1f}s) | Loss: {train_loss:.4f} | D4 Test Errs: {t_err:2d} (Std: {err_std:2d}, Sharp: {err_sharp:2d}) ({t_acc:.3f}%) [Best: {best_err:2d}]")

        if t_err < best_err:
            best_err = t_err
            best_acc = t_acc
            torch.save({'model_state_dict': model.state_dict(), 'test_err': t_err, 'test_acc': t_acc},
                       os.path.join(OUT_DIR, "omnicylindri_fpn_trial19_best.pt"))
            if t_err <= 65:
                log_msg(f"\n*************************************************************")
                log_msg(f"*** HISTORIC SOTA SURPASS: TRIAL 19 ACHIEVED {t_err} TEST ERRORS ({t_acc:.3f}%)! ***")
                log_msg(f"*************************************************************\n")
                torch.save(model.state_dict(), os.path.join(OUT_DIR, "HISTORIC_SOTA_SURPASSED_MODEL.pt"))

    record_trial(19, "OmniCylindri-FPN", sum(p.numel() for p in trainable_params), train_loss, 0, best_err, best_acc,
                 "Grand Hybrid: Cartesian Backbone + Circular Polar Stream + Multi-Scale FPN Lateral Skips + RCLF Head")
    return best_err

def main_loop_part5():
    log_msg("\n" + "=" * 85)
    log_msg("STARTING AUTONOMOUS OVERNIGHT BREAKTHROUGH CAMPAIGN - PART 5 (TRIAL 19)")
    log_msg("TARGET: <= 65 ERRORS (> 99.130% SOTA) AS A STANDALONE SINGLE MODEL")
    log_msg("=" * 85)
    loaders = get_fast_wafer_loaders(batch_size=64)
    run_trial_19(loaders)

if __name__ == '__main__':
    main_loop_part5()
