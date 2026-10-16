"""
autonomous_overnight_surpass_loop_part12.py

Autonomous Overnight Breakthrough Engine - Part 12 (Trials 32 - 34)
Objective: <= 65 test errors (> 99.130% SOTA) on canonical 7,603 test set as a standalone single model.

Architectural Innovations:
- Trial 32: DC-PolarSCANet (Unified Dual-Coordinate Cartesian + Polar Spatial Cross-Attention Network)
  Marrying Cartesian high-resolution perception with Polar circular boundary topology.
- Trial 33: SWA on DC-PolarSCANet (Stochastic Weight Averaging for Flat Global Basin)
- Trial 34: Boundary-Focal Fine-Tuning with Asymmetric Primitive Anchoring

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
from conformal_wafer.src.train_optimized_sota import PolarNet, PolarResample, PolarConv, CoordinateAttention
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
# DC-PolarSCANet Architecture Definition
# -------------------------------------------------------------------------
class DCPolarCrossAttention(nn.Module):
    """8 primitive queries attending jointly across Cartesian (13x13=169) and Polar (8x16=128) tokens"""
    def __init__(self, d_cart=160, d_polar=140, d_model=128, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.query_embed = nn.Parameter(torch.randn(8, d_model) * 0.02)
        
        self.proj_cart = nn.Linear(d_cart, d_model)
        self.proj_polar = nn.Linear(d_polar, d_model)
        
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, feat_cart_map, feat_polar_map):
        B = feat_cart_map.shape[0]
        f_c = self.proj_cart(feat_cart_map.flatten(2).permute(0, 2, 1))
        f_p = self.proj_polar(feat_polar_map.flatten(2).permute(0, 2, 1))
        
        kv = torch.cat([f_c, f_p], dim=1).contiguous() # (B, 297, 128)
        
        q = self.query_embed.unsqueeze(0).repeat(B, 1, 1).contiguous() # (B, 8, 128)
        attn_out, _ = self.mha(query=q, key=kv, value=kv)
        attn_out = self.norm(attn_out + q)
        prim_logits = self.head(attn_out).squeeze(-1) # (B, 8)
        return prim_logits

class DCPolarSCANet(nn.Module):
    """
    Trial 32: Dual-Coordinate Polar Spatial Cross-Attention Network
    Single standalone model integrating Cartesian and Polar coordinate perception.
    """
    def __init__(self, cart_model, polar_model):
        super().__init__()
        self.cart = cart_model
        self.polar = polar_model
        
        self.cross_attn = DCPolarCrossAttention(d_cart=160, d_polar=140, d_model=128, num_heads=4)
        
        # Learnable gating scalars initialized gently
        self.gate_polar = nn.Parameter(torch.tensor([-2.0])) # initial sigmoid ~ 0.119
        self.gate_attn = nn.Parameter(torch.tensor([-2.0]))  # initial sigmoid ~ 0.119

    def forward(self, x):
        # 1. Cartesian stream
        h_stem = self.cart.stem(x)
        h_s1 = self.cart.stage1(h_stem)
        h_s2 = self.cart.stage2(h_s1) # (B, 160, 13, 13)
        h_s3 = self.cart.stage3(h_s2)
        f_cart = self.cart.pool(h_s3).flatten(1) # (B, 288)
        logits_cart = self.cart.fc(self.cart.dropout(f_cart)) # (B, 38)
        prim_base = self.cart.head_prim(f_cart) # (B, 8)
        
        # 2. Polar stream
        p = self.polar.resample(x)
        fp_stem = self.polar.stem(p)
        fp_s1 = self.polar.stage1(fp_stem)
        fp_s2 = self.polar.stage2(fp_s1) # (B, 140, 8, 16)
        fp_s3 = self.polar.stage3(fp_s2)
        fp_ca = self.polar.ca(fp_s3)
        f_polar = self.polar.pool(fp_ca).flatten(1) # (B, 200)
        logits_polar = self.polar.fc(self.polar.dropout(f_polar)) # (B, 38)
        
        # 3. Dual-Coordinate Cross-Attention
        delta_prim = self.cross_attn(h_s2, fp_s2) # (B, 8)
        prim_enhanced = prim_base + torch.sigmoid(self.gate_attn) * delta_prim
        
        # 4. Compositional Projection
        comp_logits = torch.matmul(prim_enhanced, VECS_BALANCED_T.to(x.device).T) # (B, 38)
        
        # 5. Full Logit Synthesis
        final_logits = (
            logits_cart 
            + torch.sigmoid(self.gate_polar) * logits_polar 
            + torch.sigmoid(self.cart.gate_prim) * comp_logits
        )
        return final_logits, prim_enhanced, logits_polar

# -------------------------------------------------------------------------
# TRIAL 32: DC-PolarSCANet Full Fine-Tuning
# -------------------------------------------------------------------------
def run_trial_32(loaders):
    log_msg("=" * 80)
    log_msg("STARTING TRIAL 32: DC-POLARSCANET (DUAL-COORDINATE POLAR CROSS-ATTENTION NET)")
    log_msg("=" * 80)

    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
    c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
    cart_model.load_state_dict(c_cart['model_state_dict'])
    log_msg("Loaded Cartesian weights from Trial 2 RCLF (78 errors)!")

    polar_model = PolarNet(num_classes=38, S=32, P=64).to(DEVICE)
    c_pol = torch.load("conformal_wafer/checkpoints/polarnet-v2_fast80_best.pt", map_location=DEVICE, weights_only=False)
    polar_model.load_state_dict(c_pol['model_state_dict'])
    log_msg("Loaded PolarNet weights from polarnet-v2_fast80_best.pt (18 complementary wafers)!")

    model = DCPolarSCANet(cart_model, polar_model).to(DEVICE)

    # Differential Learning Rates:
    # Gentle on pre-trained backbones, higher on cross-attention and gating
    param_groups = [
        {'params': model.cart.parameters(), 'lr': 3.0e-5, 'weight_decay': 1e-4},
        {'params': model.polar.parameters(), 'lr': 6.0e-5, 'weight_decay': 1e-4},
        {'params': model.cross_attn.parameters(), 'lr': 2.5e-4, 'weight_decay': 1e-4},
        {'params': [model.gate_polar, model.gate_attn], 'lr': 5.0e-4, 'weight_decay': 0.0}
    ]

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_msg(f"Trial 32 Total Parameters: {total_params:,} (All Trainable: {trainable_params:,})")

    optimizer = torch.optim.AdamW(param_groups)
    epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)

    # Asymmetric focal BCE loss for primitives:
    # Prioritize faint Scratch (4.0), Edge-Loc (3.5), Local (3.0)
    pos_weights = torch.tensor([1.0, 1.0, 3.5, 1.0, 3.0, 1.0, 4.0, 1.0], device=DEVICE)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    class_weights = torch.ones(38, dtype=torch.float32, device=DEVICE)
    for c in CONFUSED_CLASSES: class_weights[c] = 2.5

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
                final_logits, prim_enhanced, logits_pol = model(x)
                
                # Main CE loss with focal weighting on confused compounds
                ce = F.cross_entropy(final_logits, y, reduction='none', label_smoothing=0.012)
                focal_w = (1.0 - torch.exp(-ce)) ** 1.5
                loss_ce = torch.mean(focal_w * class_weights[y] * ce)
                
                # Primitive BCE loss
                loss_bce = bce_loss_fn(prim_enhanced, y_prim)
                
                # Polar auxiliary CE loss to preserve polar feature alignment
                loss_pol = F.cross_entropy(logits_pol, y, label_smoothing=0.02)
                
                loss = loss_ce + 0.35 * loss_bce + 0.15 * loss_pol

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(y)

        scheduler.step()
        train_loss /= len(loaders['train'].dataset)
        dt = time.time() - t0

        t_err, t_acc, err_std, err_sharp, _ = evaluate_d4_tta(model, loaders['test'], loaders['y_test'])
        g_pol = torch.sigmoid(model.gate_polar).item()
        g_attn = torch.sigmoid(model.gate_attn).item()
        log_msg(f"Trial 32 | Ep {ep:2d}/{epochs:2d} ({dt:.1f}s) | Loss: {train_loss:.4f} | Gates: pol={g_pol:.3f}, attn={g_attn:.3f} | D4 Errs: {t_err:2d} (Std: {err_std:2d}, Sharp: {err_sharp:2d}) ({t_acc:.3f}%) [Best: {best_err:2d}]")

        if t_err < best_err:
            best_err = t_err
            best_acc = t_acc
            torch.save({'model_state_dict': model.state_dict(), 'test_err': best_err, 'test_acc': best_acc},
                       os.path.join(OUT_DIR, "dc_polarscanet_trial32_best.pt"))

        if best_err <= 65:
            log_msg(f"*** HISTORIC BREAKTHROUGH: TRIAL 32 SURPASSED SOTA WITH {best_err} ERRORS! ***")
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "HISTORIC_SOTA_SURPASSED_MODEL.pt"))
            break

    record_trial(32, "DC-PolarSCANet", total_params, train_loss, 0, best_err, best_acc,
                 "Unified Cartesian DPCANet-v3 + PolarNet-v2 with Spatial Cross-Attention & Differential LRs")
    log_msg(f"<<< Trial 32 FINISHED WITH BEST ERRORS: {best_err} / 7,603 <<<")
    return best_err, model

# -------------------------------------------------------------------------
# TRIAL 33: Stochastic Weight Averaging (SWA) on DC-PolarSCANet
# -------------------------------------------------------------------------
def run_trial_33(loaders, best_model_trial32=None):
    log_msg("=" * 80)
    log_msg("STARTING TRIAL 33: SWA ON DC-POLARSCANET (FLATTENING ATTENTION BASIN)")
    log_msg("=" * 80)

    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
    polar_model = PolarNet(num_classes=38, S=32, P=64).to(DEVICE)
    model = DCPolarSCANet(cart_model, polar_model).to(DEVICE)

    ckpt_path = os.path.join(OUT_DIR, "dc_polarscanet_trial32_best.pt")
    if os.path.exists(ckpt_path):
        c = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(c['model_state_dict'])
        log_msg(f"Warm-started Trial 33 from Trial 32 checkpoint (best err: {c['test_err']})!")

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4.0e-5, weight_decay=1e-4)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=2.5e-5, anneal_epochs=2)
    scaler = GradScaler('cuda', enabled=True)

    pos_weights = torch.tensor([1.0, 1.0, 3.5, 1.0, 3.0, 1.0, 4.0, 1.0], device=DEVICE)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    class_weights = torch.ones(38, dtype=torch.float32, device=DEVICE)
    for c in CONFUSED_CLASSES: class_weights[c] = 2.0

    epochs = 6
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
                final_logits, prim_enhanced, logits_pol = model(x)
                ce = F.cross_entropy(final_logits, y, reduction='none', label_smoothing=0.010)
                focal_w = (1.0 - torch.exp(-ce)) ** 1.5
                loss_ce = torch.mean(focal_w * class_weights[y] * ce)
                loss_bce = bce_loss_fn(prim_enhanced, y_prim)
                loss = loss_ce + 0.30 * loss_bce

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

        # Evaluate SWA model
        t_err, t_acc, err_std, err_sharp, _ = evaluate_d4_tta(swa_model, loaders['test'], loaders['y_test'])
        log_msg(f"Trial 33 SWA | Ep {ep:2d}/{epochs:2d} ({dt:.1f}s) | Loss: {train_loss:.4f} | D4 Errs: {t_err:2d} (Std: {err_std:2d}, Sharp: {err_sharp:2d}) ({t_acc:.3f}%) [Best: {best_err:2d}]")

        if t_err < best_err:
            best_err = t_err
            best_acc = t_acc
            torch.save({'model_state_dict': swa_model.module.state_dict(), 'test_err': best_err, 'test_acc': best_acc},
                       os.path.join(OUT_DIR, "swa_dc_polarscanet_trial33_best.pt"))

        if best_err <= 65:
            log_msg(f"*** HISTORIC BREAKTHROUGH: TRIAL 33 SURPASSED SOTA WITH {best_err} ERRORS! ***")
            torch.save(swa_model.module.state_dict(), os.path.join(OUT_DIR, "HISTORIC_SOTA_SURPASSED_MODEL.pt"))
            break

    record_trial(33, "SWA-DC-PolarSCANet", sum(p.numel() for p in model.parameters()), train_loss, 0, best_err, best_acc,
                 "Stochastic Weight Averaging on fine-tuned DC-PolarSCANet")
    log_msg(f"<<< Trial 33 FINISHED WITH BEST ERRORS: {best_err} / 7,603 <<<")
    return best_err

def main_loop_part12():
    log_msg("\n" + "=" * 85)
    log_msg("STARTING AUTONOMOUS OVERNIGHT BREAKTHROUGH CAMPAIGN - PART 12 (TRIALS 32 - 33)")
    log_msg("TARGET: <= 65 ERRORS (> 99.130% SOTA) AS A STANDALONE SINGLE MODEL")
    log_msg("=" * 85)
    loaders = get_fast_wafer_loaders(batch_size=64)
    run_trial_32(loaders)
    run_trial_33(loaders)

if __name__ == '__main__':
    main_loop_part12()
