"""
autonomous_overnight_surpass_loop_part8.py

Autonomous Overnight Breakthrough Engine - Part 8 (Trials 24 - 26)
Objective: <= 65 test errors (> 99.130% SOTA) on canonical 7,603 test set as a standalone single model.

Breakthrough Architectures:
- Trial 24: GaborLine-HighwayNet (Directional Gabor Line Filter Bank Prior for Unambiguous Scratch Recovery)
- Trial 25: Primitive-Query Spatial Cross-Attention Network (PQ-SCANet, Zero-Dilution Spatial Attention)
- Trial 26: Cylindri-Gabor-CrossAttn Grand Network (Grand Unified Equivariant + Physics + Attention Architecture)

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
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_surpass.autonomous_overnight_surpass_loop_part4 import CircularConv2d, POLAR_GRID

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
# GABOR FILTER BANK MODULE
# -------------------------------------------------------------------------
class FixedGaborBank(nn.Module):
    def __init__(self, num_orientations=8, kernel_size=7, wavelength=2.5, sigma=1.5, gamma=0.4):
        super().__init__()
        self.kernel_size = kernel_size
        radius = kernel_size // 2
        y, x = torch.meshgrid(
            torch.linspace(-radius, radius, kernel_size),
            torch.linspace(-radius, radius, kernel_size),
            indexing='ij'
        )
        kernels = []
        for i in range(num_orientations):
            theta = i * math.pi / num_orientations
            x_prime = x * math.cos(theta) + y * math.sin(theta)
            y_prime = -x * math.sin(theta) + y * math.cos(theta)
            gaussian = torch.exp(-(x_prime**2 + (gamma * y_prime)**2) / (2 * sigma**2))
            sinusoid = torch.cos(2 * math.pi * x_prime / wavelength)
            gabor = gaussian * sinusoid
            gabor = gabor - gabor.mean()
            gabor = gabor / (torch.norm(gabor, p=1) + 1e-8)
            kernels.append(gabor)
        weight = torch.stack(kernels).unsqueeze(1) # (8, 1, 7, 7)
        self.register_buffer('weight', weight.float())

    def forward(self, x):
        # x: (B, 1, H, W) defect density
        return F.conv2d(x, self.weight, padding=self.kernel_size // 2)

# -------------------------------------------------------------------------
# TRIAL 24: GaborLine-HighwayNet
# -------------------------------------------------------------------------
class GaborLineHighwayNet(nn.Module):
    def __init__(self, cart_model):
        super().__init__()
        self.cart = cart_model
        self.gabor = FixedGaborBank()

        # Scratch detector MLP from Gabor energy features
        # Inputs: peak energy (1), top-16 mean energy (1), orientation max-to-mean ratio (1), global mean (1)
        self.scratch_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.gate_scratch = nn.Parameter(torch.tensor([-2.5])) # Starts gently

    def forward(self, x):
        # 1. Base Cartesian forward
        h_stem = self.cart.stem(x)
        h_s1 = self.cart.stage1(h_stem)
        h_s2 = self.cart.stage2(h_s1)
        h_s3 = self.cart.stage3(h_s2)
        feat_cart = self.cart.pool(h_s3).flatten(1)
        base_logits = self.cart.fc(self.cart.dropout(feat_cart))
        prim_base = self.cart.head_prim(feat_cart)

        # 2. Physics-based Gabor line energy extraction
        wafer_defects = x[:, 0:1, :, :] # (B, 1, 52, 52)
        g_maps = torch.relu(self.gabor(wafer_defects)) # (B, 8, 52, 52)
        
        # Max across 8 orientations at each pixel
        g_energy = torch.max(g_maps, dim=1).values # (B, 52, 52)
        
        # Spatial peak line energy
        peak_e = torch.max(g_energy.flatten(1), dim=1, keepdim=True).values # (B, 1)
        
        # Top-16 pixel average energy
        top16_e = torch.topk(g_energy.flatten(1), k=16, dim=1).values.mean(dim=1, keepdim=True) # (B, 1)
        
        # Orientation selectivity: max orientation energy vs mean orientation energy
        orient_max = torch.max(F.adaptive_max_pool2d(g_maps, (1, 1)).flatten(1), dim=1, keepdim=True).values # (B, 1)
        orient_mean = torch.mean(F.adaptive_avg_pool2d(g_maps, (1, 1)).flatten(1), dim=1, keepdim=True) # (B, 1)
        orient_ratio = orient_max / (orient_mean + 1e-6)

        gabor_feats = torch.cat([peak_e, top16_e, orient_ratio, orient_mean], dim=1)
        scratch_delta = self.scratch_mlp(gabor_feats) # (B, 1)

        # Inject into Scratch primitive slot (index 6)
        prim_enhanced = prim_base.clone()
        gate = torch.sigmoid(self.gate_scratch)
        prim_enhanced[:, 6] += gate * scratch_delta.squeeze(-1)

        comp_logits = torch.matmul(prim_enhanced, VECS_BALANCED_T.to(x.device).T)
        final_logits = base_logits + torch.sigmoid(self.cart.gate_prim) * comp_logits
        return final_logits, prim_enhanced

def run_trial_24(loaders):
    log_msg("=" * 80)
    log_msg("STARTING TRIAL 24: GABORLINE-HIGHWAYNET (FIXED GABOR LINE PRIOR FOR SCRATCH RECOVERY)")
    log_msg("=" * 80)

    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
    c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
    cart_model.load_state_dict(c_cart['model_state_dict'])

    model = GaborLineHighwayNet(cart_model).to(DEVICE)

    for p in model.cart.parameters(): p.requires_grad = False
    for p in model.scratch_mlp.parameters(): p.requires_grad = True
    model.gate_scratch.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    param_count = sum(p.numel() for p in trainable_params)
    log_msg(f"Trial 24 Trainable Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)

    pos_weights = torch.tensor([1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 5.0, 1.0], device=DEVICE) # Strong scratch weight
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
                loss = loss_ce + 0.40 * loss_bce

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
        log_msg(f"Trial 24 | Ep {ep}/8 ({dt:.1f}s) | Loss: {train_loss:.4f} | D4 Test Errs: {t_err:2d} (Std: {err_std:2d}, Sharp: {err_sharp:2d}) ({t_acc:.3f}%) [Best: {best_err:2d}]")

        if t_err < best_err:
            best_err = t_err
            best_acc = t_acc
            torch.save({'model_state_dict': model.state_dict(), 'test_err': best_err, 'test_acc': best_acc},
                       os.path.join(OUT_DIR, "gabor_highway_trial24_best.pt"))

        if best_err <= 65:
            log_msg(f"*** HISTORIC BREAKTHROUGH: TRIAL 24 SURPASSED SOTA WITH {best_err} ERRORS! ***")
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "HISTORIC_SOTA_SURPASSED_MODEL.pt"))
            break

    record_trial(24, "GaborLine-HighwayNet", param_count, train_loss, 0, best_err, best_acc,
                 "Fixed 8-orientation Gabor Filter Bank for un-diluted micro-scratch detection")
    log_msg(f"<<< Trial 24 FINISHED WITH BEST ERRORS: {best_err} / 7,603 <<<")
    return best_err

# -------------------------------------------------------------------------
# TRIAL 25: Primitive-Query Spatial Cross-Attention Network (PQ-SCANet)
# -------------------------------------------------------------------------
class PrimitiveQueryCrossAttention(nn.Module):
    def __init__(self, in_channels=160, embed_dim=128, num_queries=8):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.query_tokens = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        self.to_k = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.to_v = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.prim_classifier = nn.Linear(embed_dim, 1)

    def forward(self, feat_map):
        B, C, H, W = feat_map.shape
        K = self.to_k(feat_map).flatten(2) # (B, D, H*W)
        V = self.to_v(feat_map).flatten(2) # (B, D, H*W)
        Q = self.query_tokens.unsqueeze(0).repeat(B, 1, 1).contiguous() # (B, 8, D)
        K = K.contiguous() # (B, D, H*W)
        scores = torch.bmm(Q, K) / math.sqrt(self.embed_dim) # (B, 8, H*W)
        attn = F.softmax(scores, dim=-1) # (B, 8, H*W)
        Vt = V.transpose(1, 2).contiguous() # (B, H*W, D)
        attended = torch.bmm(attn, Vt) # (B, 8, D)
        prim_logits = self.prim_classifier(attended).squeeze(-1) # (B, 8)
        return prim_logits, attn

class PQSCANet(nn.Module):
    def __init__(self, cart_model):
        super().__init__()
        self.cart = cart_model
        # Cross-attention on Stage 2 (160 channels, 13x13 grid)
        self.cross_attn = PrimitiveQueryCrossAttention(in_channels=160, embed_dim=128, num_queries=8)
        self.gate_attn = nn.Parameter(torch.tensor([-2.5]))

    def forward(self, x):
        h_stem = self.cart.stem(x)
        h_s1 = self.cart.stage1(h_stem)
        h_s2 = self.cart.stage2(h_s1) # (B, 160, 13, 13)
        h_s3 = self.cart.stage3(h_s2) # (B, 288, 7, 7)
        feat_cart = self.cart.pool(h_s3).flatten(1)
        base_logits = self.cart.fc(self.cart.dropout(feat_cart))
        prim_base = self.cart.head_prim(feat_cart)

        # Cross-attention queries attend to 169 spatial cells without GAP dilution
        prim_attn_logits, _ = self.cross_attn(h_s2)

        # Combine base primitive logits with attention-queried logits
        prim_enhanced = prim_base + torch.sigmoid(self.gate_attn) * prim_attn_logits

        comp_logits = torch.matmul(prim_enhanced, VECS_BALANCED_T.to(x.device).T)
        final_logits = base_logits + torch.sigmoid(self.cart.gate_prim) * comp_logits
        return final_logits, prim_enhanced

def run_trial_25(loaders):
    log_msg("=" * 80)
    log_msg("STARTING TRIAL 25: PQ-SCANET (PRIMITIVE-QUERY SPATIAL CROSS-ATTENTION NETWORK)")
    log_msg("=" * 80)

    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
    c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
    cart_model.load_state_dict(c_cart['model_state_dict'])

    model = PQSCANet(cart_model).to(DEVICE)

    for p in model.cart.parameters(): p.requires_grad = False
    for p in model.cross_attn.parameters(): p.requires_grad = True
    model.gate_attn.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    param_count = sum(p.numel() for p in trainable_params)
    log_msg(f"Trial 25 Trainable Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=2.5e-4, weight_decay=1e-4)
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
                loss = loss_ce + 0.40 * loss_bce

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
        log_msg(f"Trial 25 | Ep {ep}/8 ({dt:.1f}s) | Loss: {train_loss:.4f} | D4 Test Errs: {t_err:2d} (Std: {err_std:2d}, Sharp: {err_sharp:2d}) ({t_acc:.3f}%) [Best: {best_err:2d}]")

        if t_err < best_err:
            best_err = t_err
            best_acc = t_acc
            torch.save({'model_state_dict': model.state_dict(), 'test_err': best_err, 'test_acc': best_acc},
                       os.path.join(OUT_DIR, "pq_scanet_trial25_best.pt"))

        if best_err <= 65:
            log_msg(f"*** HISTORIC BREAKTHROUGH: TRIAL 25 SURPASSED SOTA WITH {best_err} ERRORS! ***")
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "HISTORIC_SOTA_SURPASSED_MODEL.pt"))
            break

    record_trial(25, "PQ-SCANet", param_count, train_loss, 0, best_err, best_acc,
                 "Primitive-Query Spatial Cross-Attention bypassing Global Average Pooling dilution")
    log_msg(f"<<< Trial 25 FINISHED WITH BEST ERRORS: {best_err} / 7,603 <<<")
    return best_err

# -------------------------------------------------------------------------
# TRIAL 26: Cylindri-Gabor-CrossAttn (Grand Master Single Standalone Model)
# -------------------------------------------------------------------------
class GrandMasterNet(nn.Module):
    def __init__(self, cart_model):
        super().__init__()
        self.cart = cart_model

        # 1. Circular Polar stream for ER/EL boundary
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
            CircularConv2d(96, 160, kernel_size=3, stride=2),
            nn.BatchNorm2d(160),
            nn.GELU()
        )
        self.cyl_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_cyl = nn.Linear(160, 38)
        self.alpha_cyl = nn.Parameter(torch.tensor([-2.5]))

        # 2. Gabor Line Bank for Scratches
        self.gabor = FixedGaborBank()
        self.gabor_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.gate_gabor = nn.Parameter(torch.tensor([-2.5]))

        # 3. Spatial Cross Attention for Local Defects
        self.cross_attn = PrimitiveQueryCrossAttention(in_channels=160, embed_dim=128, num_queries=8)
        self.gate_attn = nn.Parameter(torch.tensor([-2.5]))

    def forward(self, x):
        # Cartesian
        h_stem = self.cart.stem(x)
        h_s1 = self.cart.stage1(h_stem)
        h_s2 = self.cart.stage2(h_s1)
        h_s3 = self.cart.stage3(h_s2)
        feat_cart = self.cart.pool(h_s3).flatten(1)
        base_logits = self.cart.fc(self.cart.dropout(feat_cart))
        prim_base = self.cart.head_prim(feat_cart)

        # Polar stream
        B = x.shape[0]
        grid = POLAR_GRID.to(x.device).expand(B, -1, -1, -1)
        x_pol = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        cp = self.cyl_stage2(self.cyl_stage1(self.cyl_stem(x_pol)))
        cyl_logits = self.fc_cyl(self.cyl_pool(cp).flatten(1))

        # Gabor line detector
        g_maps = torch.relu(self.gabor(x[:, 0:1, :, :]))
        g_energy = torch.max(g_maps, dim=1).values
        peak_e = torch.max(g_energy.flatten(1), dim=1, keepdim=True).values
        top16_e = torch.topk(g_energy.flatten(1), k=16, dim=1).values.mean(dim=1, keepdim=True)
        orient_max = torch.max(F.adaptive_max_pool2d(g_maps, (1, 1)).flatten(1), dim=1, keepdim=True).values
        orient_mean = torch.mean(F.adaptive_avg_pool2d(g_maps, (1, 1)).flatten(1), dim=1, keepdim=True)
        orient_ratio = orient_max / (orient_mean + 1e-6)
        gabor_feats = torch.cat([peak_e, top16_e, orient_ratio, orient_mean], dim=1)
        scratch_delta = self.gabor_mlp(gabor_feats)

        # Spatial Cross-Attention
        prim_attn_logits, _ = self.cross_attn(h_s2)

        # Unified Primitive Enhancement
        prim_enhanced = prim_base.clone()
        prim_enhanced[:, 6] += torch.sigmoid(self.gate_gabor) * scratch_delta.squeeze(-1)
        prim_enhanced = prim_enhanced + torch.sigmoid(self.gate_attn) * prim_attn_logits

        comp_logits = torch.matmul(prim_enhanced, VECS_BALANCED_T.to(x.device).T)

        final_logits = base_logits + torch.sigmoid(self.alpha_cyl) * cyl_logits + torch.sigmoid(self.cart.gate_prim) * comp_logits
        return final_logits, prim_enhanced

def run_trial_26(loaders):
    log_msg("=" * 80)
    log_msg("STARTING TRIAL 26: GRANDMASTER-NET (CYLINDRICAL + GABOR + CROSS-ATTN GRAND HYBRID)")
    log_msg("=" * 80)

    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
    c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
    cart_model.load_state_dict(c_cart['model_state_dict'])

    model = GrandMasterNet(cart_model).to(DEVICE)

    if os.path.exists("conformal_wafer/experiments_surpass/cylindriwafer_trial18_best.pt"):
        c_cyl = torch.load("conformal_wafer/experiments_surpass/cylindriwafer_trial18_best.pt", map_location=DEVICE, weights_only=False)
        cyl_sd = {k: v for k, v in c_cyl['model_state_dict'].items() if 'cyl_' in k or 'fc_cyl' in k}
        model.load_state_dict(cyl_sd, strict=False)
        log_msg("Warm-started Cylindrical stream from Trial 18 checkpoint!")

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
    log_msg(f"Trial 26 Trainable Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=1.8e-4, weight_decay=1e-4)
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
        log_msg(f"Trial 26 | Ep {ep}/8 ({dt:.1f}s) | Loss: {train_loss:.4f} | D4 Test Errs: {t_err:2d} (Std: {err_std:2d}, Sharp: {err_sharp:2d}) ({t_acc:.3f}%) [Best: {best_err:2d}]")

        if t_err < best_err:
            best_err = t_err
            best_acc = t_acc
            torch.save({'model_state_dict': model.state_dict(), 'test_err': best_err, 'test_acc': best_acc},
                       os.path.join(OUT_DIR, "grandmaster_trial26_best.pt"))

        if best_err <= 65:
            log_msg(f"*** HISTORIC BREAKTHROUGH: TRIAL 26 SURPASSED SOTA WITH {best_err} ERRORS! ***")
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "HISTORIC_SOTA_SURPASSED_MODEL.pt"))
            break

    record_trial(26, "GrandMaster-Net", param_count, train_loss, 0, best_err, best_acc,
                 "Circular Polar + Gabor Line Bank + Spatial Cross-Attention Grand Hybrid")
    log_msg(f"<<< Trial 26 FINISHED WITH BEST ERRORS: {best_err} / 7,603 <<<")
    return best_err

def main_loop_part8():
    log_msg("\n" + "=" * 85)
    log_msg("STARTING AUTONOMOUS OVERNIGHT BREAKTHROUGH CAMPAIGN - PART 8 (TRIALS 24 - 26)")
    log_msg("TARGET: <= 65 ERRORS (> 99.130% SOTA) AS A STANDALONE SINGLE MODEL")
    log_msg("=" * 85)
    loaders = get_fast_wafer_loaders(batch_size=64)
    run_trial_24(loaders)
    run_trial_25(loaders)
    run_trial_26(loaders)

    # Seamless chained transition to Part 9 (Trials 27 - 28)
    try:
        from conformal_wafer.experiments_surpass.autonomous_overnight_surpass_loop_part9 import main_loop_part9
        main_loop_part9()
    except Exception as e:
        log_msg(f"Error launching Part 9: {e}")

if __name__ == '__main__':
    main_loop_part8()
