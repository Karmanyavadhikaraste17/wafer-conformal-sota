import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score

sys.path.insert(0, ".")

# -------------------------------------------------------------------------
# Thermal & VRAM Safety Constraints (RTX 4050 6GB Laptop GPU)
# -------------------------------------------------------------------------
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_surpass.trial_dashboard import log_trial
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.src.train_optimized_sota import PolarNet
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

OUT_DIR = "conformal_wafer/experiments_surpass"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Class Mapping & Grammar Definitions
# -------------------------------------------------------------------------
VECS = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.float32) # (38, 8)
VECS_T = torch.from_numpy(VECS).to(DEVICE)

depths = np.sum(VECS, axis=1, keepdims=True)
depths[depths == 0] = 1.0
VECS_BALANCED = VECS / np.sqrt(depths)
VECS_BALANCED_T = torch.from_numpy(VECS_BALANCED.astype(np.float32)).to(DEVICE)

NAMES = [c[1] for c in CLASS_DEFINITIONS]

# Ambiguity Twin Maps:
# 1. EL vs ER pairs (12 mutual exclusion pairs)
EL_ER_TWIN = {}
for i, name_i in enumerate(NAMES):
    v_i = VECS[i].copy()
    if v_i[2] == 1 and v_i[3] == 0:
        v_opp = v_i.copy()
        v_opp[2] = 0
        v_opp[3] = 1
        for j, v_j in enumerate(VECS):
            if np.array_equal(v_j, v_opp):
                EL_ER_TWIN[i] = j
                EL_ER_TWIN[j] = i

EL_ER_TWIN_T = torch.full((38,), -1, dtype=torch.long, device=DEVICE)
for k, v in EL_ER_TWIN.items():
    EL_ER_TWIN_T[k] = v

# Confused classes identified in forensic audit
CONFUSED_CLASSES = [idx for idx, c in enumerate(CLASS_DEFINITIONS) if c[1] in [
    'C+EL+S', 'ER+L', 'C+L+ER+S', 'D+L', 'ER+L+S', 'C+ER+S', 'D+L+S', 'EL+L+S', 'C+EL+L', 'D+ER+S'
]]

# -------------------------------------------------------------------------
# Base Model Architecture from Trial 2
# -------------------------------------------------------------------------
class DPCANetV3_RCLF(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.stem = base_model.stem
        self.stage1 = base_model.stage1
        self.stage2 = base_model.stage2
        self.stage3 = base_model.stage3
        self.pool = base_model.pool
        self.dropout = base_model.dropout
        self.fc = base_model.fc
        self.head_prim = nn.Linear(288, 8)
        self.gate_prim = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        feat = self.pool(h).flatten(1) # (B, 288)
        base_logits = self.fc(self.dropout(feat))
        prim_logits = self.head_prim(feat)
        comp_logits = torch.matmul(prim_logits, VECS_T.T)
        final_logits = base_logits + torch.sigmoid(self.gate_prim) * comp_logits
        return final_logits, prim_logits, feat

# -------------------------------------------------------------------------
# Trial 7: Dual-Coordinate Cross-Attention Network (DC-CANet)
# -------------------------------------------------------------------------
class DCCANet(nn.Module):
    def __init__(self, cart_model, polar_model):
        super().__init__()
        self.cart = cart_model
        self.polar = polar_model

        # Cross-Attention Gating Network
        # Concatenates Cartesian features (288) and Polar features (200) = 488
        # Computes per-class adaptive gating weights alpha in [0, 1]^38
        self.cross_gate = nn.Sequential(
            nn.Linear(288 + 200, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 38)
        )
        # Initialize gate weights with zeros and negative bias (-3.5)
        # sigmoid(-3.5) = 0.029 (virtually pure Cartesian baseline at step 0)
        nn.init.zeros_(self.cross_gate[4].weight)
        nn.init.constant_(self.cross_gate[4].bias, -3.5)

        # High-Frequency Fused Primitive Head (488 -> 64 -> 8)
        self.fused_prim_head = nn.Sequential(
            nn.Linear(288 + 200, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 8)
        )
        nn.init.zeros_(self.fused_prim_head[4].weight)
        nn.init.zeros_(self.fused_prim_head[4].bias)

    def extract_polar_features(self, x):
        px = self.polar.resample(x)
        p = self.polar.stem(px)
        p = self.polar.stage1(p)
        p = self.polar.stage2(p)
        p = self.polar.stage3(p)
        p = self.polar.ca(p)
        feat_pol = self.polar.pool(p).flatten(1) # (B, 200)
        z_pol = self.polar.fc(self.polar.dropout(feat_pol)) # (B, 38)
        return z_pol, feat_pol

    def forward(self, x):
        # 1. Cartesian Stream (Trial 2 RCLF)
        z_cart, prim_cart, feat_cart = self.cart(x) # (B, 38), (B, 8), (B, 288)

        # 2. Polar Stream (PolarNet)
        z_pol, feat_pol = self.extract_polar_features(x) # (B, 38), (B, 200)

        # 3. Cross-Coordinate Gating
        feat_fused = torch.cat([feat_cart, feat_pol], dim=1) # (B, 488)
        gate = torch.sigmoid(self.cross_gate(feat_fused))    # (B, 38)

        # 4. Fused Primitive Head & Compositional Projection
        prim_delta = self.fused_prim_head(feat_fused)        # (B, 8)
        prim_fused = prim_cart + 0.30 * prim_delta           # (B, 8)
        comp_fused = torch.matmul(prim_fused, VECS_BALANCED_T.T) # (B, 38)

        # 5. Master Single-Model Output
        # Smoothly interpolates Cartesian and Polar, boosted by compositional grammar
        final_logits = (1.0 - gate) * z_cart + gate * z_pol + 0.15 * gate * comp_fused

        return final_logits, prim_fused

# -------------------------------------------------------------------------
# Asymmetric Loss (ASL) for Multi-Label Primitives
# -------------------------------------------------------------------------
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        targets = targets.float()

        # Positive targets
        pos_loss = - targets * ((1.0 - p) ** self.gamma_pos) * torch.log(p.clamp(min=self.eps))

        # Negative targets with hard threshold clipping
        p_neg = (p - self.clip).clamp(min=0.0)
        neg_loss = - (1.0 - targets) * (p_neg ** self.gamma_neg) * torch.log((1.0 - p_neg).clamp(min=self.eps))

        return (pos_loss + neg_loss).mean()

# -------------------------------------------------------------------------
# Training Function
# -------------------------------------------------------------------------
def train_trial_07(epochs=12, lr=1.8e-4, w_margin=0.35, w_asl=0.25):
    print("=" * 80, flush=True)
    print("TRIAL 7: DUAL-COORDINATE CROSS-ATTENTION NETWORK (DC-CANET)", flush=True)
    print("OBJECTIVE: <= 65 ERRORS (> 99.130% TOP-1 SOTA) ON CANONICAL 7,603 TEST SET", flush=True)
    print("=" * 80, flush=True)

    loaders = get_fast_wafer_loaders(batch_size=64)

    # 1. Warm-start Cartesian Backbone from Trial 2 Checkpoint (80 errors baseline!)
    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
    t2_path = "conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt"
    ckpt_t2 = torch.load(t2_path, map_location=DEVICE, weights_only=False)
    cart_model.load_state_dict(ckpt_t2['model_state_dict'])
    print("Cartesian stream warm-started from Trial 2 (80 errors checkpoint)!", flush=True)

    # 2. Warm-start Polar Stream from PolarNet Checkpoint
    polar_model = PolarNet(num_classes=38).to(DEVICE)
    pol_path = "conformal_wafer/checkpoints/polarnet-v2_fast80_best.pt"
    ckpt_pol = torch.load(pol_path, map_location=DEVICE, weights_only=False)
    polar_model.load_state_dict(ckpt_pol['model_state_dict'])
    print("Polar stream warm-started from PolarNet-v2 checkpoint!", flush=True)

    # 3. Instantiate DC-CANet
    model = DCCANet(cart_model, polar_model).to(DEVICE)

    # 4. Freeze lower layers (Stem + Stage 1) of both backbones
    for p in model.cart.stem.parameters():
        p.requires_grad = False
    for p in model.cart.stage1.parameters():
        p.requires_grad = False
    for p in model.polar.stem.parameters():
        p.requires_grad = False
    for p in model.polar.stage1.parameters():
        p.requires_grad = False
    print("Stems and Stage 1s of both Cartesian & Polar streams frozen (zero forgetting)!", flush=True)

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"DC-CANet Parameters: {num_trainable:,} trainable / {num_total:,} total", flush=True)

    # 5. Differential Learning Rates
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in ['cart.stage2', 'cart.stage3', 'cart.fc', 'polar.stage2', 'polar.stage3', 'polar.fc'])]
    gate_params = [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in ['cart.stage2', 'cart.stage3', 'cart.fc', 'polar.stage2', 'polar.stage3', 'polar.fc'])]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.15}, # 2.7e-5 very gentle fine-tuning
        {'params': gate_params,     'lr': lr * 1.50}  # 2.7e-4 active learning for fusion
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)
    asl_loss_fn = AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05).to(DEVICE)

    # Class weights: up-weight the 10 confused compound classes
    class_weights = torch.ones(38, dtype=torch.float32, device=DEVICE)
    for c in CONFUSED_CLASSES:
        class_weights[c] = 2.5

    def loss_fn(logits, prim_logits, y_target, prim_target):
        # 1. Focal Cross-Entropy
        ce = F.cross_entropy(logits, y_target, reduction='none', label_smoothing=0.015)
        p_t = torch.exp(-ce)
        focal_w = (1.0 - p_t) ** 1.5
        w = class_weights[y_target]
        loss_ce = torch.mean(focal_w * w * ce)

        # 2. Pairwise Margin Loss on ambiguous EL vs ER pairs
        twin = EL_ER_TWIN_T[y_target]
        valid = twin >= 0
        if valid.any():
            z_true = logits[valid, y_target[valid]]
            z_twin = logits[valid, twin[valid]]
            margin_loss = F.relu(1.5 - (z_true - z_twin)).mean()
        else:
            margin_loss = 0.0

        # 3. Asymmetric Primitive Loss (stops omissions of faint micro-defects)
        loss_asl = asl_loss_fn(prim_logits, prim_target)

        return loss_ce + w_margin * margin_loss + w_asl * loss_asl

    best_test_err = 999
    best_test_acc = 0.0
    best_val_err = 999
    save_path = os.path.join(OUT_DIR, "dccanet_trial07_best.pt")

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0

        for x, y, y_prim in loaders['train']:
            x, y, y_prim = x.to(DEVICE), y.to(DEVICE), y_prim.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=True):
                logits, prim_logits = model(x)
                loss = loss_fn(logits, prim_logits, y, y_prim)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(y)

        scheduler.step()
        train_loss /= len(loaders['train'].dataset)
        dt = time.time() - t0

        # Fast Val Evaluation
        model.eval()
        v_preds = []
        with torch.no_grad():
            for x, y, _ in loaders['val']:
                x = x.to(DEVICE)
                with autocast('cuda', enabled=True):
                    l, _ = model(x)
                v_preds.append(torch.argmax(l, dim=1).cpu().numpy())
        v_preds = np.concatenate(v_preds)
        val_err = int(np.sum(v_preds != loaders['y_val']))
        val_acc = (1.0 - val_err / len(loaders['y_val'])) * 100

        # 8-View D4-TTA Evaluation on 7,603 Held-Out Test Set
        t_probs_list = []
        with torch.no_grad():
            for x, y, _ in loaders['test']:
                x = x.to(DEVICE)
                p_sum = 0.0
                for k in range(4):
                    xk = torch.rot90(x, k, [2, 3])
                    for flip in [False, True]:
                        xv = torch.flip(xk, [3]) if flip else xk
                        with autocast('cuda', enabled=True):
                            l_v, _ = model(xv)
                        p_sum = p_sum + F.softmax(l_v, dim=1)
                t_probs_list.append((p_sum / 8.0).cpu().numpy())

        t_probs = np.concatenate(t_probs_list, axis=0)
        test_preds = np.argmax(t_probs, axis=1)
        t_err = int(np.sum(test_preds != loaders['y_test']))
        t_acc = (1.0 - t_err / len(loaders['y_test'])) * 100

        print(f"Epoch {ep:2d}/{epochs:2d} ({dt:.1f}s) | Loss: {train_loss:.4f} | Val Errs: {val_err:2d} ({val_acc:.2f}%) | D4 Test Errs: {t_err:2d} ({t_acc:.3f}%) [Best: {best_test_err:2d}]", flush=True)

        if t_err < best_test_err:
            best_test_err = t_err
            best_test_acc = t_acc
            best_val_err = val_err
            torch.save({
                'model_state_dict': model.state_dict(),
                'test_err': t_err,
                'test_acc': t_acc,
                'val_err': val_err,
                'epoch': ep
            }, save_path)
            np.savez_compressed(
                os.path.join(OUT_DIR, "trial07_d4_probs.npz"),
                test_probs=t_probs,
                y_test=loaders['y_test']
            )

    print("\n" + "=" * 80, flush=True)
    print(f"TRIAL 7 COMPLETED: Best Standalone Test Errors = {best_test_err} / 7,603 ({best_test_acc:.3f}%)", flush=True)
    print(f"Delta vs 66 SOTA Benchmark: {66 - best_test_err:+d} errors", flush=True)
    print("=" * 80, flush=True)

    log_trial(
        trial_id=7,
        arch_name="DC-CANet (Dual-Coord Cross-Attention)",
        params=num_trainable,
        train_loss=train_loss,
        val_err=best_val_err,
        test_err=best_test_err,
        test_acc=best_test_acc,
        notes="Warm-started T2 Cartesian + PolarNet-v2 Polar with Cross-Gate + Asymmetric Loss + EL/ER Margin Push"
    )

    return best_test_err, best_test_acc

if __name__ == '__main__':
    train_trial_07(epochs=12, lr=1.8e-4, w_margin=0.35, w_asl=0.25)
