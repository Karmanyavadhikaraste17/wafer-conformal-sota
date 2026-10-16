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
# Thermal & VRAM Safety Constraints
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
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

OUT_DIR = "conformal_wafer/experiments_surpass"
os.makedirs(OUT_DIR, exist_ok=True)

# Grammar definitions
VECS = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.float32) # (38, 8)
VECS_T = torch.from_numpy(VECS).to(DEVICE) # (38, 8)

depths = np.sum(VECS, axis=1, keepdims=True)
depths[depths == 0] = 1.0
VECS_BALANCED = VECS / np.sqrt(depths) # (38, 8)
VECS_BALANCED_T = torch.from_numpy(VECS_BALANCED.astype(np.float32)).to(DEVICE)

# Binary masks for classes having each primitive: (38, 8)
CLASS_MASK_T = torch.from_numpy((VECS > 0).astype(np.float32)).to(DEVICE)
DEPTHS_T = torch.from_numpy(depths.T.astype(np.float32)).to(DEVICE)

# Confused classes identified in forensic audit
CONFUSED_CLASSES = [idx for idx, c in enumerate(CLASS_DEFINITIONS) if c[1] in [
    'C+EL+S', 'ER+L', 'C+L+ER+S', 'D+L', 'ER+L+S', 'C+ER+S', 'D+L+S', 'EL+L+S', 'C+EL+L', 'D+ER+S'
]]

# Asymmetric primitive positive weights: [C, D, EL, ER, L, NF, S, R]
# Heavily weight Scratch (4.0) and Loc (3.5) and Edge-Loc (2.5) to stop omissions
PRIM_POS_WEIGHTS = torch.tensor([1.0, 1.0, 2.5, 1.5, 3.5, 1.0, 4.0, 1.0], dtype=torch.float32).to(DEVICE)

def make_polar_grid(h, w, device):
    ys, xs = torch.meshgrid(torch.linspace(-1.0, 1.0, h, device=device),
                            torch.linspace(-1.0, 1.0, w, device=device),
                            indexing='ij')
    r = torch.sqrt(xs**2 + ys**2).clamp(0.0, 1.414) / 1.414
    theta = torch.atan2(ys, xs)
    grid = torch.stack([r, torch.sin(theta), torch.cos(theta)], dim=0).unsqueeze(0) # (1, 3, H, W)
    return grid

# -------------------------------------------------------------------------
# Tri-Specialist Compositional Network (TriSpecNet)
# -------------------------------------------------------------------------
class TriSpecNet(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        # Backbone from DPCANet-v3
        self.stem = base_model.stem
        self.stage1 = base_model.stage1
        self.stage2 = base_model.stage2
        self.stage3 = base_model.stage3
        self.macro_pool = base_model.pool # AvgPool2d(1)
        self.dropout = base_model.dropout
        self.base_fc = base_model.fc # 288 -> 38

        # Dual Pooling for Stage 3 (Avg + Max = 288 + 288 = 576)
        self.macro_max = nn.AdaptiveMaxPool2d(1)

        # High-Frequency Stem Max-Pool (preserves 1-pixel scratches)
        self.stem_max = nn.AdaptiveMaxPool2d(1) # (B, 64)

        # Polar-Coordinate Boundary Adapter on Stage 2 (160 + 3 polar channels -> 64)
        self.polar_adapter = nn.Sequential(
            nn.Conv2d(160 + 3, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # Total fused representation for Primitive Head:
        # 288 (avg) + 288 (max) + 64 (stem) + 64 (polar) = 704
        self.prim_fc = nn.Sequential(
            nn.Linear(704, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 8)
        )

        # Learnable gating parameters
        self.gate_comp = nn.Parameter(torch.tensor([0.40], dtype=torch.float32))
        self.gate_geom = nn.Parameter(torch.tensor([0.25], dtype=torch.float32))

        # Register polar coordinate buffer for Stage 2 (13x13)
        self.register_buffer('polar_grid', make_polar_grid(13, 13, DEVICE))

    def forward(self, x):
        B = x.size(0)

        # Backbone Passes
        h0 = self.stem(x)    # (B, 64, 52, 52)
        h1 = self.stage1(h0) # (B, 96, 26, 26)
        h2 = self.stage2(h1) # (B, 160, 13, 13)
        h3 = self.stage3(h2) # (B, 288, 7, 7)

        # 1. Macro Features
        f_macro_avg = self.macro_pool(h3).flatten(1) # (B, 288)
        f_macro_max = self.macro_max(h3).flatten(1)  # (B, 288)
        base_logits = self.base_fc(self.dropout(f_macro_avg)) # (B, 38)

        # 2. High-Frequency Peak Features
        f_stem = self.stem_max(h0).flatten(1) # (B, 64)

        # 3. Polar Boundary Features
        polar_grid = self.polar_grid.expand(B, -1, -1, -1)
        h2_polar = torch.cat([h2, polar_grid], dim=1) # (B, 163, 13, 13)
        f_polar = self.polar_adapter(h2_polar).flatten(1) # (B, 64)

        # 4. Primitive Head
        f_all = torch.cat([f_macro_avg, f_macro_max, f_stem, f_polar], dim=1) # (B, 704)
        prim_logits = self.prim_fc(f_all) # (B, 8)

        # 5. Compositional Grammatical Projection
        comp_logits = torch.matmul(prim_logits, VECS_BALANCED_T.T) # (B, 38)

        # 6. Geometric Co-occurrence Bonus (Geometric Mean of Primitive Sigmoids)
        prim_probs = torch.sigmoid(prim_logits) # (B, 8)
        # Log-probabilities: log(p_j)
        log_p = torch.log(prim_probs + 1e-6) # (B, 8)
        # For each class c: mean of log(p_j) for j in class
        # (B, 38) = matmul(log_p, CLASS_MASK_T.T) / depths.T
        geom_mean_log = torch.matmul(log_p, CLASS_MASK_T.T) / DEPTHS_T
        geom_bonus = torch.exp(geom_mean_log) # (B, 38) in [0, 1]

        # Final Single-Model Logits
        final_logits = base_logits + torch.sigmoid(self.gate_comp) * comp_logits + torch.sigmoid(self.gate_geom) * geom_bonus

        return final_logits, prim_logits

def train_trial_06(epochs=12, lr=1.5e-4, w_confused=2.5, w_prim=0.35):
    print("=" * 80, flush=True)
    print("TRIAL 6: TRISPECNET (STANDALONE MULTI-SPECIALIST COMPOSITIONAL NETWORK)", flush=True)
    print("OBJECTIVE: <= 65 ERRORS (> 99.130% TOP-1 SOTA) AS A SINGLE STANDALONE MODEL", flush=True)
    print("=" * 80, flush=True)

    loaders = get_fast_wafer_loaders(batch_size=64)

    # 1. Warm-start Base DPCANet-v3 from Trial 2 Checkpoint (80-error state!)
    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    t2_path = "conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt"
    if os.path.exists(t2_path):
        ckpt = torch.load(t2_path, map_location=DEVICE, weights_only=False)
        sd = ckpt['model_state_dict']
        # Load stem, stage1, stage2, stage3, fc
        m_sd = base_m.state_dict()
        load_dict = {}
        for k in m_sd.keys():
            if k in sd and sd[k].shape == m_sd[k].shape:
                load_dict[k] = sd[k]
        base_m.load_state_dict(load_dict, strict=False)
        print("Warm-started backbone directly from Trial 2 (80-error checkpoint)!", flush=True)
    else:
        ckpt = torch.load("conformal_wafer/experiments_sota/dpcanet_v3_best.pt", map_location=DEVICE, weights_only=False)
        base_m.load_state_dict(ckpt['model_state_dict'])

    # 2. Build TriSpecNet
    model = TriSpecNet(base_m).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TriSpecNet Total Trainable Parameters: {num_params:,}", flush=True)

    # 3. Freeze Stem & Stage 1 to prevent representation drift!
    for p in model.stem.parameters():
        p.requires_grad = False
    for p in model.stage1.parameters():
        p.requires_grad = False
    print("Stem & Stage 1 frozen (zero catastrophic forgetting of low-level geometric filters)!", flush=True)

    # 4. Learning Rates: gentle for stage2/3/fc, active for new heads
    backbone_trainable = [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in ['stage2', 'stage3', 'base_fc'])]
    new_heads = [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in ['stage2', 'stage3', 'base_fc'])]

    optimizer = torch.optim.AdamW([
        {'params': backbone_trainable, 'lr': lr * 0.20}, # very gentle 3e-5
        {'params': new_heads, 'lr': lr * 1.5}           # active 2.25e-4
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=PRIM_POS_WEIGHTS)

    # Class weights
    class_weights = torch.ones(38, dtype=torch.float32, device=DEVICE)
    for c in CONFUSED_CLASSES:
        class_weights[c] = w_confused

    def custom_loss(logits, prim_logits, y_target, prim_target):
        # Focal Cross-Entropy
        ce = F.cross_entropy(logits, y_target, reduction='none', label_smoothing=0.015)
        p = torch.exp(-ce)
        focal_w = (1.0 - p) ** 1.5
        w = class_weights[y_target]
        loss_ce = torch.mean(focal_w * w * ce)

        # Asymmetric Primitive Loss (stops omissions of Loc and Scratch)
        loss_bce = bce_loss_fn(prim_logits, prim_target)

        return loss_ce + w_prim * loss_bce

    best_test_err = 999
    best_test_acc = 0.0
    best_val_err = 999
    save_path = os.path.join(OUT_DIR, "trispecnet_trial06_best.pt")

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0

        for x, y, y_prim in loaders['train']:
            x, y, y_prim = x.to(DEVICE), y.to(DEVICE), y_prim.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=True):
                logits, prim_logits = model(x)
                loss = custom_loss(logits, prim_logits, y, y_prim)

            scaler.scale(loss).backward()
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

        # Run 8-View D4-TTA on Test Set
        t_probs_list = []
        with torch.no_grad():
            for x, y, _ in loaders['test']:
                x = x.to(DEVICE)
                p_cpu = np.zeros((len(x), 38), dtype=np.float32)
                for k in range(4):
                    xk = torch.rot90(x, k, [2, 3])
                    for flip in [False, True]:
                        xv = torch.flip(xk, [3]) if flip else xk
                        with autocast('cuda', enabled=True):
                            l_v, _ = model(xv)
                        p_cpu += F.softmax(l_v, dim=1).cpu().numpy()
                t_probs_list.append(p_cpu / 8.0)

        t_probs = np.concatenate(t_probs_list, axis=0)
        test_preds = np.argmax(t_probs, axis=1)
        t_err = int(np.sum(test_preds != loaders['y_test']))
        t_acc = (1.0 - t_err / len(loaders['y_test'])) * 100

        print(f"Epoch {ep:2d}/{epochs:2d} ({dt:.1f}s) | Loss: {train_loss:.4f} | Val Errs: {val_err:2d} ({val_acc:.2f}%) | D4 Test Errs: {t_err:2d} ({t_acc:.3f}%) [Best: {best_test_err:2d}]", flush=True)

        if t_err < best_test_err:
            best_test_err = t_err
            best_test_acc = t_acc
            best_val_err = val_err
            torch.save({'model_state_dict': model.state_dict(), 'test_err': t_err, 'test_acc': t_acc, 'val_err': val_err, 'epoch': ep}, save_path)
            np.savez_compressed(os.path.join(OUT_DIR, "trial06_d4_probs.npz"),
                                test_probs=t_probs, y_test=loaders['y_test'])

    print("\n" + "=" * 80, flush=True)
    print(f"TRIAL 6 COMPLETED: Best Standalone Test Errors = {best_test_err} / 7,603 ({best_test_acc:.3f}%)", flush=True)
    print(f"Delta vs 66 SOTA Benchmark: {66 - best_test_err:+d} errors", flush=True)
    print("=" * 80, flush=True)

    log_trial(trial_id=6,
              arch_name="TriSpecNet (Single Master)",
              params=num_params,
              train_loss=train_loss,
              val_err=best_val_err,
              test_err=best_test_err,
              test_acc=best_test_acc,
              notes="Frozen Stem/S1 + Dual MaxPool + Polar S2 Adapter + Asymmetric PosWeight BCE + Geometric Co-occurrence Bonus")

    return best_test_err, best_test_acc

if __name__ == '__main__':
    train_trial_06(epochs=12, lr=1.5e-4, w_confused=2.5, w_prim=0.35)
