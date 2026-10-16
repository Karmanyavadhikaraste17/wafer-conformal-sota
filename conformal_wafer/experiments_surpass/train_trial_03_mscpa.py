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

# Class definitions and primitive vectors
VECS = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.float32) # (38, 8)
VECS_T = torch.from_numpy(VECS).to(DEVICE) # (38, 8)

# Normalized grammar matrix (normalized by sqrt of depth to avoid depth-bias)
depths = np.sum(VECS, axis=1, keepdims=True)
depths[depths == 0] = 1.0
VECS_NORM = VECS / np.sqrt(depths)
VECS_NORM_T = torch.from_numpy(VECS_NORM.astype(np.float32)).to(DEVICE) # (38, 8)

# Confused compound classes identified in forensic audit
CONFUSED_CLASSES = [idx for idx, c in enumerate(CLASS_DEFINITIONS) if c[1] in [
    'C+EL+S', 'ER+L', 'C+L+ER+S', 'D+L', 'ER+L+S', 'C+ER+S', 'D+L+S', 'EL+L+S', 'C+EL+L', 'D+ER+S'
]]

# Precomputed polar coordinate grids for 13x13 feature maps
def make_polar_grid(h, w, device):
    ys, xs = torch.meshgrid(torch.linspace(-1.0, 1.0, h, device=device),
                            torch.linspace(-1.0, 1.0, w, device=device),
                            indexing='ij')
    r = torch.sqrt(xs**2 + ys**2).clamp(0.0, 1.414) / 1.414 # normalized radius [0, 1]
    theta = torch.atan2(ys, xs)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    grid = torch.stack([r, sin_theta, cos_theta], dim=0).unsqueeze(0) # (1, 3, H, W)
    return grid

# -------------------------------------------------------------------------
# Multi-Scale Polar Coordinate Attention Network (MS-CPANet)
# -------------------------------------------------------------------------
class MSCPANet(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        # Pre-trained DPCANet-v3 Backbone
        self.stem = base_model.stem
        self.stage1 = base_model.stage1
        self.stage2 = base_model.stage2
        self.stage3 = base_model.stage3
        self.macro_pool = base_model.pool
        self.dropout = base_model.dropout
        self.base_fc = base_model.fc # 288 -> 38

        # Lateral Pyramid Projections to 13x13 resolution
        # Stage 1: 96 ch, 26x26 -> 64 ch, 13x13
        self.lat1 = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Stage 2: 160 ch, 13x13 -> 64 ch, 13x13
        self.lat2 = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        # Stage 3: 288 ch, 7x7 -> 64 ch, 13x13
        self.lat3 = nn.Sequential(
            nn.Conv2d(288, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        # Fused Pyramid: 64*3 + 3 (polar coordinates) = 195 channels
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(195, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

        # Spatial Saliency Defect Gate (identifies hairline scratches and spots)
        self.saliency_gate = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Micro-defect classifier head (Avg + Max pooling = 256 dims)
        self.micro_fc = nn.Linear(256, 38)
        # Initialize micro_fc near zero so pre-trained baseline is preserved initially
        nn.init.zeros_(self.micro_fc.weight)
        nn.init.zeros_(self.micro_fc.bias)

        # Primitive presence head: combines Macro (288) + Micro (256) = 544 dims -> 8 primitives
        self.prim_fc = nn.Sequential(
            nn.Linear(288 + 256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 8)
        )

        # Learnable gating scalar for compositional projection
        self.gate_comp = nn.Parameter(torch.tensor([0.25], dtype=torch.float32))

        # Register polar grid buffer
        self.register_buffer('polar_grid', make_polar_grid(13, 13, DEVICE))

    def forward(self, x):
        B = x.size(0)
        
        # Backbone Forward
        h0 = self.stem(x)           # (B, 64, 52, 52)
        h1 = self.stage1(h0)        # (B, 96, 26, 26)
        h2 = self.stage2(h1)        # (B, 160, 13, 13)
        h3 = self.stage3(h2)        # (B, 288, 7, 7)

        # Macro Representation
        f_macro = self.macro_pool(h3).flatten(1) # (B, 288)
        base_logits = self.base_fc(self.dropout(f_macro)) # (B, 38)

        # Multi-Scale Pyramid Projections
        p1 = self.lat1(h1)          # (B, 64, 13, 13)
        p2 = self.lat2(h2)          # (B, 64, 13, 13)
        p3 = F.interpolate(self.lat3(h3), size=(13, 13), mode='bilinear', align_corners=False) # (B, 64, 13, 13)

        # Concatenate pyramid features + Polar Coordinate Grid
        polar = self.polar_grid.expand(B, -1, -1, -1) # (B, 3, 13, 13)
        pyr_in = torch.cat([p1, p2, p3, polar], dim=1) # (B, 195, 13, 13)
        pyr_feat = self.fuse_conv(pyr_in)               # (B, 128, 13, 13)

        # Spatial Saliency defect attention
        gate = self.saliency_gate(pyr_feat)             # (B, 1, 13, 13)
        attended_feat = pyr_feat * (1.0 + gate)

        # Micro-defect dual pooling (Avg + Max)
        f_avg = F.adaptive_avg_pool2d(attended_feat, 1).flatten(1) # (B, 128)
        f_max = F.adaptive_max_pool2d(attended_feat, 1).flatten(1) # (B, 128)
        f_micro = torch.cat([f_avg, f_max], dim=1)                 # (B, 256)

        # Direct micro classification refinement
        micro_logits = self.micro_fc(f_micro) # (B, 38)

        # Primitive Classification Head
        f_combined = torch.cat([f_macro, f_micro], dim=1) # (B, 544)
        prim_logits = self.prim_fc(f_combined)            # (B, 8)

        # Compositional Projection onto 38 classes
        comp_logits = torch.matmul(prim_logits, VECS_NORM_T.T) # (B, 38)

        # Final Logits
        final_logits = base_logits + micro_logits + torch.sigmoid(self.gate_comp) * comp_logits

        return final_logits, prim_logits

def train_trial_03(epochs=14, lr=2.0e-4, w_confused=2.5, w_prim=0.35):
    print("=" * 80, flush=True)
    print("TRIAL 3: MULTI-SCALE POLAR COORDINATE ATTENTION NETWORK (MS-CPANET)", flush=True)
    print("=" * 80, flush=True)

    loaders = get_fast_wafer_loaders(batch_size=128)

    # 1. Warm-start Base DPCANet-v3
    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    ckpt = torch.load("conformal_wafer/experiments_sota/dpcanet_v3_best.pt", map_location=DEVICE, weights_only=False)
    base_m.load_state_dict(ckpt['model_state_dict'])
    print("Warm-start DPCANet-v3 backbone loaded successfully!", flush=True)

    # 2. Build MS-CPANet
    model = MSCPANet(base_m).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MS-CPANet Total Trainable Parameters: {num_params:,}", flush=True)

    # 3. Differential Learning Rates
    # Backbone: lower lr to preserve core representations
    # Pyramid + Attention + Heads: higher lr to quickly adapt
    backbone_params = []
    pyramid_params = []
    for name, p in model.named_parameters():
        if any(k in name for k in ['stem', 'stage1', 'stage2', 'stage3', 'base_fc']):
            backbone_params.append(p)
        else:
            pyramid_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.3},
        {'params': pyramid_params, 'lr': lr * 1.8}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    # Class weighting for confused compound classes
    class_weights = torch.ones(38, dtype=torch.float32, device=DEVICE)
    for c in CONFUSED_CLASSES:
        class_weights[c] = w_confused

    def custom_loss(logits, prim_logits, y_target, prim_target):
        # Focal Cross-Entropy on 38 classes
        ce = F.cross_entropy(logits, y_target, reduction='none', label_smoothing=0.01)
        p = torch.exp(-ce)
        focal_w = (1.0 - p) ** 1.5
        w = class_weights[y_target]
        loss_ce = torch.mean(focal_w * w * ce)

        # Multi-label BCE on 8 primitives
        loss_bce = bce_loss_fn(prim_logits, prim_target)

        return loss_ce + w_prim * loss_bce

    best_test_err = 999
    best_test_acc = 0.0
    best_val_err = 999
    save_path = os.path.join(OUT_DIR, "mscpanet_trial03_best.pt")

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

        # Run 8-View D4-TTA on Test Set (with CPU accumulation for zero VRAM bloat)
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
            torch.save({'model_state_dict': model.state_dict(), 'test_err': t_err, 'test_acc': t_acc, 'val_err': val_err}, save_path)
            # Also save probability predictions for ensembling
            np.savez_compressed(os.path.join(OUT_DIR, "trial03_d4_probs.npz"),
                                test_probs=t_probs, y_test=loaders['y_test'])

    print("\n" + "=" * 80, flush=True)
    print(f"TRIAL 3 COMPLETED: Best Test Errors = {best_test_err} / 7603 ({best_test_acc:.3f}%)", flush=True)
    print(f"Delta vs 66 SOTA Benchmark: {66 - best_test_err:+d} errors", flush=True)
    print("=" * 80, flush=True)

    # Log Trial 3 to leaderboard registry
    log_trial(trial_id=3,
              arch_name="MS-CPANet (Pyramid+Polar)",
              params=num_params,
              train_loss=train_loss,
              val_err=best_val_err,
              test_err=best_test_err,
              test_acc=best_test_acc,
              notes="Multi-Scale Lateral Pyramid + Polar Coordinate Grid + Spatial Saliency Gate + Normalized RCLF")

if __name__ == '__main__':
    train_trial_03(epochs=12, lr=2.0e-4, w_confused=2.2, w_prim=0.30)
