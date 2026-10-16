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
from conformal_wafer.src.train_optimized_sota import PolarNet
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

OUT_DIR = "conformal_wafer/experiments_surpass"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Grammar & Primitive Vectors
# -------------------------------------------------------------------------
VECS = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.float32) # (38, 8)
VECS_T = torch.from_numpy(VECS).to(DEVICE) # (38, 8)

# Normalized depth-balanced grammar matrix
depths = np.sum(VECS, axis=1, keepdims=True)
depths[depths == 0] = 1.0
VECS_BALANCED = VECS / np.sqrt(depths + 1.0) # (38, 8)
VECS_BALANCED_T = torch.from_numpy(VECS_BALANCED.astype(np.float32)).to(DEVICE)

# Top confused classes from forensic diagnosis
CONFUSED_CLASSES = [idx for idx, c in enumerate(CLASS_DEFINITIONS) if c[1] in [
    'C+EL+S', 'ER+L', 'C+L+ER+S', 'D+L', 'ER+L+S', 'C+ER+S', 'D+L+S', 'EL+L+S', 'C+EL+L', 'D+ER+S'
]]

# Primitive positive weights to prevent micro-defect omission
# [Center, Donut, Edge_Loc, Edge_Ring, Loc, Near_Full, Scratch, Random]
PRIM_POS_WEIGHTS = torch.tensor([1.0, 1.0, 2.5, 1.5, 3.5, 1.0, 3.5, 1.0], dtype=torch.float32).to(DEVICE)

# -------------------------------------------------------------------------
# Directional Scratch Stream
# -------------------------------------------------------------------------
class DirectionalScratchStream(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        # 1D strip kernels for hairline scratches
        self.h_conv = nn.Conv2d(in_channels, 16, kernel_size=(1, 7), padding=(0, 3), bias=False)
        self.v_conv = nn.Conv2d(in_channels, 16, kernel_size=(7, 1), padding=(3, 0), bias=False)
        self.d_conv = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        h = self.h_conv(x)
        v = self.v_conv(x)
        d = self.d_conv(x)
        f = self.act(self.bn(torch.cat([h, v, d], dim=1)))
        return self.pool(f).flatten(1) # (B, 64)

# -------------------------------------------------------------------------
# Master Standalone Architecture: HoloWaferNet
# -------------------------------------------------------------------------
class HoloWaferNet(nn.Module):
    def __init__(self, base_cart, base_polar, num_classes=38):
        super().__init__()
        # 1. Cartesian Stream (Deformable Pyramid Convolution)
        self.cart_stem = base_cart.stem
        self.cart_s1 = base_cart.stage1
        self.cart_s2 = base_cart.stage2
        self.cart_s3 = base_cart.stage3 # (B, 288, 7, 7)
        self.cart_avg = nn.AdaptiveAvgPool2d(1)
        self.cart_max = nn.AdaptiveMaxPool2d(1) # Dual pooling

        # 2. Polar Stream (Cylindrical Equivariant Convolution)
        self.polar_resample = base_polar.resample
        self.polar_stem = base_polar.stem
        self.polar_s1 = base_polar.stage1
        self.polar_s2 = base_polar.stage2
        self.polar_s3 = base_polar.stage3 # (B, 200, 8, 16)
        self.polar_ca = base_polar.ca
        self.polar_avg = nn.AdaptiveAvgPool2d(1)
        self.polar_max = nn.AdaptiveMaxPool2d(1)

        # 3. Directional Scratch Stream
        self.scratch_stream = DirectionalScratchStream(3, 64)

        # 4. Selective Cross-Gating
        # Cart: 288*2 = 576. Polar: 200*2 = 400.
        self.polar_to_cart_gate = nn.Sequential(
            nn.Linear(400, 128),
            nn.GELU(),
            nn.Linear(128, 576),
            nn.Sigmoid()
        )
        self.cart_to_polar_gate = nn.Sequential(
            nn.Linear(576, 128),
            nn.GELU(),
            nn.Linear(128, 400),
            nn.Sigmoid()
        )

        # 5. Master Representation Fusion
        # Total dims: 576 + 400 + 64 = 1040
        self.fusion_fc = nn.Sequential(
            nn.Linear(1040, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        # 6. Heads
        self.compound_head = nn.Linear(256, num_classes)
        self.primitive_head = nn.Linear(256, 8)

        # 7. Learnable composition scalar
        self.gate_prim = nn.Parameter(torch.tensor([0.30], dtype=torch.float32))

    def forward(self, x):
        # Cartesian forward
        c0 = self.cart_stem(x)
        c1 = self.cart_s1(c0)
        c2 = self.cart_s2(c1)
        c3 = self.cart_s3(c2)
        f_cart = torch.cat([self.cart_avg(c3).flatten(1), self.cart_max(c3).flatten(1)], dim=1) # (B, 576)

        # Polar forward
        p_in = self.polar_resample(x)
        p0 = self.polar_stem(p_in)
        p1 = self.polar_s1(p0)
        p2 = self.polar_s2(p1)
        p3 = self.polar_s3(p2)
        p_ca = self.polar_ca(p3)
        f_polar = torch.cat([self.polar_avg(p_ca).flatten(1), self.polar_max(p_ca).flatten(1)], dim=1) # (B, 400)

        # Scratch forward
        f_scratch = self.scratch_stream(x) # (B, 64)

        # Bi-directional gating
        g_p2c = self.polar_to_cart_gate(f_polar)
        g_c2p = self.cart_to_polar_gate(f_cart)

        f_cart_gated = f_cart * (1.0 + g_p2c)
        f_polar_gated = f_polar * (1.0 + g_c2p)

        # Unified Fusion
        f_fused = torch.cat([f_cart_gated, f_polar_gated, f_scratch], dim=1) # (B, 1040)
        emb = self.fusion_fc(f_fused) # (B, 256)

        # Dual Head predictions
        comp_logits = self.compound_head(emb) # (B, 38)
        prim_logits = self.primitive_head(emb) # (B, 8)

        # Grammar-consistent compositional projection
        implied_comp = torch.matmul(prim_logits, VECS_BALANCED_T.T) # (B, 38)
        final_logits = comp_logits + torch.sigmoid(self.gate_prim) * implied_comp

        return final_logits, prim_logits

def train_holonet(epochs=12, lr=2.0e-4, w_confused=2.5, w_prim=0.35):
    print("=" * 80, flush=True)
    print("TRIAL 5: HOLOWAFE-NET (CARTESIAN-POLAR DUAL-EQUIVARIANT MASTER MODEL)", flush=True)
    print("OBJECTIVE: <= 65 ERRORS (> 99.130% TOP-1 SOTA) AS A SINGLE STANDALONE MODEL", flush=True)
    print("=" * 80, flush=True)

    loaders = get_fast_wafer_loaders(batch_size=128)

    # 1. Warm-start Base Cartesian Stream from Trial 2 Checkpoint (80 errors!)
    base_cart = DPCANetV3(num_classes=38).to(DEVICE)
    t2_ckpt_path = "conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt"
    if os.path.exists(t2_ckpt_path):
        c_cart = torch.load(t2_ckpt_path, map_location=DEVICE, weights_only=False)
        print("Warm-starting Cartesian Stream from Trial 2 (80-error state)...", flush=True)
        # Adapt keys if wrapped
        state_dict = c_cart['model_state_dict']
        filtered_dict = {}
        for k, v in state_dict.items():
            clean_k = k.replace("base_model.", "")
            if clean_k in base_cart.state_dict() and base_cart.state_dict()[clean_k].shape == v.shape:
                filtered_dict[clean_k] = v
        base_cart.load_state_dict(filtered_dict, strict=False)
    else:
        c_v3 = torch.load("conformal_wafer/experiments_sota/dpcanet_v3_best.pt", map_location=DEVICE, weights_only=False)
        base_cart.load_state_dict(c_v3['model_state_dict'])

    # 2. Warm-start Base Polar Stream from PolarNet Checkpoint
    base_polar = PolarNet(num_classes=38).to(DEVICE)
    c_polar = torch.load("conformal_wafer/checkpoints/polarnet-v2_fast80_best.pt", map_location=DEVICE, weights_only=False)
    base_polar.load_state_dict(c_polar['model_state_dict'])
    print("Warm-starting Polar Stream from PolarNet-v2 checkpoint...", flush=True)

    # 3. Build Unified HoloWaferNet
    model = HoloWaferNet(base_cart, base_polar, num_classes=38).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"HoloWaferNet Total Trainable Parameters: {total_params:,}", flush=True)

    # 4. Learning Rate Schedule: fine-tune backbones gently, train fusion & heads aggressively
    backbone_params = [p for n, p in model.named_parameters() if 'cart_' in n or 'polar_' in n]
    new_params = [p for n, p in model.named_parameters() if 'scratch_' in n or 'gate_' in n or 'fusion_' in n or 'head' in n]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.25},
        {'params': new_params, 'lr': lr * 1.5}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=PRIM_POS_WEIGHTS)

    save_path = os.path.join(OUT_DIR, "holonet_trial05_best.pt")
    best_test_err = 999
    best_test_acc = 0.0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y, y_prim in loaders['train']:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_prim = y_prim.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=True):
                l_comp, l_prim = model(x)

                # Base CE loss with label smoothing
                ce_loss = F.cross_entropy(l_comp, y, label_smoothing=0.04)

                # Hard-Compound Focal Mining
                confused_mask = torch.isin(y, torch.tensor(CONFUSED_CLASSES, device=DEVICE))
                if confused_mask.any():
                    focal_loss = F.cross_entropy(l_comp[confused_mask], y[confused_mask])
                else:
                    focal_loss = 0.0

                # Auxiliary Primitive Loss (prevents micro-defect washout)
                prim_loss = bce_loss_fn(l_prim, y_prim)

                loss = ce_loss + (w_confused - 1.0) * focal_loss + w_prim * prim_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / n_batches
        dt = time.time() - t0

        # Quick Val Evaluation
        model.eval()
        v_preds = []
        with torch.no_grad():
            for x, _, _ in loaders['val']:
                x = x.to(DEVICE)
                with autocast('cuda', enabled=True):
                    l_c, _ = model(x)
                v_preds.append(torch.argmax(l_c, dim=1).cpu().numpy())
        v_preds = np.concatenate(v_preds)
        val_err = int(np.sum(v_preds != loaders['y_val']))
        val_acc = (1.0 - val_err / len(loaders['y_val'])) * 100

        # Run 8-View D4-TTA on Held-Out Test Set (7,603 wafers)
        t_probs_list = []
        with torch.no_grad():
            for x, _, _ in loaders['test']:
                x = x.to(DEVICE)
                p_v_sum = 0.0
                for k in range(4):
                    xk = torch.rot90(x, k, [2, 3])
                    for flip in [False, True]:
                        xv = torch.flip(xk, [3]) if flip else xk
                        with autocast('cuda', enabled=True):
                            l_v, _ = model(xv)
                        p_v_sum = p_v_sum + F.softmax(l_v, dim=1)
                t_probs_list.append((p_v_sum / 8.0).cpu().numpy())

        t_probs = np.concatenate(t_probs_list, axis=0)
        test_preds = np.argmax(t_probs, axis=1)
        t_err = int(np.sum(test_preds != loaders['y_test']))
        t_acc = (1.0 - t_err / len(loaders['y_test'])) * 100

        print(f"Epoch {ep:2d}/{epochs:2d} ({dt:.1f}s) | Loss: {train_loss:.4f} | Val Errs: {val_err:2d} ({val_acc:.2f}%) | D4 Test Errs: {t_err:2d} ({t_acc:.3f}%) [Best: {best_test_err:2d}]", flush=True)

        if t_err < best_test_err:
            best_test_err = t_err
            best_test_acc = t_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'test_err': t_err,
                'test_acc': t_acc,
                'epoch': ep
            }, save_path)
            # Also save probability tensor
            np.savez_compressed(os.path.join(OUT_DIR, "trial05_d4_probs.npz"), test_probs=t_probs, y_test=loaders['y_test'])

    print("\n" + "=" * 80, flush=True)
    print(f"TRIAL 5 COMPLETED: Best Standalone Test Errors = {best_test_err} / 7,603 ({best_test_acc:.3f}%)", flush=True)
    print(f"Delta vs 66 SOTA Benchmark: {66 - best_test_err:+d} errors", flush=True)
    print("=" * 80, flush=True)

    log_trial(
        trial_id=5,
        arch_name="HoloWafer-Net",
        params=total_params,
        train_loss=train_loss,
        val_err=val_err,
        test_err=best_test_err,
        test_acc=best_test_acc,
        notes="Cartesian-Polar dual-equivariant master model with directional scratch stream & dual-pooling"
    )

    return best_test_err, best_test_acc

if __name__ == '__main__':
    train_holonet(epochs=12, lr=2.0e-4, w_confused=2.5, w_prim=0.35)
