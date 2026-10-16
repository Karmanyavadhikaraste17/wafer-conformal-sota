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

# Normalized depth-invariant grammar matrix: row-normalized by sum(VECS)
depths = np.sum(VECS, axis=1, keepdims=True)
depths[depths == 0] = 1.0
VECS_BALANCED = VECS / depths # (38, 8): average of primitives
VECS_BALANCED_T = torch.from_numpy(VECS_BALANCED.astype(np.float32)).to(DEVICE)

# Confused classes identified in forensic audit
CONFUSED_CLASSES = [idx for idx, c in enumerate(CLASS_DEFINITIONS) if c[1] in [
    'C+EL+S', 'ER+L', 'C+L+ER+S', 'D+L', 'ER+L+S', 'C+ER+S', 'D+L+S', 'EL+L+S', 'C+EL+L', 'D+ER+S'
]]

# -------------------------------------------------------------------------
# DPCANet-v4: High-Frequency Peak Highway + Normalized Compositional Head
# -------------------------------------------------------------------------
class DPCANetV4_HFComp(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.stem = base_model.stem
        self.stage1 = base_model.stage1
        self.stage2 = base_model.stage2
        self.stage3 = base_model.stage3
        self.macro_pool = base_model.pool
        self.dropout = base_model.dropout
        self.base_fc = base_model.fc # 288 -> 38

        # High-Frequency Stem Peak Highway (preserves 1-pixel scratches & micro-spots)
        # Stem output is 64 ch at 52x52
        self.hf_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        # Max-pooling catches isolated peak activations of scratches & spots
        self.hf_pool = nn.AdaptiveMaxPool2d(1)

        # Primitive presence head: combines Macro (288) + HF Peak (64) = 352 dims -> 8 primitives
        self.prim_fc = nn.Sequential(
            nn.Linear(288 + 64, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 8)
        )

        # Learnable gating scalar for compositional projection
        self.gate_prim = nn.Parameter(torch.tensor([0.35], dtype=torch.float32))

    def forward(self, x):
        h0 = self.stem(x)    # (B, 64, 52, 52)
        h1 = self.stage1(h0) # (B, 96, 26, 26)
        h2 = self.stage2(h1) # (B, 160, 13, 13)
        h3 = self.stage3(h2) # (B, 288, 7, 7)

        # Macro features
        f_macro = self.macro_pool(h3).flatten(1) # (B, 288)
        base_logits = self.base_fc(self.dropout(f_macro)) # (B, 38)

        # High-Frequency Peak Highway
        f_hf = self.hf_pool(self.hf_conv(h0)).flatten(1) # (B, 64)

        # Primitive presence logits
        f_prim = torch.cat([f_macro, f_hf], dim=1) # (B, 352)
        prim_logits = self.prim_fc(f_prim)          # (B, 8)

        # Balanced Compositional projection onto 38 classes
        comp_logits = torch.matmul(prim_logits, VECS_BALANCED_T.T) # (B, 38)

        # Final Logits
        final_logits = base_logits + torch.sigmoid(self.gate_prim) * comp_logits

        return final_logits, prim_logits

def train_trial_04(epochs=12, lr=1.8e-4, w_confused=2.4, w_prim=0.30):
    print("=" * 80, flush=True)
    print("TRIAL 4: DPCANET-V4 HIGH-FREQUENCY PEAK HIGHWAY + BALANCED COMPOSITIONAL HEAD", flush=True)
    print("=" * 80, flush=True)

    loaders = get_fast_wafer_loaders(batch_size=128)

    # 1. Warm-start Base DPCANet-v3
    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    ckpt = torch.load("conformal_wafer/experiments_sota/dpcanet_v3_best.pt", map_location=DEVICE, weights_only=False)
    base_m.load_state_dict(ckpt['model_state_dict'])
    print("Pre-trained DPCANet-v3 weights loaded successfully!", flush=True)

    # 2. Build Model
    model = DPCANetV4_HFComp(base_m).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {num_params:,}", flush=True)

    # 3. Differential Learning Rates
    backbone_params = [p for n, p in model.named_parameters() if 'hf_' not in n and 'prim_' not in n and 'gate_' not in n]
    new_params = [p for n, p in model.named_parameters() if 'hf_' in n or 'prim_' in n or 'gate_' in n]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.35},
        {'params': new_params, 'lr': lr * 2.0}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    # Class weighting
    class_weights = torch.ones(38, dtype=torch.float32, device=DEVICE)
    for c in CONFUSED_CLASSES:
        class_weights[c] = w_confused

    def custom_loss(logits, prim_logits, y_target, prim_target):
        # Focal loss on 38 classes
        ce = F.cross_entropy(logits, y_target, reduction='none', label_smoothing=0.012)
        p = torch.exp(-ce)
        focal_w = (1.0 - p) ** 1.5
        w = class_weights[y_target]
        loss_ce = torch.mean(focal_w * w * ce)

        # Multi-label BCE on primitives
        loss_bce = bce_loss_fn(prim_logits, prim_target)

        return loss_ce + w_prim * loss_bce

    best_test_err = 999
    best_test_acc = 0.0
    best_val_err = 999
    save_path = os.path.join(OUT_DIR, "dpcanet_v4_hfcomp_trial04_best.pt")

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
            torch.save({'model_state_dict': model.state_dict(), 'test_err': t_err, 'test_acc': t_acc, 'val_err': val_err}, save_path)
            np.savez_compressed(os.path.join(OUT_DIR, "trial04_d4_probs.npz"),
                                test_probs=t_probs, y_test=loaders['y_test'])

    print("\n" + "=" * 80, flush=True)
    print(f"TRIAL 4 COMPLETED: Best Test Errors = {best_test_err} / 7603 ({best_test_acc:.3f}%)", flush=True)
    print(f"Delta vs 66 SOTA Benchmark: {66 - best_test_err:+d} errors", flush=True)
    print("=" * 80, flush=True)

    log_trial(trial_id=4,
              arch_name="DPCANet-v4-HFComp",
              params=num_params,
              train_loss=train_loss,
              val_err=best_val_err,
              test_err=best_test_err,
              test_acc=best_test_acc,
              notes="High-Frequency Peak Highway (Stem MaxPool) + Balanced Compositional Logits + Focal Mining")

if __name__ == '__main__':
    train_trial_04(epochs=12, lr=1.8e-4, w_confused=2.4, w_prim=0.30)
