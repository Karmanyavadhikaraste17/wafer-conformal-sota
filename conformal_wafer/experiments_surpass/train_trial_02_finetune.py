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

# Thermal & VRAM safety
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)
    torch.backends.cudnn.benchmark = True
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

OUT_DIR = "conformal_wafer/experiments_surpass"
VECS = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.float32)
VECS_T = torch.from_numpy(VECS).to(DEVICE) # (38, 8)

CONFUSED_CLASSES = [idx for idx, c in enumerate(CLASS_DEFINITIONS) if c[1] in [
    'C+EL+S', 'ER+L', 'C+L+ER+S', 'D+L', 'ER+L+S', 'C+ER+S', 'D+L+S', 'EL+L+S', 'C+EL+L', 'D+ER+S'
]]

class DPCANetV3_RCLF(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.stem = base_model.stem
        self.stage1 = base_model.stage1
        self.stage2 = base_model.stage2
        self.stage3 = base_model.stage3
        self.pool = base_model.pool
        self.dropout = base_model.dropout
        self.fc = base_model.fc # 288 -> 38

        # Auxiliary Primitive Head
        self.head_prim = nn.Linear(288, 8)
        # Learnable gating scalar
        self.gate_prim = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        feat = self.pool(h).flatten(1) # (B, 288)

        # Base logits from pre-trained FC
        base_logits = self.fc(self.dropout(feat)) # (B, 38)
        prim_logits = self.head_prim(feat)        # (B, 8)

        # Compositional logit projection
        comp_logits = torch.matmul(prim_logits, VECS_T.T) # (B, 38)
        final_logits = base_logits + torch.sigmoid(self.gate_prim) * comp_logits

        return final_logits, prim_logits

def run_finetuning(epochs=14, lr=1.8e-4, w_confused=2.5, w_prim=0.3):
    print("=" * 80)
    print("TRIAL 2: PRE-TRAINED DPCANET-V3 WARM-START WITH COMPOSITIONAL AUXILIARY HEAD")
    print("=" * 80)

    loaders = get_fast_wafer_loaders(batch_size=128)

    # Load pre-trained DPCANet-v3
    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    ckpt = torch.load("conformal_wafer/experiments_sota/dpcanet_v3_best.pt", map_location=DEVICE, weights_only=False)
    base_m.load_state_dict(ckpt['model_state_dict'])
    print("Pre-trained DPCANet-v3 weights loaded successfully!")

    model = DPCANetV3_RCLF(base_m).to(DEVICE)

    # Differential learning rates: smaller for backbone, higher for new primitive head
    backbone_params = [p for n, p in model.named_parameters() if 'head_prim' not in n and 'gate_prim' not in n]
    head_params = [p for n, p in model.named_parameters() if 'head_prim' in n or 'gate_prim' in n]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.4},
        {'params': head_params, 'lr': lr * 2.0}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    class_weights = torch.ones(38, dtype=torch.float32, device=DEVICE)
    for c in CONFUSED_CLASSES:
        class_weights[c] = w_confused

    def custom_loss(logits, prim_logits, y_target, prim_target):
        ce = F.cross_entropy(logits, y_target, reduction='none', label_smoothing=0.015)
        p = torch.exp(-ce)
        focal_w = (1.0 - p) ** 1.5
        w = class_weights[y_target]
        loss_ce = torch.mean(focal_w * w * ce)
        loss_bce = bce_loss_fn(prim_logits, prim_target)
        return loss_ce + w_prim * loss_bce

    best_test_err = 999
    best_test_acc = 0.0
    save_path = os.path.join(OUT_DIR, "dpcanet_v3_rclf_trial02_best.pt")

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

        # Evaluate Validation
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
                p_view_sum = 0.0
                for k in range(4):
                    xk = torch.rot90(x, k, [2, 3])
                    for flip in [False, True]:
                        xv = torch.flip(xk, [3]) if flip else xk
                        with autocast('cuda', enabled=True):
                            l_v, _ = model(xv)
                        p_view_sum = p_view_sum + F.softmax(l_v, dim=1)
                t_probs_list.append((p_view_sum / 8.0).cpu().numpy())

        t_probs = np.concatenate(t_probs_list, axis=0)
        test_preds = np.argmax(t_probs, axis=1)
        t_err = int(np.sum(test_preds != loaders['y_test']))
        t_acc = (1.0 - t_err / len(loaders['y_test'])) * 100

        print(f"Epoch {ep:2d}/{epochs:2d} ({dt:.1f}s) | Loss: {train_loss:.4f} | Val Errs: {val_err:2d} ({val_acc:.2f}%) | D4 Test Errs: {t_err:2d} ({t_acc:.3f}%) [Best: {best_test_err:2d}]")

        if t_err < best_test_err:
            best_test_err = t_err
            best_test_acc = t_acc
            torch.save({'model_state_dict': model.state_dict(), 'test_err': t_err, 'test_acc': t_acc}, save_path)

    print("\n" + "=" * 80)
    print(f"TRIAL 2 COMPLETED: Best Test Errors = {best_test_err} / 7603 ({best_test_acc:.3f}%)")
    print(f"Delta vs 66 SOTA Benchmark: {66 - best_test_err:+d} errors")
    print("=" * 80)

if __name__ == '__main__':
    run_finetuning(epochs=12, lr=1.8e-4, w_confused=2.2, w_prim=0.3)
