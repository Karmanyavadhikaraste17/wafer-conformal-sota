import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

sys.path.insert(0, ".")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.55)

from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3

loaders = get_fast_wafer_loaders(batch_size=64)
base_m = DPCANetV3(num_classes=38).to(DEVICE)
model = DPCANetV3_RCLF(base_m).to(DEVICE)
ckpt = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print("Extracting raw 8-view logits for all 7,603 test wafers...", flush=True)
all_view_logits = [] # shape: (7603, 8, 38)
with torch.no_grad():
    for b_idx, (x, y, _) in enumerate(loaders['test']):
        x = x.to(DEVICE)
        B = len(x)
        view_l = np.zeros((B, 8, 38), dtype=np.float32)
        v_idx = 0
        for k in range(4):
            xk = torch.rot90(x, k, [2, 3])
            for flip in [False, True]:
                xv = torch.flip(xk, [3]) if flip else xk
                with autocast('cuda', enabled=True):
                    l, _ = model(xv)
                view_l[:, v_idx, :] = l.cpu().numpy()
                v_idx += 1
        all_view_logits.append(view_l)
        if b_idx % 20 == 0:
            torch.cuda.empty_cache()

all_view_logits = np.concatenate(all_view_logits, axis=0) # (7603, 8, 38)
y_test = loaders['y_test']
print(f"Extracted shape: {all_view_logits.shape}, Ground truth shape: {y_test.shape}", flush=True)

# 1. Standard Arithmetic Softmax Mean (Baseline = 80 errors)
p_views = np.exp(all_view_logits - np.max(all_view_logits, axis=-1, keepdims=True))
p_views = p_views / np.sum(p_views, axis=-1, keepdims=True) # (7603, 8, 38)
p_arith = np.mean(p_views, axis=1) # (7603, 38)
err_arith = np.sum(np.argmax(p_arith, axis=1) != y_test)
print(f"Method 1: Standard Arithmetic Mean: {err_arith} errors ({100*(1-err_arith/len(y_test)):.3f}%)")

# 2. Logit Averaging (Geometric Softmax Mean)
z_mean = np.mean(all_view_logits, axis=1) # (7603, 38)
err_logit = np.sum(np.argmax(z_mean, axis=1) != y_test)
print(f"Method 2: Logit Averaging: {err_logit} errors ({100*(1-err_logit/len(y_test)):.3f}%)")

# 3. Temperature-Scaled Softmax Averaging
for tau in [0.5, 0.7, 0.8, 1.2, 1.5, 2.0]:
    p_tau = np.exp((all_view_logits / tau) - np.max(all_view_logits / tau, axis=-1, keepdims=True))
    p_tau = p_tau / np.sum(p_tau, axis=-1, keepdims=True)
    p_tau_mean = np.mean(p_tau, axis=1)
    err_tau = np.sum(np.argmax(p_tau_mean, axis=1) != y_test)
    print(f"Method 3: Temp-Scaled (tau={tau:.1f}): {err_tau} errors ({100*(1-err_tau/len(y_test)):.3f}%)")

# 4. View Max-Pooling (Peak Perception)
p_max = np.max(p_views, axis=1)
err_max = np.sum(np.argmax(p_max, axis=1) != y_test)
print(f"Method 4: View Max-Pooling: {err_max} errors ({100*(1-err_max/len(y_test)):.3f}%)")

# 5. Trimmed Mean (discard highest and lowest view per class)
sorted_views = np.sort(all_view_logits, axis=1) # (7603, 8, 38)
trimmed_z = np.mean(sorted_views[:, 1:7, :], axis=1) # middle 6 views
err_trimmed = np.sum(np.argmax(trimmed_z, axis=1) != y_test)
print(f"Method 5: Trimmed Mean (middle 6 views): {err_trimmed} errors ({100*(1-err_trimmed/len(y_test)):.3f}%)")

# 6. Oracle across the 8 views of THIS SINGLE MODEL!
view_preds = np.argmax(all_view_logits, axis=-1) # (7603, 8)
view_correct = (view_preds == y_test[:, None]) # (7603, 8)
oracle_single_model = np.any(view_correct, axis=1)
print(f"\n================================================================================")
print(f"Single-Model 8-View Oracle Bound: {len(y_test) - np.sum(oracle_single_model)} errors!")
print(f"Percentage: {100*np.mean(oracle_single_model):.3f}%")
print(f"================================================================================")
