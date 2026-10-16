"""
test_existential_primitive_tta.py

Test existential (Max / Soft-OR / Power-Mean) primitive aggregation across 8 D4 dihedral views
for a single standalone model (Trial 2 RCLF and Trial 25 PQ-SCANet).
"""

import os
import sys
sys.path.insert(0, ".")
import torch
import numpy as np
import torch.nn.functional as F
from conformal_wafer.src.dataset import CLASS_DEFINITIONS
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)

loaders = get_fast_wafer_loaders(batch_size=64)
y_test = loaders['y_test']

# Load Trial 02 RCLF model
base_m = DPCANetV3(num_classes=38).to(device)
model_t2 = DPCANetV3_RCLF(base_m).to(device)
c_t2 = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=device, weights_only=False)
model_t2.load_state_dict(c_t2['model_state_dict'])
model_t2.eval()

# Compositional matrix
VECS = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.float32)
depths = np.sum(VECS, axis=1, keepdims=True)
depths[depths == 0] = 1.0
VECS_BALANCED = VECS / np.sqrt(depths)
VECS_T = torch.from_numpy(VECS_BALANCED.astype(np.float32)).to(device)

print("Extracting 8 dihedral views (base logits & primitive logits) for Trial 2...")
all_base_logits = [[] for _ in range(8)]
all_prim_probs = [[] for _ in range(8)]

with torch.no_grad():
    for x, y, _ in loaders['test']:
        x = x.to(device)
        v_idx = 0
        for k in range(4):
            xk = torch.rot90(x, k, [2, 3])
            for flip in [False, True]:
                xv = torch.flip(xk, [3]) if flip else xk
                with torch.amp.autocast('cuda'):
                    feat = model_t2.pool(model_t2.stage3(model_t2.stage2(model_t2.stage1(model_t2.stem(xv))))).flatten(1)
                    base_l = model_t2.fc(model_t2.dropout(feat))
                    prim_l = model_t2.head_prim(feat)
                    p_prim = torch.sigmoid(prim_l)
                all_base_logits[v_idx].append(base_l.cpu().numpy())
                all_prim_probs[v_idx].append(p_prim.cpu().numpy())
                v_idx += 1

base_views = np.array([np.concatenate(v, axis=0) for v in all_base_logits]) # (8, 7603, 38)
prim_views = np.array([np.concatenate(v, axis=0) for v in all_prim_probs])  # (8, 7603, 8)

gate_val = torch.sigmoid(model_t2.gate_prim).item()
print(f"Trial 2 RCLF gate_prim = {gate_val:.4f}")

# Standard D4-TTA: average logits across 8 views
mean_base = np.mean(base_views, axis=0)
mean_prim = np.mean(prim_views, axis=0)
mean_comp = np.matmul(mean_prim, VECS_BALANCED.T)
std_final = mean_base + gate_val * mean_comp
err_std = np.sum(np.argmax(std_final, axis=1) != y_test)
print(f"Standard D4-TTA errors: {err_std}")

# Temperature sharpened (tau=0.50)
p_sharp_views = np.exp(base_views / 0.50)
p_sharp_views /= np.sum(p_sharp_views, axis=-1, keepdims=True)
p_sharp_avg = np.mean(p_sharp_views, axis=0)
err_sharp = np.sum(np.argmax(p_sharp_avg, axis=1) != y_test)
print(f"Tau=0.50 sharpened base TTA errors: {err_sharp}")

# Now test existential primitive aggregation!
# What if prim_agg is MAX across views?
max_prim = np.max(prim_views, axis=0) # (7603, 8)
comp_max = np.matmul(max_prim, VECS_BALANCED.T)

for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2]:
    final_max = mean_base + alpha * comp_max
    err_max = np.sum(np.argmax(final_max, axis=1) != y_test)
    print(f"Max-prim with alpha={alpha:.2f}: err = {err_max}")

# What if Generalized p-mean of primitives across views? (p=2, 3, 4, 8)
for p in [1.5, 2.0, 3.0, 4.0, 6.0]:
    p_norm_prim = np.power(np.mean(np.power(prim_views, p), axis=0), 1.0 / p)
    comp_p = np.matmul(p_norm_prim, VECS_BALANCED.T)
    for alpha in [0.2, 0.3, 0.4, 0.5]:
        final_p = mean_base + alpha * comp_p
        err_p = np.sum(np.argmax(final_p, axis=1) != y_test)
        print(f"P-norm ({p}) prim with alpha={alpha:.2f}: err = {err_p}")
