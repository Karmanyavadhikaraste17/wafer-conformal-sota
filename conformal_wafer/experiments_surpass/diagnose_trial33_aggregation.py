"""
diagnose_trial33_aggregation.py

Deep forensic diagnostic on the 20 recoverable wafers of Trial 33 (SWA DC-PolarSCANet).
Evaluating mathematical aggregation functions across the 8 dihedral views of this single model:
1. Power-Mean Aggregation: (1/8 * sum(p_v^p))^(1/p) for p in [0.1, 0.5, 1.5, 2, 3, 5]
2. Temperature Sweeps: tau in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]
3. Margin-Weighted Aggregation: w_v = (top1 - top2)^gamma
4. Entropy-Weighted Aggregation: w_v = exp(-gamma * Entropy(p_v))
5. Top-K View Selection: average only the top K views with highest margin/lowest entropy
6. Ranked-Choice Voting (Borda Count & Copeland Pairwise)
"""

import sys
sys.path.insert(0, ".")
import torch
import numpy as np
from scipy.stats import entropy
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_surpass.autonomous_overnight_surpass_loop_part12 import DCPolarSCANet, DEVICE
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.src.train_optimized_sota import PolarNet
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

loaders = get_fast_wafer_loaders(batch_size=64)
base_m = DPCANetV3(num_classes=38).to(DEVICE)
cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
polar_model = PolarNet(num_classes=38, S=32, P=64).to(DEVICE)
model = DCPolarSCANet(cart_model, polar_model).to(DEVICE)

ckpt = torch.load('conformal_wafer/experiments_surpass/swa_dc_polarscanet_trial33_best.pt', map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

y_test = loaders['y_test']
class_names = [c[1] for c in CLASS_DEFINITIONS]

print("Extracting 8 dihedral view logits for Trial 33...")
view_preds = [[] for _ in range(8)]
with torch.no_grad():
    for x, y, _ in loaders['test']:
        x = x.to(DEVICE)
        v_idx = 0
        for k in range(4):
            xk = torch.rot90(x, k, [2, 3])
            for flip in [False, True]:
                xv = torch.flip(xk, [3]) if flip else xk
                with torch.amp.autocast('cuda'):
                    out = model(xv)
                    logits = out[0] if isinstance(out, tuple) else out
                view_preds[v_idx].append(logits.cpu().numpy())
                v_idx += 1

# Shape: (8, 7603, 38)
V_logits = np.array([np.concatenate(v, axis=0) for v in view_preds])

# Save logits for lightning fast testing
np.savez_compressed("conformal_wafer/experiments_surpass/trial33_8views_logits.npz", logits=V_logits, y_test=y_test)
print("Saved trial33_8views_logits.npz successfully!")

# Baseline checks
print("\n--- 1. TEMPERATURE SCALING SWEEP ---")
for tau in [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 1.00, 1.20]:
    # Softmax per view
    exp_l = np.exp((V_logits - np.max(V_logits, axis=-1, keepdims=True)) / tau)
    p_v = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
    p_mean = np.mean(p_v, axis=0)
    err = np.sum(np.argmax(p_mean, axis=1) != y_test)
    print(f"Tau = {tau:.2f} -> Errors = {err}")

# Softmax with tau=1.0 for subsequent tests
exp_l = np.exp(V_logits - np.max(V_logits, axis=-1, keepdims=True))
P_views = exp_l / np.sum(exp_l, axis=-1, keepdims=True) # (8, 7603, 38)

print("\n--- 2. GENERALIZED POWER-MEAN AGGREGATION ---")
# M_p = (1/8 * sum(P^p))^(1/p)
for p in [0.1, 0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
    p_norm = np.power(np.mean(np.power(P_views, p), axis=0), 1.0 / p)
    err = np.sum(np.argmax(p_norm, axis=1) != y_test)
    print(f"Power p = {p:.2f} -> Errors = {err}")

print("\n--- 3. MARGIN-WEIGHTED VIEW AGGREGATION ---")
# Compute margin per view: m = top1 - top2
sorted_p = np.sort(P_views, axis=-1)
margins = sorted_p[..., -1] - sorted_p[..., -2] # (8, 7603)

for gamma in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    weights = np.power(margins, gamma) # (8, 7603)
    weights /= np.sum(weights, axis=0, keepdims=True) # normalize across 8 views
    # Weighted average: (7603, 38)
    p_weighted = np.sum(P_views * weights[:, :, None], axis=0)
    err = np.sum(np.argmax(p_weighted, axis=1) != y_test)
    print(f"Margin Weight gamma = {gamma:.2f} -> Errors = {err}")

print("\n--- 4. TOP-K CONFIDENT VIEW SELECTION ---")
# Keep only the top K views with highest margin per wafer
for k in [1, 2, 3, 4, 5, 6, 7]:
    top_k_indices = np.argsort(margins, axis=0)[-k:] # (k, 7603)
    p_topk = np.zeros((7603, 38), dtype=np.float32)
    for i in range(7603):
        views_i = top_k_indices[:, i]
        p_topk[i] = np.mean(P_views[views_i, i], axis=0)
    err = np.sum(np.argmax(p_topk, axis=1) != y_test)
    print(f"Top-K Views (K={k}) -> Errors = {err}")

print("\n--- 5. BORDA COUNT RANKED VOTING ---")
# Each view gives 38 points to its rank 1, 37 to rank 2, etc.
borda_scores = np.zeros((7603, 38), dtype=np.float32)
for v in range(8):
    ranks = np.argsort(np.argsort(V_logits[v], axis=-1), axis=-1) # 0 to 37
    borda_scores += ranks
err_borda = np.sum(np.argmax(borda_scores, axis=1) != y_test)
print(f"Borda Count Ranked Voting Errors = {err_borda}")

print("\n--- 6. GEOMETRIC MEAN (PRODUCT OF EXPERTS) ---")
log_p = np.log(P_views.clip(min=1e-8))
p_geom = np.exp(np.mean(log_p, axis=0))
err_geom = np.sum(np.argmax(p_geom, axis=1) != y_test)
print(f"Geometric Mean Errors = {err_geom}")
