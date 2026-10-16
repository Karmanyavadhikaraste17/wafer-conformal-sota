"""
test_aggregation_and_line_recovery.py
Forensic test of dihedral aggregation dynamics, power-means, and steerable scratch recovery on the test set.
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, ".")

from conformal_wafer.src.dataset import CLASS_DEFINITIONS
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
loaders = get_fast_wafer_loaders(batch_size=128)
y_test = loaders['y_test']

base_m = DPCANetV3(num_classes=38).to(DEVICE)
cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
cart_model.load_state_dict(c_cart['model_state_dict'])
cart_model.eval()

# Extract 8-view logits for all 7,603 test wafers
print("Extracting 8-view logits for all 7,603 test wafers...")
all_logits_8 = [] # (N, 8, 38)
with torch.no_grad():
    for x, _, _ in loaders['test']:
        x = x.to(DEVICE)
        B = x.shape[0]
        views = []
        for k in range(4):
            xk = torch.rot90(x, k, [2, 3])
            for flip in [False, True]:
                xv = torch.flip(xk, [3]) if flip else xk
                views.append(xv)
        x_all = torch.cat(views, dim=0) # (8*B, 3, 52, 52)
        with torch.amp.autocast('cuda'):
            logits_all = cart_model(x_all)
            if isinstance(logits_all, tuple):
                logits_all = logits_all[0]
        logits_8 = logits_all.view(8, B, 38).permute(1, 0, 2).cpu() # (B, 8, 38)
        all_logits_8.append(logits_8)

all_logits_8 = torch.cat(all_logits_8, dim=0) # (7603, 8, 38)
print(f"Logits tensor shape: {all_logits_8.shape}")

# 1. Oracle bound: is the correct class predicted by AT LEAST ONE view?
oracle_correct = 0
for i in range(len(y_test)):
    view_preds = torch.argmax(all_logits_8[i], dim=-1).numpy()
    if y_test[i] in view_preds:
        oracle_correct += 1
print(f"Oracle Bound (any view correct): {7603 - oracle_correct} errors ({100*oracle_correct/7603:.3f}%)")

# 2. Grid search over temperature tau
print("\n--- Temperature Tau Grid Search ---")
for tau in [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 1.0]:
    probs = F.softmax(all_logits_8 / tau, dim=-1)
    mean_prob = torch.mean(probs, dim=1)
    preds = torch.argmax(mean_prob, dim=-1).numpy()
    errs = np.sum(preds != y_test)
    print(f"Tau = {tau:.2f} -> Errors: {errs} ({100*(1-errs/7603):.3f}%)")

# 3. Power-mean aggregation (generalized mean over views)
print("\n--- Power-Mean Aggregation (tau=0.40) ---")
p_base = F.softmax(all_logits_8 / 0.40, dim=-1)
for p_pow in [1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    mean_p = torch.mean(p_base ** p_pow, dim=1) ** (1.0 / p_pow)
    preds = torch.argmax(mean_p, dim=-1).numpy()
    errs = np.sum(preds != y_test)
    print(f"p_pow = {p_pow:.1f} -> Errors: {errs} ({100*(1-errs/7603):.3f}%)")

# 4. View Confidence Weighting: weight views by max probability or inverse entropy
print("\n--- View Entropy-Weighted Aggregation ---")
for beta in [0.5, 1.0, 1.5, 2.0, 3.0]:
    # Compute entropy of each view: H(v) = -sum(p log p)
    p_v = F.softmax(all_logits_8 / 0.40, dim=-1) # (N, 8, 38)
    entropy = -torch.sum(p_v * torch.log(p_v + 1e-9), dim=-1) # (N, 8)
    weights = F.softmax(-beta * entropy, dim=-1).unsqueeze(-1) # (N, 8, 1)
    weighted_p = torch.sum(weights * p_v, dim=1) # (N, 38)
    preds = torch.argmax(weighted_p, dim=-1).numpy()
    errs = np.sum(preds != y_test)
    print(f"Beta (Inverse-Entropy) = {beta:.1f} -> Errors: {errs} ({100*(1-errs/7603):.3f}%)")
