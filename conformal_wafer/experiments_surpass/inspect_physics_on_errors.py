"""
inspect_physics_on_errors.py
Inspect Steerable Line Bank and Polar Harmonic responses specifically on the 78 error wafers.
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
from conformal_wafer.experiments_surpass.test_dcsnet_mechanisms import SteerableLineFilterBank, PolarHarmonicModule

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
loaders = get_fast_wafer_loaders(batch_size=128)
y_test = loaders['y_test']

base_m = DPCANetV3(num_classes=38).to(DEVICE)
cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
cart_model.load_state_dict(c_cart['model_state_dict'])
cart_model.eval()

# Steerable line filter bank and Polar harmonic module
line_bank = SteerableLineFilterBank(kernel_size=9, num_angles=8).to(DEVICE)
polar_module = PolarHarmonicModule(S=32, P=64).to(DEVICE)

# Run full test set evaluation to identify the 78 error wafers
all_preds = []
all_test_x = []
with torch.no_grad():
    for x, _, _ in loaders['test']:
        all_test_x.append(x)
        x = x.to(DEVICE)
        B = x.shape[0]
        views = []
        for k in range(4):
            xk = torch.rot90(x, k, [2, 3])
            for flip in [False, True]:
                xv = torch.flip(xk, [3]) if flip else xk
                views.append(xv)
        x_all = torch.cat(views, dim=0)
        with torch.amp.autocast('cuda'):
            logits_all = cart_model(x_all)
            if isinstance(logits_all, tuple):
                logits_all = logits_all[0]
        l_8 = logits_all.view(8, B, 38)
        p_8 = F.softmax(l_8 / 0.50, dim=-1)
        p_mean = torch.mean(p_8, dim=0)
        all_preds.append(torch.argmax(p_mean, dim=-1).cpu().numpy())

all_preds = np.concatenate(all_preds)
all_test_x = torch.cat(all_test_x, dim=0)

err_mask = (all_preds != y_test)
err_indices = np.where(err_mask)[0]
print(f"Total Test Errors: {len(err_indices)} / 7603")

# Analyze Steerable Line responses on the errors
x_err = all_test_x[err_indices].to(DEVICE)
with torch.no_grad():
    feat_line, max_response = line_bank(x_err) # (N_err, 32, 52, 52), (N_err, 1, 52, 52)
    max_line_per_wafer, _ = torch.max(max_response.flatten(1), dim=-1) # (N_err,)
    
    _, p_polar = polar_module(x_err) # (N_err, 3, 32, 64)
    p_def = p_polar[:, 2] # (N_err, 32, 64)
    r_var = torch.var(p_def, dim=-1, unbiased=False) # (N_err, 32)
    max_ang_var = torch.max(r_var, dim=-1)[0] # (N_err,)

print("\n--- Physics Response on Top Error Classes ---")
for idx_pos, idx in enumerate(err_indices):
    t_cls = CLASS_DEFINITIONS[y_test[idx]][1]
    p_cls = CLASS_DEFINITIONS[all_preds[idx]][1]
    ml = max_line_per_wafer[idx_pos].item()
    mv = max_ang_var[idx_pos].item()
    
    # Only print compounds or interesting confusions
    if 'S' in t_cls or 'EL' in t_cls or 'ER' in t_cls:
        print(f"Wafer {idx:4d} | True: {t_cls:12s} -> Pred: {p_cls:12s} | Max Line Resp: {ml:.3f} | Max Ang Var: {mv:.4f}")
