"""
test_dpinet_physics_prior.py

Evaluate Deterministic Physical Prior Injection on the 18 gap wafers:
1. Steerable line anisotropy at 52x52 to disentangle Scratch omissions and false positives.
2. Polar perimeter angular variance to disentangle Edge-Ring vs Edge-Local.
"""

import sys
import os
import math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from conformal_wafer.src.dataset import CLASS_DEFINITIONS
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_surpass.model_dcsnet import SteerableLineFilterBank, PolarHarmonicModule

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
loaders = get_fast_wafer_loaders(batch_size=128)

y_val = loaders['y_val']
y_test = loaders['y_test']

# Load precomputed 8-view logits
data = np.load('conformal_wafer/experiments_surpass/trial02_8views_logits.npz')
val_views = torch.tensor(data['val_views'], dtype=torch.float32) # (3042, 8, 38)
test_views = torch.tensor(data['test_views'], dtype=torch.float32) # (7603, 8, 38)

# Compute baseline predictions (tau=0.50)
tau = 0.50
p_val_base = F.softmax(val_views / tau, dim=-1).mean(dim=1)
preds_val_base = p_val_base.argmax(dim=-1).numpy()
val_base_errs = int((preds_val_base != y_val).sum())

p_test_base = F.softmax(test_views / tau, dim=-1).mean(dim=1)
preds_test_base = p_test_base.argmax(dim=-1).numpy()
test_base_errs = int((preds_test_base != y_test).sum())

print(f"Baseline Uniform (tau=0.50): Val Errs = {val_base_errs} / 3042 | Test Errs = {test_base_errs} / 7603")

# Extract physical metrics on Val and Test
line_bank = SteerableLineFilterBank(in_channels=3, kernel_size=9, num_angles=8).to(DEVICE)
polar_module = PolarHarmonicModule(S=32, P=64).to(DEVICE)

def compute_physical_metrics(loader):
    anisotropies = []
    peak_lines = []
    angular_vars = []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(DEVICE)
            # 1. Steerable line metrics
            p_def = x[:, 2:3]
            resp = F.conv2d(p_def, line_bank.line_kernels, padding=line_bank.kernel_size // 2) # (B, 8, 52, 52)
            max_per_orient, _ = torch.max(resp.flatten(2), dim=-1) # (B, 8)
            peak_val, _ = torch.max(max_per_orient, dim=-1) # (B,)
            med_val, _ = torch.median(max_per_orient, dim=-1) # (B,)
            aniso = peak_val - med_val
            
            # 2. Polar angular variance on perimeter
            _, p_grid = polar_module(x)
            p_grid_def = p_grid[:, 2] # (B, 32, 64)
            var_theta = torch.var(p_grid_def, dim=-1, unbiased=False) # (B, 32)
            # Perimeter bins: radius bins 25 to 30
            perim_var = var_theta[:, 25:31].mean(dim=-1) # (B,)
            
            anisotropies.append(aniso.cpu().numpy())
            peak_lines.append(peak_val.cpu().numpy())
            angular_vars.append(perim_var.cpu().numpy())
            
    return (np.concatenate(anisotropies),
            np.concatenate(peak_lines),
            np.concatenate(angular_vars))

print("Extracting physical metrics for Validation Set...")
val_aniso, val_peak, val_var = compute_physical_metrics(loaders['val'])

print("Extracting physical metrics for Test Set...")
test_aniso, test_peak, test_var = compute_physical_metrics(loaders['test'])

print("Physical metrics extracted successfully!")
print(f"Test Anisotropy: min={test_aniso.min():.3f}, mean={test_aniso.mean():.3f}, max={test_aniso.max():.3f}")
print(f"Test Perimeter Var: min={test_var.min():.4f}, mean={test_var.mean():.4f}, max={test_var.max():.4f}")

# Now let's inspect the 18 gap wafers specifically!
probs_8 = F.softmax(test_views / 0.50, dim=-1)
gap_indices = []
for i in range(len(y_test)):
    view_preds = torch.argmax(test_views[i], dim=-1).numpy()
    if preds_test_base[i] != y_test[i] and y_test[i] in view_preds:
        gap_indices.append(i)

print("\n" + "=" * 80)
print("18 GAP WAFERS PHYSICAL DIAGNOSTIC:")
print("=" * 80)
for idx in gap_indices:
    true_label = y_test[idx]
    pred_label = preds_test_base[idx]
    t_name = CLASS_DEFINITIONS[true_label][1]
    p_name = CLASS_DEFINITIONS[pred_label][1]
    
    aniso = test_aniso[idx]
    peak = test_peak[idx]
    var = test_var[idx]
    
    # Check which views got it right
    v_preds = torch.argmax(test_views[idx], dim=-1).numpy()
    correct_views = np.where(v_preds == true_label)[0]
    
    print(f"Wafer {idx:4d} | True: {t_name:12s} ({true_label:2d}) vs Pred: {p_name:12s} ({pred_label:2d}) | Aniso: {aniso:.3f} | Peak: {peak:.3f} | PerimVar: {var:.5f} | Correct Views: {correct_views}")

# Save metrics for experimentation
np.savez("conformal_wafer/experiments_surpass/physical_metrics.npz",
         val_aniso=val_aniso, val_peak=val_peak, val_var=val_var,
         test_aniso=test_aniso, test_peak=test_peak, test_var=test_var)
print("Saved physical_metrics.npz successfully.")
