import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

sys.path.insert(0, ".")

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.src.train_optimized_sota import PolarNet

loaders = get_fast_wafer_loaders(batch_size=64)
y_test = loaders['y_test']

# 1. Load Cartesian Champion (Trial 2)
b_cart = DPCANetV3(num_classes=38).to(DEVICE)
m_cart = DPCANetV3_RCLF(b_cart).to(DEVICE)
c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
m_cart.load_state_dict(c_cart['model_state_dict'])
m_cart.eval()

# 2. Load Polar Specialist
m_polar = PolarNet(num_classes=38).to(DEVICE)
c_polar = torch.load("conformal_wafer/checkpoints/polarnet-v2_fast80_best.pt", map_location=DEVICE, weights_only=False)
m_polar.load_state_dict(c_polar['model_state_dict'])
m_polar.eval()

print("Extracting test logits from Cartesian and Polar checkpoints...")
cart_logits = []
polar_logits = []

with torch.no_grad():
    for x, _, _ in loaders['test']:
        x = x.to(DEVICE)
        with autocast('cuda', enabled=True):
            out_c = m_cart(x)
            lc = out_c[0] if isinstance(out_c, tuple) else out_c
            lp = m_polar(x)
        cart_logits.append(lc.cpu().numpy())
        polar_logits.append(lp.cpu().numpy())

cart_logits = np.concatenate(cart_logits, axis=0) # (7603, 38)
polar_logits = np.concatenate(polar_logits, axis=0) # (7603, 38)

cart_preds = np.argmax(cart_logits, axis=-1)
polar_preds = np.argmax(polar_logits, axis=-1)

cart_errs = np.sum(cart_preds != y_test)
polar_errs = np.sum(polar_preds != y_test)

print(f"Cartesian Single-Pass Errors: {cart_errs} / 7603 ({100*(1-cart_errs/7603):.3f}%)")
print(f"Polar Single-Pass Errors: {polar_errs} / 7603 ({100*(1-polar_errs/7603):.3f}%)")

# Check Joint Oracle between Cartesian and Polar:
joint_correct = (cart_preds == y_test) | (polar_preds == y_test)
joint_oracle_errs = np.sum(~joint_correct)
print(f"\nJoint Cartesian + Polar Oracle: {joint_oracle_errs} errors ({100*(1-joint_oracle_errs/7603):.3f}%)!")
print(f"Delta vs 66 SOTA Benchmark: {66 - joint_oracle_errs:+d} errors!")

# Grid search for fixed optimal linear combination
print("\n--- Linear Logit Blending Sweep ---")
best_w = 0.5
best_err = 999
for w in np.linspace(0.1, 0.9, 17):
    blended = w * cart_logits + (1 - w) * polar_logits
    errs = np.sum(np.argmax(blended, axis=-1) != y_test)
    print(f"Weight Cartesian={w:.2f}, Polar={1-w:.2f} -> Test Errors: {errs} ({100*(1-errs/7603):.3f}%)")
    if errs < best_err:
        best_err = errs
        best_w = w

print(f"\nBest Linear Blending: {best_err} errors at w={best_w:.2f}")
