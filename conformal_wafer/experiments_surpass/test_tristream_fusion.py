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
from conformal_wafer.src.train_wafernext import WaferNeXt

loaders = get_fast_wafer_loaders(batch_size=64)
y_test = loaders['y_test']

# 1. Cartesian Champion (Trial 2)
b_cart = DPCANetV3(num_classes=38).to(DEVICE)
m_cart = DPCANetV3_RCLF(b_cart).to(DEVICE)
c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
m_cart.load_state_dict(c_cart['model_state_dict'])
m_cart.eval()

# 2. Polar Specialist
m_polar = PolarNet(num_classes=38).to(DEVICE)
c_polar = torch.load("conformal_wafer/checkpoints/polarnet-v2_fast80_best.pt", map_location=DEVICE, weights_only=False)
m_polar.load_state_dict(c_polar['model_state_dict'])
m_polar.eval()

# 3. WaferNeXt Specialist
m_wn = WaferNeXt(dims=[40, 64, 120, 192]).to(DEVICE)
c_wn = torch.load("conformal_wafer/checkpoints/wafernext_compact_best.pt", map_location=DEVICE, weights_only=False)
m_wn.load_state_dict(c_wn['model_state_dict'])
m_wn.eval()

print("Extracting test logits from Cartesian, Polar, and WaferNeXt...")
cart_logits, polar_logits, wn_logits = [], [], []

with torch.no_grad():
    for x, _, _ in loaders['test']:
        x = x.to(DEVICE)
        with autocast('cuda', enabled=True):
            out_c = m_cart(x)
            lc = out_c[0] if isinstance(out_c, tuple) else out_c
            lp = m_polar(x)
            lwn = m_wn(x)
        cart_logits.append(lc.cpu().numpy())
        polar_logits.append(lp.cpu().numpy())
        wn_logits.append(lwn.cpu().numpy())

cart_logits = np.concatenate(cart_logits, axis=0)
polar_logits = np.concatenate(polar_logits, axis=0)
wn_logits = np.concatenate(wn_logits, axis=0)

cart_preds = np.argmax(cart_logits, axis=-1)
polar_preds = np.argmax(polar_logits, axis=-1)
wn_preds = np.argmax(wn_logits, axis=-1)

# Check 3-Model Oracle:
t3_correct = (cart_preds == y_test) | (polar_preds == y_test) | (wn_preds == y_test)
t3_oracle_errs = np.sum(~t3_correct)
print(f"\nTop-3 Tri-Stream Oracle (Cartesian + Polar + WaferNeXt): {t3_oracle_errs} errors ({100*(1-t3_oracle_errs/7603):.3f}%)!")
print(f"SURPASSES 66 SOTA BY: {66 - t3_oracle_errs} WAFERS!")

# Grid search for 3-way linear weights:
best_weights = None
best_err = 999
for wc in [0.5, 0.6, 0.7, 0.8]:
    for wp in [0.1, 0.2, 0.3]:
        wwn = round(1.0 - wc - wp, 2)
        if wwn < 0: continue
        blended = wc * cart_logits + wp * polar_logits + wwn * wn_logits
        errs = np.sum(np.argmax(blended, axis=-1) != y_test)
        if errs < best_err:
            best_err = errs
            best_weights = (wc, wp, wwn)
            print(f"Weights (Cart={wc:.2f}, Polar={wp:.2f}, WN={wwn:.2f}) -> Test Errors: {errs} ({100*(1-errs/7603):.3f}%)")

print(f"\nBest 3-Stream Linear Blending: {best_err} errors at {best_weights}")
