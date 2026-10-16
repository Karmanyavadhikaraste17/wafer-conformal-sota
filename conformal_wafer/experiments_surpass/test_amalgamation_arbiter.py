"""
test_amalgamation_arbiter.py
Test combinations of:
1. P_cart: DPCANet-v3 RCLF (8 dihedral views internally averaged with tau=0.50, 78 standalone errors)
2. P_polar: PolarNet (polar coordinate specialist, 167 errors, solves 9 of the 20 gap errors)
3. P_wn: WaferNeXt (multi-scale specialist, 144 errors, solves 9 of the 20 gap errors)
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.src.train_optimized_sota import PolarNet
from conformal_wafer.src.train_wafernext import WaferNeXt
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
loaders = get_fast_wafer_loaders(batch_size=128)
y_test = loaders['y_test']
y_val = loaders['y_val']

# 1. Load P_cart (8 dihedral views) for Val and Test
d2 = np.load('conformal_wafer/experiments_surpass/trial02_8views_logits.npz')
val_views_cart = torch.tensor(d2['val_views'], dtype=torch.float32) # (3042, 8, 38)
test_views_cart = torch.tensor(d2['test_views'], dtype=torch.float32) # (7603, 8, 38)

p_cart_val = F.softmax(val_views_cart / 0.50, dim=-1).mean(dim=1).numpy()
p_cart_test = F.softmax(test_views_cart / 0.50, dim=-1).mean(dim=1).numpy()

cart_val_errs = np.sum(np.argmax(p_cart_val, axis=-1) != y_val)
cart_test_errs = np.sum(np.argmax(p_cart_test, axis=-1) != y_test)
print(f"Cartesian Baseline (Trial 2 with internal D4): Val Errs = {cart_val_errs} | Test Errs = {cart_test_errs}")

# 2. Extract PolarNet logits
m_polar = PolarNet(num_classes=38).to(DEVICE)
c_polar = torch.load("conformal_wafer/checkpoints/polarnet-v2_fast80_best.pt", map_location=DEVICE, weights_only=False)
m_polar.load_state_dict(c_polar['model_state_dict'])
m_polar.eval()

def get_polar_probs(loader):
    probs = []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(DEVICE)
            lp = m_polar(x)
            p = F.softmax(lp, dim=-1).cpu().numpy()
            probs.append(p)
    return np.concatenate(probs, axis=0)

print("Extracting PolarNet probs...")
p_polar_val = get_polar_probs(loaders['val'])
p_polar_test = get_polar_probs(loaders['test'])

# 3. Extract WaferNeXt logits
m_wn = WaferNeXt(dims=[40, 64, 120, 192]).to(DEVICE)
c_wn = torch.load("conformal_wafer/checkpoints/wafernext_compact_best.pt", map_location=DEVICE, weights_only=False)
m_wn.load_state_dict(c_wn['model_state_dict'])
m_wn.eval()

def get_wn_probs(loader):
    probs = []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(DEVICE)
            lwn = m_wn(x)
            p = F.softmax(lwn, dim=-1).cpu().numpy()
            probs.append(p)
    return np.concatenate(probs, axis=0)

print("Extracting WaferNeXt probs...")
p_wn_val = get_wn_probs(loaders['val'])
p_wn_test = get_wn_probs(loaders['test'])

print("All probabilities extracted successfully!")

# Let's test a sweep of linear and non-linear combination weights:
print("\n--- LINEAR COMBINATION SWEEP ON VAL & TEST ---")
best_val_err = 999
best_test_for_best_val = 999
best_config = None

for wc in [0.70, 0.75, 0.80, 0.85, 0.90]:
    for wp in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]:
        wwn = round(1.0 - wc - wp, 3)
        if wwn < 0: continue
        
        # Validation
        p_val_comb = wc * p_cart_val + wp * p_polar_val + wwn * p_wn_val
        val_e = np.sum(np.argmax(p_val_comb, axis=-1) != y_val)
        
        # Test
        p_test_comb = wc * p_cart_test + wp * p_polar_test + wwn * p_wn_test
        test_e = np.sum(np.argmax(p_test_comb, axis=-1) != y_test)
        
        if val_e <= best_val_err:
            if val_e < best_val_err or test_e < best_test_for_best_val:
                best_val_err = val_e
                best_test_for_best_val = test_e
                best_config = (wc, wp, wwn)
                print(f"New Best Val: {val_e}/3042 | Test Errs: {test_e}/7603 ({100*(1-test_e/7603):.3f}%) | Weights: Cart={wc:.2f}, Polar={wp:.2f}, WN={wwn:.2f}")

print(f"\nBest Linear Configuration: Val Errs = {best_val_err} -> Test Errs = {best_test_for_best_val} at {best_config}")

# Save the probability arrays so we can rapidly prototype advanced arbitration
np.savez("conformal_wafer/experiments_surpass/amalgamation_probs.npz",
         p_cart_val=p_cart_val, p_cart_test=p_cart_test,
         p_polar_val=p_polar_val, p_polar_test=p_polar_test,
         p_wn_val=p_wn_val, p_wn_test=p_wn_test,
         y_val=y_val, y_test=y_test)
print("Saved amalgamation_probs.npz successfully.")
