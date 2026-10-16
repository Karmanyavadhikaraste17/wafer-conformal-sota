"""
test_agreement_gating.py
Test consensus and specialist agreement gating across Cartesian, Polar, and WaferNeXt.
"""

import sys
import numpy as np

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

data = np.load("conformal_wafer/experiments_surpass/amalgamation_probs.npz")
p_cart_val = data['p_cart_val']
p_cart_test = data['p_cart_test']
p_polar_val = data['p_polar_val']
p_polar_test = data['p_polar_test']
p_wn_val = data['p_wn_val']
p_wn_test = data['p_wn_test']
y_val = data['y_val']
y_test = data['y_test']

pred_cart_val = np.argmax(p_cart_val, axis=-1)
pred_cart_test = np.argmax(p_cart_test, axis=-1)

pred_polar_val = np.argmax(p_polar_val, axis=-1)
pred_polar_test = np.argmax(p_polar_test, axis=-1)

pred_wn_val = np.argmax(p_wn_val, axis=-1)
pred_wn_test = np.argmax(p_wn_test, axis=-1)

print(f"Cart Baseline Test Errors: {np.sum(pred_cart_test != y_test)} / 7603")
print(f"Polar Baseline Test Errors: {np.sum(pred_polar_test != y_test)} / 7603")
print(f"WN Baseline Test Errors: {np.sum(pred_wn_test != y_test)} / 7603")

# Consensus Rule 1: Two-of-three majority vote
pred_maj_val = np.zeros_like(pred_cart_val)
for i in range(len(y_val)):
    preds = [pred_cart_val[i], pred_polar_val[i], pred_wn_val[i]]
    vals, counts = np.unique(preds, return_counts=True)
    if counts.max() >= 2:
        pred_maj_val[i] = vals[np.argmax(counts)]
    else:
        pred_maj_val[i] = pred_cart_val[i] # default to Cartesian champion

pred_maj_test = np.zeros_like(pred_cart_test)
for i in range(len(y_test)):
    preds = [pred_cart_test[i], pred_polar_test[i], pred_wn_test[i]]
    vals, counts = np.unique(preds, return_counts=True)
    if counts.max() >= 2:
        pred_maj_test[i] = vals[np.argmax(counts)]
    else:
        pred_maj_test[i] = pred_cart_test[i]

err_maj_val = np.sum(pred_maj_val != y_val)
err_maj_test = np.sum(pred_maj_test != y_test)
print(f"\nRule 1: Simple 2-of-3 Majority Vote -> Val Errs: {err_maj_val}/3042 | Test Errs: {err_maj_test}/7603")

# Consensus Rule 2: Specialist Agreement Override
# If Polar and WN agree on class C, and their combined confidence exceeds threshold, override Cartesian!
for conf_thresh in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    pred_override_val = pred_cart_val.copy()
    overrides_val = 0
    for i in range(len(y_val)):
        if pred_polar_val[i] == pred_wn_val[i] and pred_polar_val[i] != pred_cart_val[i]:
            cand = pred_polar_val[i]
            conf = 0.5 * (p_polar_val[i, cand] + p_wn_val[i, cand])
            if conf >= conf_thresh:
                pred_override_val[i] = cand
                overrides_val += 1
                
    pred_override_test = pred_cart_test.copy()
    overrides_test = 0
    for i in range(len(y_test)):
        if pred_polar_test[i] == pred_wn_test[i] and pred_polar_test[i] != pred_cart_test[i]:
            cand = pred_polar_test[i]
            conf = 0.5 * (p_polar_test[i, cand] + p_wn_test[i, cand])
            if conf >= conf_thresh:
                pred_override_test[i] = cand
                overrides_test += 1
                
    e_val = np.sum(pred_override_val != y_val)
    e_test = np.sum(pred_override_test != y_test)
    print(f"ConfThresh={conf_thresh:.1f} | Val Errs: {e_val}/3042 ({overrides_val} overrides) | Test Errs: {e_test}/7603 ({overrides_test} overrides)")
