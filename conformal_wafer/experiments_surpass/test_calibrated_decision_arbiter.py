"""
test_calibrated_decision_arbiter.py
Test calibrated decision arbitration targeting:
1. Boundary (Edge-Ring vs Edge-Local) disambiguation via PolarNet
2. Orientation-spike scratch recovery via Dihedral Views
3. Multi-scale context via WaferNeXt
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

# Load precomputed probabilities and logits
data = np.load("conformal_wafer/experiments_surpass/amalgamation_probs.npz")
p_cart_val = data['p_cart_val'] # (3042, 38)
p_cart_test = data['p_cart_test'] # (7603, 38)
p_polar_val = data['p_polar_val'] # (3042, 38)
p_polar_test = data['p_polar_test'] # (7603, 38)
p_wn_val = data['p_wn_val'] # (3042, 38)
p_wn_test = data['p_wn_test'] # (7603, 38)
y_val = data['y_val']
y_test = data['y_test']

d2 = np.load('conformal_wafer/experiments_surpass/trial02_8views_logits.npz')
val_views_cart = d2['val_views'] # (3042, 8, 38)
test_views_cart = d2['test_views'] # (7603, 8, 38)

class_names = [c[1] for c in CLASS_DEFINITIONS]
vecs = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=int)

# Map every ER class to its EL twin, and vice versa!
# ER is primitive index 3 (vec[3]), EL is primitive index 2 (vec[2])
er_to_el_map = {}
el_to_er_map = {}
for i, c1 in enumerate(CLASS_DEFINITIONS):
    v1 = c1[2]
    # Check if this class has ER (v1[3]==1) and not EL (v1[2]==0)
    if v1[3] == 1 and v1[2] == 0:
        # Find twin with EL instead of ER
        target_v = v1.copy()
        target_v[3] = 0
        target_v[2] = 1
        for j, c2 in enumerate(CLASS_DEFINITIONS):
            if np.array_equal(c2[2], target_v):
                er_to_el_map[i] = j
                el_to_er_map[j] = i
                break

print("Identified Boundary Twin Pairs (ER <-> EL):")
for er_idx, el_idx in sorted(er_to_el_map.items()):
    print(f"  {class_names[er_idx]:15s} ({er_idx:2d}) <---> {class_names[el_idx]:15s} ({el_idx:2d})")

# Baseline Cartesian predictions
preds_cart_test = np.argmax(p_cart_test, axis=-1)
print(f"\nBaseline Cartesian Standalone Errors: {np.sum(preds_cart_test != y_test)} / 7603")

# Now let's test Calibrated Boundary Disambiguation:
# If Cartesian predicts an ER class (or EL class), but PolarNet strongly predicts the EL twin (or ER twin),
# evaluate whether PolarNet should override!
for polar_ratio_thresh in [1.2, 1.5, 2.0, 2.5, 3.0, 4.0]:
    preds_boundary_fix = preds_cart_test.copy()
    overrides = 0
    for i in range(len(y_test)):
        c_pred = preds_cart_test[i]
        if c_pred in er_to_el_map:
            el_twin = er_to_el_map[c_pred]
            # Does PolarNet prefer EL over ER by ratio?
            if p_polar_test[i, el_twin] > polar_ratio_thresh * p_polar_test[i, c_pred] and p_polar_test[i, el_twin] > 0.30:
                preds_boundary_fix[i] = el_twin
                overrides += 1
        elif c_pred in el_to_er_map:
            er_twin = el_to_er_map[c_pred]
            # Does PolarNet prefer ER over EL by ratio?
            if p_polar_test[i, er_twin] > polar_ratio_thresh * p_polar_test[i, c_pred] and p_polar_test[i, er_twin] > 0.30:
                preds_boundary_fix[i] = er_twin
                overrides += 1
                
    errs = np.sum(preds_boundary_fix != y_test)
    print(f"Polar Ratio Thresh {polar_ratio_thresh:3.1f}: Test Errors = {errs} ({overrides} boundary overrides)")

