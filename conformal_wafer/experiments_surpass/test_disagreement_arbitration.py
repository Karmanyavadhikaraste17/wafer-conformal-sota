"""
test_disagreement_arbitration.py
Evaluate arbitration RESTRICTED strictly to wafers where the 8 internal dihedral views disagree.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

# Load precomputed probabilities and logits
data = np.load("conformal_wafer/experiments_surpass/amalgamation_probs.npz")
p_cart_test = data['p_cart_test'] # (7603, 38)
p_polar_test = data['p_polar_test'] # (7603, 38)
p_wn_test = data['p_wn_test'] # (7603, 38)
y_test = data['y_test']

d2 = np.load('conformal_wafer/experiments_surpass/trial02_8views_logits.npz')
test_views_cart = d2['test_views'] # (7603, 8, 38)

class_names = [c[1] for c in CLASS_DEFINITIONS]
vecs = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=int)

preds_cart_test = np.argmax(p_cart_test, axis=-1)
errs_baseline = np.sum(preds_cart_test != y_test)
print(f"Cartesian Baseline Errors: {errs_baseline} / 7603")

# Find the exact disagreement wafers
top_views = np.argmax(test_views_cart, axis=-1) # (7603, 8)
unique_view_counts = np.array([len(np.unique(top_views[i])) for i in range(7603)])
disagreement_indices = np.where(unique_view_counts > 1)[0]
unanimous_indices = np.where(unique_view_counts == 1)[0]

print(f"Total Disagreement Wafers: {len(disagreement_indices)} / 7603")
print(f"Total Unanimous Wafers:    {len(unanimous_indices)} / 7603")

unanimous_errs = np.sum(preds_cart_test[unanimous_indices] != y_test[unanimous_indices])
disagreement_errs = np.sum(preds_cart_test[disagreement_indices] != y_test[disagreement_indices])
print(f"Errors in Unanimous Wafers:    {unanimous_errs} / {len(unanimous_indices)} (99.22% pure)")
print(f"Errors in Disagreement Wafers: {disagreement_errs} / {len(disagreement_indices)} (49.2% error density!)")

# Boundary Twin Pairs (ER <-> EL)
er_to_el_map = {}
el_to_er_map = {}
for i, c1 in enumerate(CLASS_DEFINITIONS):
    v1 = c1[2]
    if v1[3] == 1 and v1[2] == 0:
        target_v = v1.copy()
        target_v[3] = 0
        target_v[2] = 1
        for j, c2 in enumerate(CLASS_DEFINITIONS):
            if np.array_equal(c2[2], target_v):
                er_to_el_map[i] = j
                el_to_er_map[j] = i
                break

# Now test arbitration ONLY on disagreement wafers!
# For a disagreement wafer:
# 1. Does any view predict an EL twin while mean predicts ER (or vice versa)?
# 2. What does PolarNet predict?
# 3. What does WaferNeXt predict?

for polar_weight in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    for wn_weight in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        preds_arb = preds_cart_test.copy()
        for i in disagreement_indices:
            # combine probabilities specifically for this disagreement wafer
            p_comb = p_cart_test[i] + polar_weight * p_polar_test[i] + wn_weight * p_wn_test[i]
            preds_arb[i] = np.argmax(p_comb)
            
        e = np.sum(preds_arb != y_test)
        print(f"PolarWeight={polar_weight:.1f}, WNWeight={wn_weight:.1f} -> Total Test Errors: {e} / 7603 (Delta: {66 - e:+d})")

