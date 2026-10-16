"""
test_existential_asymmetry.py
Test the Existential Primitive Asymmetry Theorem on the 122 disagreement wafers.
"""

import sys
import numpy as np

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

d2 = np.load('conformal_wafer/experiments_surpass/trial02_8views_logits.npz')
test_views_cart = d2['test_views'] # (7603, 8, 38)
y_test = d2['y_test']

# Compute sharpened probabilities
exp_v = np.exp((test_views_cart - np.max(test_views_cart, axis=-1, keepdims=True)) / 0.50)
p_8 = exp_v / np.sum(exp_v, axis=-1, keepdims=True)
p_mean = np.mean(p_8, axis=1) # (7603, 38)
preds_mean = np.argmax(p_mean, axis=-1)

top_views = np.argmax(test_views_cart, axis=-1) # (7603, 8)
unique_counts = np.array([len(np.unique(top_views[i])) for i in range(7603)])
disagreement_indices = np.where(unique_counts > 1)[0]

vecs = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=int)
class_names = [c[1] for c in CLASS_DEFINITIONS]

# Primitives: 0: C, 1: D, 2: EL, 3: ER, 4: L, 5: NF, 6: R, 7: S
# Directional / localized primitives that can be hidden by rotation:
# EL (idx 2), L (idx 4), S (idx 6) -> wait, let's check exact primitive indices in CLASS_DEFINITIONS!

# Directional primitives
DIR_PRIMS = {2, 4, 6} # Edge_Loc, Loc, Scratch

# For each of the 122 disagreement wafers:
err_wafers = [i for i in disagreement_indices if preds_mean[i] != y_test[i]]
corr_wafers = [i for i in disagreement_indices if preds_mean[i] == y_test[i]]

print(f"\nAnalyzing 18 Error Wafers vs 104 Correct Wafers:")
# In each wafer, what candidate classes are predicted by any view?
# Look at the winning view of each candidate:
# What is the maximum probability of that candidate in its winning view?
# What is the margin of that candidate over the mean_pred in that view?

for i in err_wafers[:10]:
    t_name = class_names[y_test[i]]
    m_name = class_names[preds_mean[i]]
    
    # candidate classes
    cands = np.unique(top_views[i])
    cand_info = []
    for c in cands:
        win_v = np.argmax(test_views_cart[i, :, c])
        p_win = p_8[i, win_v, c]
        diff_win = test_views_cart[i, win_v, c] - test_views_cart[i, win_v, preds_mean[i]]
        cand_info.append((class_names[c], f"v={win_v}", f"p={p_win:.3f}", f"diff={diff_win:+.2f}"))
        
    print(f"Wafer {i:4d} | True: {t_name:12s} | MeanPred: {m_name:12s} | Cands: {cand_info}")
