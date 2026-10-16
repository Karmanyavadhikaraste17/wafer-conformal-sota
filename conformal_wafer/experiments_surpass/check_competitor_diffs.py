"""
check_competitor_diffs.py
Compare the competitor diff_win in error wafers vs correct wafers.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

d2 = np.load('conformal_wafer/experiments_surpass/trial02_8views_logits.npz')
test_views_cart = d2['test_views']
y_test = d2['y_test']

exp_v = np.exp((test_views_cart - np.max(test_views_cart, axis=-1, keepdims=True)) / 0.50)
p_8 = exp_v / np.sum(exp_v, axis=-1, keepdims=True)
p_mean = np.mean(p_8, axis=1)
preds_mean = np.argmax(p_mean, axis=-1)

top_views = np.argmax(test_views_cart, axis=-1)
unique_counts = np.array([len(np.unique(top_views[i])) for i in range(7603)])
disagreement_indices = np.where(unique_counts > 1)[0]

err_wafers = [i for i in disagreement_indices if preds_mean[i] != y_test[i]]
corr_wafers = [i for i in disagreement_indices if preds_mean[i] == y_test[i]]

class_names = [c[1] for c in CLASS_DEFINITIONS]

print("--- 104 CORRECT DISAGREEMENT WAFERS (FIRST 15) ---")
for i in corr_wafers[:15]:
    t_name = class_names[y_test[i]]
    m_name = class_names[preds_mean[i]]
    
    cands = [c for c in np.unique(top_views[i]) if c != preds_mean[i]]
    cand_info = []
    for c in cands:
        win_v = np.argmax(test_views_cart[i, :, c])
        p_win = p_8[i, win_v, c]
        diff_win = test_views_cart[i, win_v, c] - test_views_cart[i, win_v, preds_mean[i]]
        cand_info.append((class_names[c], f"v={win_v}", f"p={p_win:.3f}", f"diff={diff_win:+.2f}"))
        
    print(f"Wafer {i:4d} | True: {t_name:12s} | MeanPred: {m_name:12s} | Competing: {cand_info}")
