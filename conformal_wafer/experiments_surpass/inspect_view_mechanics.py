"""
inspect_view_mechanics.py
Deep forensic analysis of what the winning views saw on the 18 gap wafers.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

d2 = np.load('conformal_wafer/experiments_surpass/trial02_8views_logits.npz')
test_views_cart = d2['test_views'] # (7603, 8, 38)
y_test = d2['y_test']

p_8 = F.softmax(torch.tensor(test_views_cart) / 0.50, dim=-1).numpy()
p_mean = p_8.mean(axis=1)
preds_mean = np.argmax(p_mean, axis=-1)

# Find the 18 gap wafers
top_views = np.argmax(test_views_cart, axis=-1)
gap_indices = [i for i in range(len(y_test)) if preds_mean[i] != y_test[i] and y_test[i] in top_views[i]]

print(f"Total Gap Wafers: {len(gap_indices)}")
print("=" * 85)

for idx in gap_indices:
    true_label = y_test[idx]
    pred_label = preds_mean[idx]
    t_name = CLASS_DEFINITIONS[true_label][1]
    p_name = CLASS_DEFINITIONS[pred_label][1]
    
    # 8 view predictions and probabilities
    v_preds = top_views[idx]
    corr_views = np.where(v_preds == true_label)[0]
    
    # For each of the 8 views, get the logit of the true class and the pred class
    l_true = test_views_cart[idx, :, true_label]
    l_pred = test_views_cart[idx, :, pred_label]
    
    # Max logit across all classes for each view
    max_logits = np.max(test_views_cart[idx], axis=-1)
    
    # Look at the difference: l_true - l_pred for each view
    diff = l_true - l_pred
    
    print(f"Wafer {idx:4d} | True: {t_name:12s} ({true_label:2d}) vs Pred: {p_name:12s} ({pred_label:2d}) | Correct Views: {corr_views.tolist()}")
    print(f"  True Class Logits across 8 views: {[round(float(x), 2) for x in l_true]}")
    print(f"  Pred Class Logits across 8 views: {[round(float(x), 2) for x in l_pred]}")
    print(f"  Logit Diff (True - Pred) 8 views: {[round(float(x), 2) for x in diff]}")
    print(f"  Max Logit in Correct View: {max_logits[corr_views[0]]:.2f} (True logit: {l_true[corr_views[0]]:.2f})")
