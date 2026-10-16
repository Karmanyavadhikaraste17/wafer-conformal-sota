"""
compare_18_errors_vs_104_correct.py
Scientific comparison of the 18 error disagreement wafers vs the 104 correct disagreement wafers.
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

# Base probabilities (tau=0.50)
p_8 = F.softmax(torch.tensor(test_views_cart) / 0.50, dim=-1).numpy() # (7603, 8, 38)
p_mean = p_8.mean(axis=1) # (7603, 38)
preds_mean = np.argmax(p_mean, axis=-1)

# Disagreement wafers
top_views = np.argmax(test_views_cart, axis=-1) # (7603, 8)
unique_counts = np.array([len(np.unique(top_views[i])) for i in range(7603)])
disagreement_indices = np.where(unique_counts > 1)[0]

err_disagree = [i for i in disagreement_indices if preds_mean[i] != y_test[i]]
corr_disagree = [i for i in disagreement_indices if preds_mean[i] == y_test[i]]

print(f"Total Disagreement: {len(disagreement_indices)}")
print(f"  Error Disagreement:   {len(err_disagree)} (these are our 18 recoverable gap wafers!)")
print(f"  Correct Disagreement: {len(corr_disagree)}")

# What happens in the 18 error wafers?
# In the 18 error wafers, what are the top-1 and top-2 classes in p_mean?
vecs = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=int)
class_names = [c[1] for c in CLASS_DEFINITIONS]

print("\n--- 18 ERROR WAFERS: P_MEAN TOP-1 VS TOP-2 ---")
for idx in err_disagree:
    true_label = y_test[idx]
    pred_1 = preds_mean[idx]
    # find top-2 in p_mean
    sorted_classes = np.argsort(p_mean[idx])[::-1]
    pred_2 = sorted_classes[1]
    
    t_name = class_names[true_label]
    p1_name = class_names[pred_1]
    p2_name = class_names[pred_2]
    
    p1_val = p_mean[idx, pred_1]
    p2_val = p_mean[idx, pred_2]
    margin = p1_val - p2_val
    
    # Is the true class either pred_1 or pred_2?
    in_top2 = (true_label in [pred_1, pred_2])
    print(f"Wafer {idx:4d} | True: {t_name:12s} | Top-1: {p1_name:12s} ({p1_val:.3f}) | Top-2: {p2_name:12s} ({p2_val:.3f}) | Margin: {margin:.3f} | True in Top-2? {in_top2}")

# Check margins of the 104 correct disagreement wafers
margins_corr = [p_mean[i, np.argsort(p_mean[i])[-1]] - p_mean[i, np.argsort(p_mean[i])[-2]] for i in corr_disagree]
margins_err = [p_mean[i, np.argsort(p_mean[i])[-1]] - p_mean[i, np.argsort(p_mean[i])[-2]] for i in err_disagree]

print("\n--- MARGIN COMPARISON ---")
print(f"Correct Disagreement Margins: min={np.min(margins_corr):.3f}, median={np.median(margins_corr):.3f}, mean={np.mean(margins_corr):.3f}")
print(f"Error   Disagreement Margins: min={np.min(margins_err):.3f}, median={np.median(margins_err):.3f}, mean={np.mean(margins_err):.3f}")
