"""
investigate_18_gap_wafers.py
Deep dive into the 18 gap wafers.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

data = np.load('conformal_wafer/experiments_surpass/trial02_8views_logits.npz')
val_views = torch.tensor(data['val_views'], dtype=torch.float32) # (3042, 8, 38)
test_views = torch.tensor(data['test_views'], dtype=torch.float32) # (7603, 8, 38)
y_val = torch.tensor(data['y_val'], dtype=torch.long)
y_test = torch.tensor(data['y_test'], dtype=torch.long)

# At tau=0.50
p_test = F.softmax(test_views / 0.50, dim=-1).mean(dim=1)
preds_mean = p_test.argmax(dim=-1).numpy()

# 18 gap wafers
gap_indices = []
for i in range(len(y_test)):
    view_preds = torch.argmax(test_views[i], dim=-1).numpy()
    if preds_mean[i] != y_test[i] and y_test[i] in view_preds:
        gap_indices.append(i)

print(f"Total gap wafers: {len(gap_indices)}")

# Let's inspect each view's confidence, entropy, and logit distribution
# For each wafer, across all 8 views:
# Is there an unsupervised property of the winning view?
# (e.g. logit of the winning class, margin between top1 and top2, total entropy, or primitive consistency)
vecs = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=float)

print("\nDetailed breakdown of the 18 gap wafers:")
for idx in gap_indices:
    l_8 = test_views[idx] # (8, 38)
    p_8 = F.softmax(l_8 / 0.50, dim=-1) # (8, 38)
    
    true_label = y_test[idx].item()
    pred_label = preds_mean[idx].item()
    
    t_name = CLASS_DEFINITIONS[true_label][1]
    p_name = CLASS_DEFINITIONS[pred_label][1]
    
    v_preds = torch.argmax(l_8, dim=-1).numpy()
    correct_views = np.where(v_preds == true_label)[0]
    wrong_views = np.where(v_preds != true_label)[0]
    
    # Compare correct views vs wrong views
    corr_max_probs = [float(p_8[v].max()) for v in correct_views]
    wrong_max_probs = [float(p_8[v].max()) for v in wrong_views]
    
    corr_margins = [float(torch.topk(l_8[v], 2)[0][0] - torch.topk(l_8[v], 2)[0][1]) for v in correct_views]
    wrong_margins = [float(torch.topk(l_8[v], 2)[0][0] - torch.topk(l_8[v], 2)[0][1]) for v in wrong_views]
    
    print(f"\nWafer {idx:4d} | True: {t_name:12s} ({true_label:2d}) vs Pred: {p_name:12s} ({pred_label:2d}) | Correct Views: {correct_views.tolist()}")
    print(f"   Correct Views Max Prob: {[round(x, 3) for x in corr_max_probs]} | Margins: {[round(x, 2) for x in corr_margins]}")
    print(f"   Wrong   Views Max Prob: {[round(x, 3) for x in wrong_max_probs]} | Margins: {[round(x, 2) for x in wrong_margins]}")
