"""
test_compound_subsumption_theorem.py

Test the Compound Subsumption Theorem across the 8 dihedral views
for Trial 2 (dpcanet_v3_rclf_trial02_best.pt) and Trial 25 (pq_scanet_trial25_best.pt).
"""

import sys
sys.path.insert(0, ".")
import numpy as np
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

# Load Trial 2 8-views logits
data_t2 = np.load("conformal_wafer/experiments_surpass/trial02_8views_logits.npz")
logits_8 = data_t2['test_views'].transpose(1, 0, 2) # (8, 7603, 38)
y_test = data_t2['y_test']

vecs = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=int) # (38, 8)
class_names = [c[1] for c in CLASS_DEFINITIONS]

# Check standard D4-TTA
p_views = np.zeros_like(logits_8)
for v in range(8):
    exp_l = np.exp(logits_8[v] - np.max(logits_8[v], axis=-1, keepdims=True))
    p_views[v] = exp_l / np.sum(exp_l, axis=-1, keepdims=True)

mean_p = np.mean(p_views, axis=0)
pred_mean = np.argmax(mean_p, axis=1)
err_mean = np.sum(pred_mean != y_test)
print(f"Standard D4-TTA errors: {err_mean}")

# Temperature sharpened (tau=0.50)
p_sharp_views = np.zeros_like(logits_8)
for v in range(8):
    exp_l = np.exp((logits_8[v] - np.max(logits_8[v], axis=-1, keepdims=True)) / 0.50)
    p_sharp_views[v] = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
p_sharp_mean = np.mean(p_sharp_views, axis=0)
pred_sharp = np.argmax(p_sharp_mean, axis=1)
err_sharp = np.sum(pred_sharp != y_test)
print(f"Sharpened (tau=0.50) D4-TTA errors: {err_sharp}")

# For each wafer, get the top prediction for each of the 8 views
top_views = np.argmax(logits_8, axis=-1) # (8, 7603)

# Let's check how many wafers have diverse view predictions
unique_preds = [len(np.unique(top_views[:, i])) for i in range(7603)]
diverse_wafers = np.where(np.array(unique_preds) > 1)[0]
print(f"Wafers with differing predictions across the 8 views: {len(diverse_wafers)}")

# For each differing wafer, analyze:
# If the majority view predicts A, but a minority view predicts B:
# Is B a superset of A? (i.e. vecs[B] >= vecs[A] and sum(vecs[B]) > sum(vecs[A]))
recovered = 0
harmed = 0

for i in diverse_wafers:
    true_label = y_test[i]
    mean_pred = pred_sharp[i]
    
    # Check all candidate classes predicted by any view for wafer i
    cand_classes, cand_counts = np.unique(top_views[:, i], return_counts=True)
    
    # Find if any candidate is a superset of the mean_pred
    supersets = []
    for c in cand_classes:
        if c != mean_pred:
            # Check if c is a superset of mean_pred
            diff = vecs[c] - vecs[mean_pred]
            if np.all(diff >= 0) and np.sum(diff) > 0:
                supersets.append(c)
    
    if len(supersets) > 0:
        # If there's a superset candidate
        for s in supersets:
            # Check maximum confidence of this superset in its winning view
            win_view = np.argmax(logits_8[:, i, s])
            conf_s = p_sharp_views[win_view, i, s]
            
            if true_label == s and mean_pred != true_label:
                # print(f"Recoverable: {class_names[mean_pred]} -> {class_names[s]} (True: {class_names[true_label]}, Conf: {conf_s:.3f})")
                recovered += 1
            elif mean_pred == true_label and s != true_label:
                # print(f"False alarm: {class_names[mean_pred]} -> {class_names[s]} (True: {class_names[true_label]}, Conf: {conf_s:.3f})")
                harmed += 1

print(f"Potential recovery with superset promotion: Recovered={recovered}, Harmed={harmed}")
