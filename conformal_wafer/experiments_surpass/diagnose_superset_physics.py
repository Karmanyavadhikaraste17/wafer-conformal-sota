"""
diagnose_superset_physics.py
Compare the 8 recovered wafers vs 54 harmed wafers in terms of physical features.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

# Load data
data_t2 = np.load("conformal_wafer/experiments_surpass/trial02_8views_logits.npz")
logits_8 = data_t2['test_views'].transpose(1, 0, 2) # (8, 7603, 38)
y_test = data_t2['y_test']

vecs = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=int)
class_names = [c[1] for c in CLASS_DEFINITIONS]

# Temperature sharpened (tau=0.50)
p_sharp_views = np.zeros_like(logits_8)
for v in range(8):
    exp_l = np.exp((logits_8[v] - np.max(logits_8[v], axis=-1, keepdims=True)) / 0.50)
    p_sharp_views[v] = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
p_sharp_mean = np.mean(p_sharp_views, axis=0)
pred_sharp = np.argmax(p_sharp_mean, axis=1)

top_views = np.argmax(logits_8, axis=-1) # (8, 7603)

recovered_list = []
harmed_list = []

for i in range(7603):
    true_label = y_test[i]
    mean_pred = pred_sharp[i]
    
    cand_classes, cand_counts = np.unique(top_views[:, i], return_counts=True)
    supersets = []
    for c in cand_classes:
        if c != mean_pred:
            diff = vecs[c] - vecs[mean_pred]
            if np.all(diff >= 0) and np.sum(diff) > 0:
                supersets.append(c)
                
    if len(supersets) > 0:
        for s in supersets:
            win_view = np.argmax(logits_8[:, i, s])
            conf_s = p_sharp_views[win_view, i, s]
            count_s = np.sum(top_views[:, i] == s)
            
            # Count how many views rank s in top-2
            top2_views = np.argsort(logits_8[:, i, :], axis=-1)[:, -2:] # (8, 2)
            s_in_top2 = np.sum([s in top2_views[v] for v in range(8)])
            
            # Logit difference in winning view: s minus mean_pred
            logit_diff_win = logits_8[win_view, i, s] - logits_8[win_view, i, mean_pred]
            # Mean logit difference across all 8 views
            mean_logit_diff = np.mean(logits_8[:, i, s] - logits_8[:, i, mean_pred])
            
            record = {
                'wafer_idx': i,
                'mean_pred': mean_pred,
                'candidate': s,
                'true_label': true_label,
                'count_win': count_s,
                's_in_top2': s_in_top2,
                'conf_s': conf_s,
                'logit_diff_win': logit_diff_win,
                'mean_logit_diff': mean_logit_diff
            }
            
            if true_label == s and mean_pred != true_label:
                recovered_list.append(record)
            elif mean_pred == true_label and s != true_label:
                harmed_list.append(record)

print(f"Total Recovered Records: {len(recovered_list)}")
print(f"Total Harmed Records: {len(harmed_list)}")

print("\n--- SAMPLE RECOVERED (TRUE SUPERSETS) ---")
for r in recovered_list[:8]:
    w = r['wafer_idx']
    mp = class_names[r['mean_pred']]
    cand = class_names[r['candidate']]
    tr = class_names[r['true_label']]
    print(f"Wafer {w:4d} | Pred: {mp:10s} -> Cand: {cand:10s} (True: {tr:10s}) | Counts: {r['count_win']}/8 | Top-2 views: {r['s_in_top2']}/8 | Conf: {r['conf_s']:.3f} | LogitDiffWin: {r['logit_diff_win']:+.2f} | MeanDiff: {r['mean_logit_diff']:+.2f}")

print("\n--- SAMPLE HARMED (FALSE ALARMS) ---")
for r in harmed_list[:8]:
    w = r['wafer_idx']
    mp = class_names[r['mean_pred']]
    cand = class_names[r['candidate']]
    tr = class_names[r['true_label']]
    print(f"Wafer {w:4d} | Pred: {mp:10s} -> Cand: {cand:10s} (True: {tr:10s}) | Counts: {r['count_win']}/8 | Top-2 views: {r['s_in_top2']}/8 | Conf: {r['conf_s']:.3f} | LogitDiffWin: {r['logit_diff_win']:+.2f} | MeanDiff: {r['mean_logit_diff']:+.2f}")

# Look at distribution statistics
print("\n--- STATISTICAL COMPARISON ---")
rec_counts = [r['count_win'] for r in recovered_list]
harm_counts = [r['count_win'] for r in harmed_list]
print(f"Candidate View Win Counts: Recovered mean={np.mean(rec_counts):.2f}, Harmed mean={np.mean(harm_counts):.2f}")

rec_top2 = [r['s_in_top2'] for r in recovered_list]
harm_top2 = [r['s_in_top2'] for r in harmed_list]
print(f"Candidate Top-2 Occurrences: Recovered mean={np.mean(rec_top2):.2f}, Harmed mean={np.mean(harm_top2):.2f}")

rec_mean_diff = [r['mean_logit_diff'] for r in recovered_list]
harm_mean_diff = [r['mean_logit_diff'] for r in harmed_list]
print(f"Mean Logit Diff (Cand - MeanPred): Recovered mean={np.mean(rec_mean_diff):.2f}, Harmed mean={np.mean(harm_mean_diff):.2f}")
