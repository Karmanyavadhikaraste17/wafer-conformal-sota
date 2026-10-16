import os
import sys
sys.path.insert(0, ".")
import numpy as np
from scipy.special import softmax
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

data = np.load("conformal_wafer/experiments_surpass/trial02_8views_logits.npz")
test_views = data['test_views'] # (7603, 8, 38)
y_test = data['y_test']

# Probabilities with tau=0.50
probs_per_view = softmax(test_views / 0.50, axis=-1) # (7603, 8, 38)
p_tta = np.mean(probs_per_view, axis=1) # (7603, 38)
y_pred_tta = np.argmax(p_tta, axis=1)

err_idx = np.where(y_pred_tta != y_test)[0]
print(f"Total TTA (tau=0.50) errors: {len(err_idx)} / 7603")

# Check which views got each wafer right
view_correct = np.zeros((len(test_views), 8), dtype=bool)
for v in range(8):
    y_pred_v = np.argmax(probs_per_view[:, v, :], axis=1)
    view_correct[:, v] = (y_pred_v == y_test)

oracle_correct = np.any(view_correct, axis=1)
recoverable_idx = [i for i in err_idx if oracle_correct[i]]
print(f"Recoverable wafers by at least one view: {len(recoverable_idx)} / {len(err_idx)}")
print(f"Hard unrecoverable errors: {len(err_idx) - len(recoverable_idx)} (Oracle bound: {len(test_views) - np.sum(oracle_correct)})")

print("\n" + "=" * 80)
print(f"{'Idx':<6} | {'True Label':<16} | {'TTA Pred':<16} | {'P(True)':<8} | {'P(Pred)':<8} | {'Num Views Correct':<18} | {'Best View P(True)'}")
print("=" * 80)

confusions = {}
for i in recoverable_idx:
    true_cls = CLASS_DEFINITIONS[y_test[i]][1]
    pred_cls = CLASS_DEFINITIONS[y_pred_tta[i]][1]
    p_true = p_tta[i, y_test[i]]
    p_pred = p_tta[i, y_pred_tta[i]]
    n_views = np.sum(view_correct[i])
    best_v = np.argmax(probs_per_view[i, :, y_test[i]])
    best_p_true = probs_per_view[i, best_v, y_test[i]]
    
    pair = f"{true_cls} -> {pred_cls}"
    confusions[pair] = confusions.get(pair, 0) + 1
    print(f"{i:<6} | {true_cls:<16} | {pred_cls:<16} | {p_true:<8.3f} | {p_pred:<8.3f} | {n_views:<18} | {best_p_true:.3f} (view {best_v})")

print("\nTop Confusion Pairs among Recoverable Samples:")
for pair, cnt in sorted(confusions.items(), key=lambda x: -x[1]):
    print(f"  {pair}: {cnt}")
