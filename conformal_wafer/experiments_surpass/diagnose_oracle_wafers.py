import sys
sys.path.insert(0, ".")
import numpy as np
import torch
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

data = np.load('conformal_wafer/experiments_surpass/trial02_8views_logits.npz')
test_views = data['test_views'] # (7603, 8, 38)
y_test = data['y_test']

# Mean prediction
mean_logits = np.mean(test_views, axis=1)
pred_mean = np.argmax(mean_logits, axis=-1)
mean_err_indices = np.where(pred_mean != y_test)[0]

print(f"Total Mean Errors: {len(mean_err_indices)}")

# Check which of these 80 errors have at least one view predicting the true class
recoverable_indices = []
for idx in mean_err_indices:
    preds_across_views = np.argmax(test_views[idx], axis=-1) # (8,)
    if y_test[idx] in preds_across_views:
        recoverable_indices.append(idx)

print(f"Recoverable Wafers by Dihedral Oracle: {len(recoverable_indices)} / {len(mean_err_indices)}")

print("\n--- Diagnostic of Recoverable Wafers ---")
for idx in recoverable_indices[:10]:
    true_cls = CLASS_DEFINITIONS[y_test[idx]][1]
    mean_pred_cls = CLASS_DEFINITIONS[pred_mean[idx]][1]
    view_preds = [CLASS_DEFINITIONS[p][1] for p in np.argmax(test_views[idx], axis=-1)]
    # count how many views got it right
    correct_views = [v for v, p in enumerate(np.argmax(test_views[idx], axis=-1)) if p == y_test[idx]]
    print(f"Wafer {idx:4d} | True: {true_cls:15s} | Mean Pred: {mean_pred_cls:15s} | Correct Views: {correct_views}")
