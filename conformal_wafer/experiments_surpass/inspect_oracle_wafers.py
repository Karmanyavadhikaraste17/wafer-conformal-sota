import sys
sys.path.insert(0, ".")
import numpy as np
from scipy.special import softmax
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

raw = np.load("conformal_wafer/experiments_surpass/trial02_8views_logits.npz")
test_views = raw['test_views'] # (7603, 8, 38)
y_test = raw['y_test']

preds_8 = np.argmax(test_views, axis=-1) # (7603, 8)
correct_any = np.any(preds_8 == y_test[:, None], axis=1) # (7603,)

mean_p = np.mean(softmax(test_views / 0.5, axis=-1), axis=1)
preds_mean = np.argmax(mean_p, axis=1)
correct_mean = (preds_mean == y_test)

# Wafers where oracle is correct, but mean is WRONG!
oracle_saved_idx = np.where(correct_any & (~correct_mean))[0]
print(f"Number of wafers oracle gets right but mean gets wrong: {len(oracle_saved_idx)}")

print("\nDetailed breakdown of these wafers:")
for i in oracle_saved_idx[:18]:
    true_cls = CLASS_DEFINITIONS[y_test[i]][1]
    pred_cls = CLASS_DEFINITIONS[preds_mean[i]][1]
    
    # Check predictions of each view
    view_names = [CLASS_DEFINITIONS[p][1] for p in preds_8[i]]
    correct_views = [v for v, p in enumerate(preds_8[i]) if p == y_test[i]]
    print(f"Wafer {i:4d}: True = {true_cls:12s} | Mean Pred = {pred_cls:12s} | Correct Views: {correct_views}")
    print(f"  All 8 View Predictions: {view_names}")
    # View confidences
    confs = [softmax(test_views[i, v] / 0.5)[preds_8[i, v]] for v in range(8)]
    print(f"  View Confidences: {[round(c, 3) for c in confs]}")
    # Logit of true class vs predicted class across views
    true_logits = [round(test_views[i, v, y_test[i]], 2) for v in range(8)]
    pred_logits = [round(test_views[i, v, preds_mean[i]], 2) for v in range(8)]
    print(f"  True class logits: {true_logits}")
    print(f"  Pred class logits: {pred_logits}\n")
