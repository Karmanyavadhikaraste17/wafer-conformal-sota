import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

NAMES = [c[1] for c in CLASS_DEFINITIONS]

data = np.load("conformal_wafer/experiments_surpass/trial02_8views_logits.npz")
val_views = data['val_views']   # (3042, 8, 38)
test_views = data['test_views'] # (7603, 8, 38)
y_val = data['y_val']
y_test = data['y_test']

# Compute sharpened probabilities
p_test_sharp = np.mean(F.softmax(torch.from_numpy(test_views) / 0.50, dim=-1).numpy(), axis=1) # (7603, 38)
preds_sharp = np.argmax(p_test_sharp, axis=1)
errs_sharp_mask = (preds_sharp != y_test)
print(f"Sharpened D4 TTA errors: {np.sum(errs_sharp_mask)}")

# 8 individual orientation predictions
p_views = F.softmax(torch.from_numpy(test_views) / 0.50, dim=-1).numpy() # (7603, 8, 38)
preds_views = np.argmax(p_views, axis=-1) # (7603, 8)

# Oracle
correct_any = np.any(preds_views == y_test[:, None], axis=1)
oracle_err_count = np.sum(~correct_any)
print(f"Oracle errors: {oracle_err_count}")

# Recoverable wafers
recoverable = np.where(errs_sharp_mask & correct_any)[0]
print(f"Recoverable wafers: {len(recoverable)}")

print("\n" + "=" * 80)
print("DETAILS OF THE RECOVERABLE WAFERS:")
print("=" * 80)
for idx in recoverable:
    y_t = y_test[idx]
    y_p = preds_sharp[idx]
    v_correct = np.where(preds_views[idx] == y_t)[0]
    
    # What was the confidence of the correct views vs incorrect views?
    conf_correct = [p_views[idx, v, y_t] for v in v_correct]
    max_conf_wrong = [np.max(p_views[idx, v]) for v in range(8) if v not in v_correct]
    
    print(f"Wafer {idx:4d} | True: {NAMES[y_t]:15s} | Pred: {NAMES[y_p]:15s} | Correct Views ({len(v_correct)}/8: {v_correct.tolist()}) | Conf Correct: {[f'{c:.2f}' for c in conf_correct]} | Max Wrong Conf: {[f'{c:.2f}' for c in max_conf_wrong][:3]}")
