import sys
import numpy as np
from collections import Counter

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

dt2 = np.load("conformal_wafer/experiments_surpass/trial02_d4_probs.npz")
y_test = dt2['y_test']
t2_probs = dt2['test_probs']
t2_preds = np.argmax(t2_probs, axis=1)

NAMES = [c[1] for c in CLASS_DEFINITIONS]
VECS = np.array([c[2] for c in CLASS_DEFINITIONS])

err_idx = np.where(t2_preds != y_test)[0]
print(f"Total Trial 2 Errors: {len(err_idx)} / {len(y_test)}")

error_pairs = Counter()
for i in err_idx:
    true_name = NAMES[y_test[i]]
    pred_name = NAMES[t2_preds[i]]
    error_pairs[(true_name, pred_name)] += 1

print("\n" + "=" * 80)
print(f"ALL ERROR PAIRS FOR TRIAL 2 ({len(err_idx)} ERRORS):")
print("=" * 80)
for (t, p), count in error_pairs.most_common():
    print(f"  {t:25s} -> {p:25s} : {count:2d} wafers")
