import numpy as np
import sys

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS
NAMES = [c[1] for c in CLASS_DEFINITIONS]

p2 = np.load("conformal_wafer/experiments_surpass/trial02_d4_probs.npz")['test_probs']
p18 = np.load("conformal_wafer/experiments_surpass/trial18_d4_probs.npz")['test_probs']
y_test = np.load("conformal_wafer/experiments_surpass/trial02_8views_logits.npz")['y_test']

pred2 = np.argmax(p2, axis=1)
pred18 = np.argmax(p18, axis=1)

err2 = (pred2 != y_test)
err18 = (pred18 != y_test)

print(f"Trial 2 Errors: {np.sum(err2)}")
print(f"Trial 18 Errors: {np.sum(err18)}")
print(f"Both Wrong: {np.sum(err2 & err18)}")
print(f"Trial 2 Right, Trial 18 Wrong: {np.sum(~err2 & err18)}")
print(f"Trial 18 Right, Trial 2 Wrong: {np.sum(err2 & ~err18)}")

# Look at the wafers Trial 18 got RIGHT that Trial 2 got WRONG:
t18_wins = np.where(err2 & ~err18)[0]
print("\n" + "=" * 70)
print(f"WAFERS WHERE CYLINDRIWAFER (T18) FIXED TRIAL 2 ERRORS (N={len(t18_wins)}):")
print("=" * 70)
for idx in t18_wins:
    print(f"Wafer {idx:4d} | True: {NAMES[y_test[idx]]:15s} | T2 Pred: {NAMES[pred2[idx]]:15s} | T18 Pred: {NAMES[pred18[idx]]:15s}")
