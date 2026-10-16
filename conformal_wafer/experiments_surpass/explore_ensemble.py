import os
import sys
sys.path.insert(0, ".")
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

EXPERIMENTS_DIR = "conformal_wafer/experiments_sota"

# 1. Load data
m12 = np.load(os.path.join(EXPERIMENTS_DIR, "base_m1_m2_probs.npz"))
v_p1, v_p2, y_val = m12['v_p1'].astype(np.float32), m12['v_p2'].astype(np.float32), m12['y_val']
t_p1, t_p2, y_test = m12['t_p1'].astype(np.float32), m12['t_p2'].astype(np.float32), m12['y_test']

v3 = np.load(os.path.join(EXPERIMENTS_DIR, "dpcanet_v3_probs.npz"))
v_p3, t_p3 = v3['p_val'].astype(np.float32), v3['p_test'].astype(np.float32)

prim = np.load(os.path.join(EXPERIMENTS_DIR, "stable_primitive_probs.npz"))
prim_val, prim_test = prim['val_probs'].astype(np.float32).copy(), prim['test_probs'].astype(np.float32).copy()

scratch = np.load(os.path.join(EXPERIMENTS_DIR, "scratch_specialist_probs.npz"))
s_val, s_test = scratch['val_probs'].astype(np.float32), scratch['test_probs'].astype(np.float32)
prim_val[:, 6] = s_val
prim_test[:, 6] = s_test

asl = np.load(os.path.join(EXPERIMENTS_DIR, "asl_primitive_probs.npz"))
asl_val, asl_test = asl['val_probs'].astype(np.float32), asl['test_probs'].astype(np.float32)

from conformal_wafer.src.dataset import CLASS_DEFINITIONS
VECS = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.float32)

print(f"Val size: {len(y_val)}, Test size: {len(y_test)}")

# 2. Individual Model Accuracies
models = {
    'DPCANet-v2': (v_p1, t_p1),
    'HFEdgeDPCANet': (v_p2, t_p2),
    'DPCANet-v3': (v_p3, t_p3),
}

for name, (vp, tp) in models.items():
    acc_v = np.mean(np.argmax(vp, axis=1) == y_val) * 100
    acc_t = np.mean(np.argmax(tp, axis=1) == y_test) * 100
    err_t = np.sum(np.argmax(tp, axis=1) != y_test)
    print(f"  {name:15s}: Val Acc = {acc_v:.2f}%, Test Acc = {acc_t:.3f}% ({err_t} errors)")

# 3. Oracle Upper Bound
preds_t = np.stack([np.argmax(tp, axis=1) for _, (_, tp) in models.items()], axis=1)
oracle_correct = np.any(preds_t == y_test[:, None], axis=1)
print(f"\n[ORACLE BOUND (v2+HF+v3)]: Max possible correct = {np.sum(oracle_correct)} / {len(y_test)} ({np.mean(oracle_correct)*100:.3f}%), Min errors = {len(y_test) - np.sum(oracle_correct)}")

# 4. Grid Search / Optimization of Ensemble Weights on Validation Set
def loss_fn(weights):
    w = np.exp(weights) / np.sum(np.exp(weights))
    p_ens = w[0] * v_p1 + w[1] * v_p2 + w[2] * v_p3
    # cross entropy
    eps = 1e-7
    ce = -np.mean(np.log(p_ens[np.arange(len(y_val)), y_val] + eps))
    return ce

res = minimize(loss_fn, [0.0, 0.0, 0.0], method='Nelder-Mead')
opt_w = np.exp(res.x) / np.sum(np.exp(res.x))
print(f"\nOptimal Val Weights (CE): v2={opt_w[0]:.3f}, HF={opt_w[1]:.3f}, v3={opt_w[2]:.3f}")

p_ens_val = opt_w[0] * v_p1 + opt_w[1] * v_p2 + opt_w[2] * v_p3
p_ens_test = opt_w[0] * t_p1 + opt_w[1] * t_p2 + opt_w[2] * t_p3
acc_ens = np.mean(np.argmax(p_ens_test, axis=1) == y_test) * 100
err_ens = np.sum(np.argmax(p_ens_test, axis=1) != y_test)
print(f"Optimal Weighted Ensemble: Test Acc = {acc_ens:.3f}% ({err_ens} errors)")

# 5. Stacking / Meta-Classifier
print("\n--- Training Meta-Classifiers on Val Set ---")
X_val = np.concatenate([v_p1, v_p2, v_p3, prim_val, asl_val], axis=1)
X_test = np.concatenate([t_p1, t_p2, t_p3, prim_test, asl_test], axis=1)

for C_reg in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
    clf = LogisticRegression(C=C_reg, max_iter=500, random_state=42)
    clf.fit(X_val, y_val)
    acc_val_clf = clf.score(X_val, y_val) * 100
    acc_test_clf = clf.score(X_test, y_test) * 100
    err_test_clf = np.sum(clf.predict(X_test) != y_test)
    print(f"  LogisticRegression (C={C_reg:4f}): Val = {acc_val_clf:.2f}%, Test = {acc_test_clf:.3f}% ({err_test_clf} errors)")

for alpha_r in [0.1, 1.0, 10.0, 100.0]:
    ridge = RidgeClassifier(alpha=alpha_r, random_state=42)
    ridge.fit(X_val, y_val)
    acc_val_r = ridge.score(X_val, y_val) * 100
    acc_test_r = ridge.score(X_test, y_test) * 100
    err_test_r = np.sum(ridge.predict(X_test) != y_test)
    print(f"  RidgeClassifier (alpha={alpha_r:4f}): Val = {acc_val_r:.2f}%, Test = {acc_test_r:.3f}% ({err_test_r} errors)")
