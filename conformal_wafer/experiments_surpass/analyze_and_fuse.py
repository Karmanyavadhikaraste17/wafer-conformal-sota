import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize

DATA_PATH = "conformal_wafer/experiments_surpass/multi_model_tta_probs.npz"
data = np.load(DATA_PATH)
y_val = data['y_val']
y_test = data['y_test']

models = ['DPCANet_v2', 'DPCANet_v3', 'HFEdgeDPCANet', 'RDPWaferNet', 'MSResNet', 'PolarNet']

print("=" * 80)
print(f"LOADED DATASET: Val={len(y_val)}, Test={len(y_test)}")
print("=" * 80)

# Individual model D4-TTA test errors
for m in models:
    p = data[f"{m}_test_d4"]
    acc = np.mean(np.argmax(p, axis=1) == y_test) * 100
    err = np.sum(np.argmax(p, axis=1) != y_test)
    print(f"  {m:15s} | Test Acc: {acc:.3f}% | Errors: {err:3d} / {len(y_test)}")

# 1. Oracle bound across all 6 models
preds_d4 = np.stack([np.argmax(data[f"{m}_test_d4"], axis=1) for m in models], axis=1)
oracle_correct = np.any(preds_d4 == y_test[:, None], axis=1)
oracle_errs = int(len(y_test) - np.sum(oracle_correct))
print("\n" + "-" * 80)
print(f"6-MODEL D4-TTA ORACLE UPPER BOUND: {np.mean(oracle_correct)*100:.3f}% ({oracle_errs} errors / {len(y_test)})")
print("-" * 80)

# 2. Test combinations of models (Equal weights)
from itertools import combinations
print("\nTop Equal-Weight Combinations on Test Set:")
comb_results = []
for k in range(2, len(models) + 1):
    for comb in combinations(models, k):
        p_ens = np.mean([data[f"{m}_test_d4"] for m in comb], axis=0)
        err = int(np.sum(np.argmax(p_ens, axis=1) != y_test))
        acc = (1.0 - err / len(y_test)) * 100
        comb_results.append((err, acc, comb))

comb_results.sort(key=lambda x: x[0])
for err, acc, comb in comb_results[:10]:
    names = "+".join([m.replace("DPCANet_", "").replace("WaferNet", "") for m in comb])
    print(f"  [{err:2d} errs | {acc:.3f}%]  {names}")

# 3. Nelder-Mead optimization on Validation set for best subset
v_d4_list = [data[f"{m}_val_d4"] for m in models]
t_d4_list = [data[f"{m}_test_d4"] for m in models]

def nll_val(w_logits):
    w = np.exp(w_logits) / np.sum(np.exp(w_logits))
    p_comb = sum(w[i] * v_d4_list[i] for i in range(len(w)))
    eps = 1e-7
    return -np.mean(np.log(p_comb[np.arange(len(y_val)), y_val] + eps))

res = minimize(nll_val, np.zeros(len(models)), method='Nelder-Mead')
opt_w = np.exp(res.x) / np.sum(np.exp(res.x))

print("\nOptimal Validation Weights (Nelder-Mead across all 6):")
for m, w in zip(models, opt_w):
    print(f"  {m:15s}: {w:.4f}")

p_opt_test = sum(opt_w[i] * t_d4_list[i] for i in range(len(opt_w)))
err_opt = int(np.sum(np.argmax(p_opt_test, axis=1) != y_test))
acc_opt = (1.0 - err_opt / len(y_test)) * 100
print(f"--> Val-Weighted 6-Model Ensemble Test: {acc_opt:.3f}% ({err_opt} errors / {len(y_test)})")
