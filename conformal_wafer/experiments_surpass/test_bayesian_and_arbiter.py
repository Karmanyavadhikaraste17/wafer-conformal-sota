import os
import sys
sys.path.insert(0, ".")
import numpy as np
from sklearn.metrics import accuracy_score

from conformal_wafer.src.dataset import CLASS_DEFINITIONS

# Load multi-model probabilities
multi = np.load("conformal_wafer/experiments_surpass/multi_model_tta_probs.npz")
y_val = multi['y_val']
y_test = multi['y_test']

# Load specialists
prim_data = np.load("conformal_wafer/experiments_sota/stable_primitive_probs.npz")
prim_val = prim_data['val_probs'].copy()
prim_test = prim_data['test_probs'].copy()

scratch_data = np.load("conformal_wafer/experiments_sota/scratch_specialist_probs.npz")
s_val = scratch_data['val_probs']
s_test = scratch_data['test_probs']

# Primitive 6 is Scratch
prim_val[:, 6] = s_val
prim_test[:, 6] = s_test

vecs = np.array([c[2] for c in CLASS_DEFINITIONS]) # (38, 8)

# Let's take the top ensemble: v2 + v3 + HFEdge + PolarNet
# Or optimal combination of probabilities
p_val_ens = (multi['DPCANet_v2_val_d4'] + multi['DPCANet_v3_val_d4'] + multi['HFEdgeDPCANet_val_d4']) / 3.0
p_test_ens = (multi['DPCANet_v2_test_d4'] + multi['DPCANet_v3_test_d4'] + multi['HFEdgeDPCANet_test_d4']) / 3.0

print("=" * 80)
print("BASE ENSEMBLE TEST ACCURACY:")
print("=" * 80)
err_base = int(np.sum(np.argmax(p_test_ens, axis=1) != y_test))
print(f"Base 3-Model Ensemble (v2+v3+HF): {err_base} errors ({(1-err_base/len(y_test))*100:.3f}%)")

# 1. Full Bayesian Log-Posterior Fusion:
# log P(C_i | x) = log P_ens(C_i | x) + \sum_j [ V_{ij} * log(p_j / prior_j) + (1 - V_{ij}) * log((1 - p_j) / (1 - prior_j)) ] * w_j
print("\n" + "=" * 80)
print("EXPERIMENT 1: BAYESIAN LOG-POSTERIOR FUSION")
print("=" * 80)

# Class prior frequencies
class_counts = np.bincount(y_val, minlength=38)
class_priors = (class_counts + 1.0) / (len(y_val) + 38)
prim_priors = np.clip(np.mean(vecs[y_val], axis=0), 1e-4, 1.0 - 1e-4)

def bayesian_fuse(p_base, p_prims, weights, eps=1e-6):
    log_p = np.log(np.clip(p_base, eps, 1.0))
    p_prims_c = np.clip(p_prims, eps, 1.0 - eps)

    # For each class i and primitive j:
    # If vecs[i, j] == 1, add weight[j] * (log p_j - log prior_j)
    # If vecs[i, j] == 0, add weight[j] * (log (1 - p_j) - log (1 - prior_j))
    log_odds_1 = np.log(p_prims_c) - np.log(prim_priors)
    log_odds_0 = np.log(1.0 - p_prims_c) - np.log(1.0 - prim_priors)

    # (N, 8) * (8,) -> (N, 8)
    # log_odds_1 @ vecs.T + log_odds_0 @ (1 - vecs).T
    # vecs is (38, 8)
    delta_log = (log_odds_1 * weights) @ vecs.T + (log_odds_0 * weights) @ (1 - vecs).T
    fused_log = log_p + delta_log
    return fused_log

# Grid search / optimize weights on validation set
from scipy.optimize import minimize

def obj_fn(w):
    log_fused = bayesian_fuse(p_val_ens, prim_val, w)
    preds = np.argmax(log_fused, axis=1)
    return int(np.sum(preds != y_val))

# Sweep uniform weight scalar first
best_w_val_err = int(np.sum(np.argmax(p_val_ens, axis=1) != y_val))
best_w_scalar = 0.0
for s in np.linspace(0.0, 0.5, 51):
    w = np.full(8, s)
    log_f = bayesian_fuse(p_val_ens, prim_val, w)
    err = int(np.sum(np.argmax(log_f, axis=1) != y_val))
    if err < best_w_val_err:
        best_w_val_err = err
        best_w_scalar = s

print(f"Validation scalar sweep: best scalar = {best_w_scalar:.3f} -> Val Errors: {best_w_val_err}")

# Evaluate on test
log_f_test = bayesian_fuse(p_test_ens, prim_test, np.full(8, best_w_scalar))
test_err_bayes = int(np.sum(np.argmax(log_f_test, axis=1) != y_test))
print(f"Bayesian Fusion Test Errors: {test_err_bayes} ({(1-test_err_bayes/len(y_test))*100:.3f}%)")

# 2. Bidirectional Arbiter (Twin Promotion + Twin Demotion)
print("\n" + "=" * 80)
print("EXPERIMENT 2: BIDIRECTIONAL ARBITER (PROMOTION + DEMOTION)")
print("=" * 80)

def run_bidirectional_arbiter(p_base, p_prims, min_ratio, tau_promo, tau_demo):
    preds = np.argmax(p_base, axis=1).copy()
    n = len(preds)
    for i in range(n):
        probs = p_base[i]
        top3 = np.argsort(probs)[::-1][:3]
        r1 = top3[0]
        p_r1 = probs[r1]

        # Check candidate promotions and demotions among top alternatives
        for r2 in top3[1:]:
            p_r2 = probs[r2]
            ratio = p_r2 / (p_r1 + 1e-12)
            if ratio < min_ratio:
                continue

            v1, v2 = vecs[r1], vecs[r2]
            # Promotion case: r2 has 1 extra primitive that r1 lacks
            if np.all(v2 >= v1) and np.sum(v2) == np.sum(v1) + 1:
                j_miss = np.where((v2 - v1) == 1)[0][0]
                if p_prims[i, j_miss] >= tau_promo:
                    preds[i] = r2
                    break
            # Demotion case: r1 has 1 extra primitive that r2 lacks
            elif np.all(v1 >= v2) and np.sum(v1) == np.sum(v2) + 1:
                j_extra = np.where((v1 - v2) == 1)[0][0]
                if p_prims[i, j_extra] <= tau_demo:
                    preds[i] = r2
                    break
    return preds

# Sweep on validation set
best_bi_val_err = int(np.sum(np.argmax(p_val_ens, axis=1) != y_val))
best_bi_params = None

for min_ratio in [0.05, 0.08, 0.10, 0.15, 0.20]:
    for tau_p in [0.10, 0.15, 0.20, 0.25, 0.30]:
        for tau_d in [0.01, 0.03, 0.05, 0.10, 0.15]:
            v_p = run_bidirectional_arbiter(p_val_ens, prim_val, min_ratio, tau_p, tau_d)
            e = int(np.sum(v_p != y_val))
            if e < best_bi_val_err:
                best_bi_val_err = e
                best_bi_params = (min_ratio, tau_p, tau_d)

print(f"Bidirectional Arbiter Val Best: {best_bi_params} -> Val Errors: {best_bi_val_err}")
if best_bi_params is not None:
    t_preds = run_bidirectional_arbiter(p_test_ens, prim_test, *best_bi_params)
    t_err = int(np.sum(t_preds != y_test))
    print(f"Bidirectional Arbiter Test Errors: {t_err} ({(1-t_err/len(y_test))*100:.3f}%)")

# Test oracle of arbiter parameters on test set to see potential ceiling
min_test_err_arb = err_base
best_test_arb = None
for min_ratio in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
    for tau_p in [0.05, 0.08, 0.10, 0.15, 0.20, 0.25]:
        for tau_d in [0.01, 0.03, 0.05, 0.10, 0.15]:
            tp = run_bidirectional_arbiter(p_test_ens, prim_test, min_ratio, tau_p, tau_d)
            e = int(np.sum(tp != y_test))
            if e < min_test_err_arb:
                min_test_err_arb = e
                best_test_arb = (min_ratio, tau_p, tau_d)

print(f"Bidirectional Arbiter Test Ceiling: {best_test_arb} -> {min_test_err_arb} errors ({(1-min_test_err_arb/len(y_test))*100:.3f}%)")
