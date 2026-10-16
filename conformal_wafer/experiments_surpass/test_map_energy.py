import os
import sys
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

VECS = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.float32) # (38, 8)

# Load ensemble probabilities
m12 = np.load('conformal_wafer/experiments_sota/base_m1_m2_probs.npz')
v3 = np.load('conformal_wafer/experiments_sota/dpcanet_v3_probs.npz')

t_p1, t_p2, y_test = m12['t_p1'], m12['t_p2'], m12['y_test'].astype(int)
v_p1, v_p2, y_val = m12['v_p1'], m12['v_p2'], m12['y_val'].astype(int)
v_p3, t_p3 = v3['p_val'], v3['p_test']

# Tri-ensemble
p_tri_val = 0.222 * v_p1 + 0.111 * v_p2 + 0.667 * v_p3
p_tri_test = 0.222 * t_p1 + 0.111 * t_p2 + 0.667 * t_p3

# Also let's load our other best models from multi_model_tta_probs
multi = np.load("conformal_wafer/experiments_surpass/multi_model_tta_probs.npz")
extra = np.load("conformal_wafer/experiments_surpass/extra_models_tta_probs.npz")

# 5-Model Core Ensemble: v2 + v3 + HF + ASL + ResNet18
p_5_val = (multi['DPCANet_v2_val_d4'] + multi['DPCANet_v3_val_d4'] + multi['HFEdgeDPCANet_val_d4'] + extra['DPCANet_ASL_val_d4'] + extra['ResNet18_val_d4']) / 5.0
p_5_test = (multi['DPCANet_v2_test_d4'] + multi['DPCANet_v3_test_d4'] + multi['HFEdgeDPCANet_test_d4'] + extra['DPCANet_ASL_test_d4'] + extra['ResNet18_test_d4']) / 5.0

# Specialists
prim_data = np.load('conformal_wafer/experiments_sota/stable_primitive_probs.npz')
scratch_data = np.load('conformal_wafer/experiments_sota/scratch_specialist_probs.npz')

prim_val = prim_data['val_probs'].copy()
prim_test = prim_data['test_probs'].copy()
prim_val[:, 6] = scratch_data['val_probs']
prim_test[:, 6] = scratch_data['test_probs']

PRIMS = ['Center', 'Donut', 'Edge_Loc', 'Edge_Ring', 'Loc', 'Near_Full', 'Scratch', 'Random']

print("=" * 80)
print("TESTING MAP ENERGY FORMULATION WITH PER-PRIMITIVE RELIABILITY WEIGHTS")
print("=" * 80)

def compute_map_scores(p_base, p_prims, lambdas, eps=1e-5):
    # p_base: (N, 38)
    # p_prims: (N, 8)
    # lambdas: (8,)
    # VECS: (38, 8)
    log_base = np.log(np.clip(p_base, eps, 1.0))
    p_clip = np.clip(p_prims, eps, 1.0 - eps)

    # (N, 8)
    log_p = np.log(p_clip)
    log_1mp = np.log(1.0 - p_clip)

    # Weighted contribution: lambdas * log_p @ VECS.T + lambdas * log_1mp @ (1 - VECS).T
    w_log_p = log_p * lambdas
    w_log_1mp = log_1mp * lambdas

    # (N, 38)
    energy = log_base + w_log_p @ VECS.T + w_log_1mp @ (1.0 - VECS).T
    return energy

# 1. Baseline without MAP:
err_tri = int(np.sum(np.argmax(p_tri_test, axis=1) != y_test))
err_5 = int(np.sum(np.argmax(p_5_test, axis=1) != y_test))
print(f"Base Tri-Ensemble Test Errors: {err_tri} ({(1-err_tri/len(y_test))*100:.3f}%)")
print(f"Base 5-Model Ensemble Test Errors: {err_5} ({(1-err_5/len(y_test))*100:.3f}%)")

# 2. Optimization on Validation Split
# Initial lambdas: proportional to accuracy
init_lambdas = np.array([0.15, 0.15, 0.05, 0.10, 0.05, 0.15, 0.10, 0.15], dtype=np.float32)

def val_obj(lam):
    # L2 penalty to keep weights small and stable
    scores = compute_map_scores(p_5_val, prim_val, lam)
    preds = np.argmax(scores, axis=1)
    err = np.sum(preds != y_val)
    return err + 0.1 * np.sum(lam**2)

res = minimize(val_obj, init_lambdas, method='Nelder-Mead', options={'maxiter': 500})
opt_lambdas = np.clip(res.x, 0.0, 1.0)

print("\nOptimized Primitive Reliability Weights (from Validation set):")
for name, l in zip(PRIMS, opt_lambdas):
    print(f"  {name:12s}: {l:.4f}")

# Evaluate on Test set
scores_test = compute_map_scores(p_5_test, prim_test, opt_lambdas)
preds_test = np.argmax(scores_test, axis=1)
err_map = int(np.sum(preds_test != y_test))
print(f"\n--> MAP Energy Model on 5-Model Ensemble Test Set: {err_map} errors ({(1-err_map/len(y_test))*100:.3f}%)")

scores_tri_test = compute_map_scores(p_tri_test, prim_test, opt_lambdas)
err_map_tri = int(np.sum(np.argmax(scores_tri_test, axis=1) != y_test))
print(f"--> MAP Energy Model on Tri-Ensemble Test Set: {err_map_tri} errors ({(1-err_map_tri/len(y_test))*100:.3f}%)")

# Sweep scalar multipliers of lambdas
print("\nSensitivity Sweep over global lambda scale (alpha * opt_lambdas):")
for alpha in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    sc = compute_map_scores(p_tri_test, prim_test, alpha * opt_lambdas)
    e = int(np.sum(np.argmax(sc, axis=1) != y_test))
    print(f"  alpha={alpha:.2f} -> Test Errors: {e:2d} ({(1-e/len(y_test))*100:.3f}%)")
