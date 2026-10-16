import sys
import numpy as np

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

vecs = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.float64) # (38, 8)
vec_to_class = {tuple(v): i for i, v in enumerate(vecs)}

# Load Val & Test
dt2 = np.load("conformal_wafer/experiments_surpass/trial02_d4_probs.npz")
dm = np.load("conformal_wafer/experiments_surpass/multi_model_tta_probs.npz")
p = np.load("conformal_wafer/experiments_sota/stable_primitive_probs.npz")
scratch_data = np.load("conformal_wafer/experiments_sota/scratch_specialist_probs.npz")

y_val = dt2['y_val']
y_test = dt2['y_test']

# Probabilities in float64
p_t2_val, p_t2_test = dt2['val_probs'].astype(np.float64), dt2['test_probs'].astype(np.float64)
p_v2_val, p_v2_test = dm['DPCANet_v2_val_d4'].astype(np.float64), dm['DPCANet_v2_test_d4'].astype(np.float64)
p_pol_val, p_pol_test = dm['PolarNet_val_d4'].astype(np.float64), dm['PolarNet_test_d4'].astype(np.float64)

p_prim_val = p['val_probs'].astype(np.float64).copy()
p_prim_test = p['test_probs'].astype(np.float64).copy()
p_prim_val[:, 6] = scratch_data['val_probs'].astype(np.float64)
p_prim_test[:, 6] = scratch_data['test_probs'].astype(np.float64)

def compute_bayesian_compositional_logits(p_prims, eps=1e-4):
    p_clamped = np.clip(p_prims, eps, 1.0 - eps)
    log_pos = np.log(p_clamped)        # (N, 8)
    log_neg = np.log(1.0 - p_clamped)  # (N, 8)

    # For each class c (38):
    # log_P[c] = sum_j (VECS[c, j] * log_pos[j] + (1 - VECS[c, j]) * log_neg[j])
    comp_logits = np.matmul(log_pos, vecs.T) + np.matmul(log_neg, (1.0 - vecs).T) # (N, 38)
    return comp_logits

comp_val = compute_bayesian_compositional_logits(p_prim_val)
comp_test = compute_bayesian_compositional_logits(p_prim_test)

comp_test_err = int(np.sum(np.argmax(comp_test, axis=1) != y_test))
print(f"Bayesian Compositional Model Test Errors: {comp_test_err} / 7603 ({(1 - comp_test_err/len(y_test))*100:.3f}%)")

print("\n" + "=" * 80)
print("GRID SEARCH ON VALIDATION SPLIT (3,042 WAFERS)")
print("=" * 80)

best_val_err = 999
best_val_config = None

for w_v2 in [0.0, 0.10, 0.20, 0.30]:
    for w_pol in [0.0, 0.05, 0.10]:
        for w_comp in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]:
            log_fused_val = (
                np.log(np.clip(p_t2_val, 1e-6, 1.0))
                + w_v2 * np.log(np.clip(p_v2_val, 1e-6, 1.0))
                + w_pol * np.log(np.clip(p_pol_val, 1e-6, 1.0))
                + w_comp * comp_val
            )
            preds_v = np.argmax(log_fused_val, axis=1)
            e_v = int(np.sum(preds_v != y_val))

            if e_v < best_val_err:
                best_val_err = e_v
                best_val_config = (w_v2, w_pol, w_comp)

print(f"Optimal Validation Weights: w_v2={best_val_config[0]}, w_pol={best_val_config[1]}, w_comp={best_val_config[2]} -> Val Errs: {best_val_err} ({(1-best_val_err/len(y_val))*100:.3f}%)")

# Apply Optimal Validation Weights to Held-Out Test Set:
w_v2, w_pol, w_comp = best_val_config
log_fused_test = (
    np.log(np.clip(p_t2_test, 1e-6, 1.0))
    + w_v2 * np.log(np.clip(p_v2_test, 1e-6, 1.0))
    + w_pol * np.log(np.clip(p_pol_test, 1e-6, 1.0))
    + w_comp * comp_test
)
preds_t = np.argmax(log_fused_test, axis=1)
e_t = int(np.sum(preds_t != y_test))
acc_t = (1.0 - e_t / len(y_test)) * 100

print("\n" + "=" * 80)
print(f"VAL-CALIBRATED BAYESIAN COMPOSITIONAL FUSION ON TEST SET:")
print(f"  Test Errors: {e_t} / 7,603 ({acc_t:.3f}%)")
print(f"  Delta vs 66 SOTA Benchmark: {66 - e_t:+d} errors")
print("=" * 80)
