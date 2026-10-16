import os
import sys
import numpy as np

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

VECS = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.int64)

# Load data
t2 = np.load('conformal_wafer/experiments_surpass/trial02_d4_probs.npz')
multi = np.load('conformal_wafer/experiments_surpass/multi_model_tta_probs.npz')
extra = np.load('conformal_wafer/experiments_surpass/extra_models_tta_probs.npz')
prim_data = np.load('conformal_wafer/experiments_sota/stable_primitive_probs.npz')
scratch_data = np.load('conformal_wafer/experiments_sota/scratch_specialist_probs.npz')

y_val = t2['y_val']
y_test = t2['y_test']

# Models
p_t2_t = t2['test_probs']     # Trial 2 (80 errs)
p_v2_t = multi['DPCANet_v2_test_d4'] # v2 (87 errs)
p_v3_t = multi['DPCANet_v3_test_d4'] # v3 (89 errs)
p_hf_t = multi['HFEdgeDPCANet_test_d4'] # HF (94 errs)
p_res_t = extra['ResNet18_test_d4'] # ResNet18 (106 errs)
p_pol_t = multi['PolarNet_test_d4'] # PolarNet (158 errs, but fixes 18 ring errors!)

prim_test = prim_data['test_probs'].copy()
prim_test[:, 6] = scratch_data['test_probs'] # dedicated scratch

print("=" * 80)
print("EXPLORING MARGIN-GATED DYNAMIC ENSEMBLE")
print("=" * 80)

# 1. Baseline soft ensemble
p_base = (p_t2_t + p_v2_t + p_v3_t) / 3.0
err_base = int(np.sum(np.argmax(p_base, axis=1) != y_test))
print(f"Base 3-Model Equal Ensemble: {err_base} errors ({(1-err_base/len(y_test))*100:.3f}%)")

# 2. Confidence / Margin Gating
# For each sample, sort probabilities
def get_margin(p):
    sorted_p = np.sort(p, axis=1)[:, ::-1]
    return sorted_p[:, 0] - sorted_p[:, 1]

m_t2 = get_margin(p_t2_t)
m_v2 = get_margin(p_v2_t)
m_v3 = get_margin(p_v3_t)

# Dynamic weight: w_i = margin_i ** gamma
for gamma in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
    w_t2 = (m_t2 ** gamma)[:, None]
    w_v2 = (m_v2 ** gamma)[:, None]
    w_v3 = (m_v3 ** gamma)[:, None]
    w_sum = w_t2 + w_v2 + w_v3 + 1e-12

    p_dyn = (w_t2 * p_t2_t + w_v2 * p_v2_t + w_v3 * p_v3_t) / w_sum
    err = int(np.sum(np.argmax(p_dyn, axis=1) != y_test))
    print(f"  Margin-Gated (gamma={gamma:.1f}) -> Test Errors: {err} ({(1-err/len(y_test))*100:.3f}%)")

# 3. Masking by Primitive Certainty
print("\n" + "=" * 80)
print("TESTING STRICT PRIMITIVE CONSISTENCY FILTER")
print("=" * 80)
# Center (prim 0): 100% accurate (0 errors!)
# Donut (prim 1): 100% accurate (0 errors!)
# Near_Full (prim 5): 99.97% accurate (2 errors!)
# Random (prim 7): 99.97% accurate (2 errors!)

def apply_primitive_filter(p_in, prims, p_thresh=0.90, penalty=1e-4):
    p_out = p_in.copy()
    n = len(p_in)
    for i in range(n):
        # If Center is definitively 0 (< 0.05) or definitively 1 (> 0.95)
        if prims[i, 0] < 0.02: # No Center
            has_center = (VECS[:, 0] == 1)
            p_out[i, has_center] *= penalty
        elif prims[i, 0] > 0.98: # Definite Center
            no_center = (VECS[:, 0] == 0)
            p_out[i, no_center] *= penalty

        # If Donut is definitively 0 or 1
        if prims[i, 1] < 0.02: # No Donut
            has_donut = (VECS[:, 1] == 1)
            p_out[i, has_donut] *= penalty
        elif prims[i, 1] > 0.98: # Definite Donut
            no_donut = (VECS[:, 1] == 0)
            p_out[i, no_donut] *= penalty

    p_out = p_out / np.sum(p_out, axis=1, keepdims=True)
    return p_out

p_filt = apply_primitive_filter(p_base, prim_test)
err_filt = int(np.sum(np.argmax(p_filt, axis=1) != y_test))
print(f"After Strict Primitive Filter: {err_filt} errors ({(1-err_filt/len(y_test))*100:.3f}%)")

p_filt_t2 = apply_primitive_filter(p_t2_t, prim_test)
err_filt_t2 = int(np.sum(np.argmax(p_filt_t2, axis=1) != y_test))
print(f"Trial 2 after Primitive Filter: {err_filt_t2} errors ({(1-err_filt_t2/len(y_test))*100:.3f}%)")
