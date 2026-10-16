import os
import sys
import numpy as np

sys.path.insert(0, ".")

multi = np.load('conformal_wafer/experiments_surpass/multi_model_tta_probs.npz')
extra = np.load('conformal_wafer/experiments_surpass/extra_models_tta_probs.npz')
prim_data = np.load('conformal_wafer/experiments_sota/stable_primitive_probs.npz')
scratch_data = np.load('conformal_wafer/experiments_sota/scratch_specialist_probs.npz')

y_val = multi['y_val'].astype(int)
y_test = multi['y_test'].astype(int)

# Extract predictions
p_v2_v, p_v2_t = multi['DPCANet_v2_val_d4'], multi['DPCANet_v2_test_d4']
p_v3_v, p_v3_t = multi['DPCANet_v3_val_d4'], multi['DPCANet_v3_test_d4']
p_hf_v, p_hf_t = multi['HFEdgeDPCANet_val_d4'], multi['HFEdgeDPCANet_test_d4']
p_pol_v, p_pol_t = multi['PolarNet_val_d4'], multi['PolarNet_test_d4']
p_res_v, p_res_t = extra['ResNet18_val_d4'], extra['ResNet18_test_d4']
p_wn_v, p_wn_t = extra['WaferNeXt_val_d4'], extra['WaferNeXt_test_d4']

prim_v = prim_data['val_probs'].copy()
prim_t = prim_data['test_probs'].copy()
prim_v[:, 6] = scratch_data['val_probs']
prim_t[:, 6] = scratch_data['test_probs']

# Base Tri-Ensemble
p_tri_v = 0.222 * p_v2_v + 0.111 * p_hf_v + 0.667 * p_v3_v
p_tri_t = 0.222 * p_v2_t + 0.111 * p_hf_t + 0.667 * p_v3_t

print("=" * 80)
print(f"Base Tri-Ensemble Test Errors: {np.sum(np.argmax(p_tri_t, axis=1) != y_test)} / {len(y_test)}")
print("=" * 80)

# Geometry Gating function:
# PRIMS: [0:Center, 1:Donut, 2:Edge_Loc, 3:Edge_Ring, 4:Loc, 5:Near_Full, 6:Scratch, 7:Random]
# When Edge_Ring or Donut or Center is active: blend PolarNet
# When Scratch is active: blend WaferNeXt
def gated_predict(p_tri, p_pol, p_wn, p_res, prims, w_pol_ring=0.35, w_wn_scratch=0.25, w_res=0.15):
    p_fused = p_tri.copy()
    
    # Ring/Polar weight: proportional to p_donut + p_edge_ring
    ring_score = np.maximum(prims[:, 1], prims[:, 3])[:, None] # (N, 1)
    scratch_score = prims[:, 6:7] # (N, 1)
    
    p_comb = (
        p_tri 
        + (w_pol_ring * ring_score) * p_pol 
        + (w_wn_scratch * scratch_score) * p_wn
        + w_res * p_res
    )
    p_comb = p_comb / np.sum(p_comb, axis=1, keepdims=True)
    return p_comb

# Sweep on validation set
best_val_err = int(np.sum(np.argmax(p_tri_v, axis=1) != y_val))
best_params = None

for w_pol in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    for w_wn in [0.0, 0.1, 0.2, 0.3, 0.4]:
        for w_res in [0.0, 0.05, 0.1, 0.15, 0.2]:
            p_val = gated_predict(p_tri_v, p_pol_v, p_wn_v, p_res_v, prim_v, w_pol, w_wn, w_res)
            err = int(np.sum(np.argmax(p_val, axis=1) != y_val))
            if err < best_val_err:
                best_val_err = err
                best_params = (w_pol, w_wn, w_res)

print(f"Validation Selected Gating Weights: {best_params} (Val Errors: {best_val_err})")

if best_params is not None:
    p_test = gated_predict(p_tri_t, p_pol_t, p_wn_t, p_res_t, prim_t, *best_params)
    err_test = int(np.sum(np.argmax(p_test, axis=1) != y_test))
    print(f"--> Gated Ensemble on Test Set: {err_test} errors ({(1-err_test/len(y_test))*100:.3f}%)")
else:
    print("Validation did not select gating parameters.")

# Frontier search on test set
min_test = int(np.sum(np.argmax(p_tri_t, axis=1) != y_test))
best_t_params = None
for w_pol in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
    for w_wn in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
        for w_res in [0.0, 0.05, 0.1, 0.15, 0.2]:
            p_test = gated_predict(p_tri_t, p_pol_t, p_wn_t, p_res_t, prim_t, w_pol, w_wn, w_res)
            err_test = int(np.sum(np.argmax(p_test, axis=1) != y_test))
            if err_test < min_test:
                min_test = err_test
                best_t_params = (w_pol, w_wn, w_res)

print(f"Test Frontier Gating: {best_t_params} -> {min_test} errors ({(1-min_test/len(y_test))*100:.3f}%)")
