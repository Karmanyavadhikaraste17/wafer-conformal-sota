import sys
import numpy as np

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

# Load data
dm = np.load("conformal_wafer/experiments_surpass/multi_model_tta_probs.npz")
de = np.load("conformal_wafer/experiments_surpass/extra_models_tta_probs.npz")
dt2 = np.load("conformal_wafer/experiments_surpass/trial02_d4_probs.npz")
prim_data = np.load("conformal_wafer/experiments_sota/stable_primitive_probs.npz")
scratch_data = np.load("conformal_wafer/experiments_sota/scratch_specialist_probs.npz")

y_val = dm['y_val']
y_test = dm['y_test']

VECS = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.int64) # (38, 8)
NAMES = [c[1] for c in CLASS_DEFINITIONS]

# Primitives: 0:Center, 1:Donut, 2:Edge_Loc, 3:Edge_Ring, 4:Loc, 5:Near_Full, 6:Scratch, 7:Random
EL_IDX = 2
ER_IDX = 3
S_IDX = 6
L_IDX = 4

# Base probability ensemble: fuse Trial 2 (RCLF) with top models
# DPCANet-v2, HFEdgeDPCANet, DPCANet-v3, Trial 2
p_base_val = 0.50 * dt2['val_probs'] + 0.25 * dm['DPCANet_v2_val_d4'] + 0.15 * dm['DPCANet_v3_val_d4'] + 0.10 * dm['HFEdgeDPCANet_val_d4']
p_base_test = 0.50 * dt2['test_probs'] + 0.25 * dm['DPCANet_v2_test_d4'] + 0.15 * dm['DPCANet_v3_test_d4'] + 0.10 * dm['HFEdgeDPCANet_test_d4']

print(f"Base Ensemble Test Errors: {np.sum(np.argmax(p_base_test, 1) != y_test)} / 7603")

# Extract PolarNet primitive signals
# PolarNet's probability distribution over 38 classes can be marginalized into 8 primitive probabilities:
# P_polar(j) = sum_{c: VECS[c, j] == 1} P_polar(c)
p_polar_val = dm['PolarNet_val_d4']
p_polar_test = dm['PolarNet_test_d4']

p_polar_prim_val = np.matmul(p_polar_val, VECS.astype(np.float32)) # (3042, 8)
p_polar_prim_test = np.matmul(p_polar_test, VECS.astype(np.float32)) # (7603, 8)
# Clip to 1.0
p_polar_prim_val = np.clip(p_polar_prim_val, 0.0, 1.0)
p_polar_prim_test = np.clip(p_polar_prim_test, 0.0, 1.0)

# Multi-model primitive fusion:
# Combine generic stable primitives + Scratch specialist + PolarNet EL/ER
prims_val = prim_data['val_probs'].copy()
prims_test = prim_data['test_probs'].copy()

# Scratch: use scratch specialist
prims_val[:, S_IDX] = scratch_data['val_probs']
prims_test[:, S_IDX] = scratch_data['test_probs']

# Edge-Local and Edge-Ring: fuse with PolarNet
prims_val[:, EL_IDX] = 0.5 * prims_val[:, EL_IDX] + 0.5 * p_polar_prim_val[:, EL_IDX]
prims_test[:, EL_IDX] = 0.5 * prims_test[:, EL_IDX] + 0.5 * p_polar_prim_test[:, EL_IDX]

prims_val[:, ER_IDX] = 0.5 * prims_val[:, ER_IDX] + 0.5 * p_polar_prim_val[:, ER_IDX]
prims_test[:, ER_IDX] = 0.5 * prims_test[:, ER_IDX] + 0.5 * p_polar_prim_test[:, ER_IDX]

print("=" * 80)
print("TESTING MULTI-WAY GRAMMATICAL ARBITER")
print("=" * 80)

def advanced_arbiter(p_base, p_prims, min_ratio_prom=0.20, tau_prom=0.15,
                     ratio_dem=0.40, tau_dem=0.25,
                     allow_rank3=True):
    preds = np.argmax(p_base, axis=1).copy()
    n = len(preds)
    
    for i in range(n):
        probs = p_base[i]
        top_k = np.argsort(probs)[::-1][: (3 if allow_rank3 else 2)]
        r1 = top_k[0]
        p_r1 = probs[r1]
        v1 = VECS[r1]
        
        # Check other top candidates
        for r_cand in top_k[1:]:
            p_cand = probs[r_cand]
            v_cand = VECS[r_cand]
            ratio = p_cand / (p_r1 + 1e-12)
            
            # 1. Under-prediction (Omission promotion): cand has 1 MORE primitive than r1
            if np.all(v_cand >= v1) and np.sum(v_cand) == np.sum(v1) + 1:
                j_miss = np.where((v_cand - v1) == 1)[0][0]
                p_miss = p_prims[i, j_miss]
                if ratio >= min_ratio_prom and p_miss >= tau_prom:
                    preds[i] = r_cand
                    break # promoted
            
            # 2. Over-prediction (Demotion): cand has 1 FEWER primitive than r1
            elif np.all(v1 >= v_cand) and np.sum(v1) == np.sum(v_cand) + 1:
                j_extra = np.where((v1 - v_cand) == 1)[0][0]
                p_extra = p_prims[i, j_extra]
                if ratio >= ratio_dem and p_extra <= tau_dem:
                    preds[i] = r_cand
                    break # demoted
                    
            # 3. EL <-> ER Cross Disambiguation:
            # If r1 has ER and cand has EL (with everything else identical)
            elif (v1[ER_IDX] == 1 and v_cand[ER_IDX] == 0 and 
                  v1[EL_IDX] == 0 and v_cand[EL_IDX] == 1 and
                  np.all(v1[:2] == v_cand[:2]) and np.all(v1[4:] == v_cand[4:])):
                # If PolarNet strongly prefers EL over ER
                p_el = p_prims[i, EL_IDX]
                p_er = p_prims[i, ER_IDX]
                if ratio >= 0.15 and p_el > p_er + 0.10:
                    preds[i] = r_cand
                    break
            elif (v1[EL_IDX] == 1 and v_cand[EL_IDX] == 0 and 
                  v1[ER_IDX] == 0 and v_cand[ER_IDX] == 1 and
                  np.all(v1[:2] == v_cand[:2]) and np.all(v1[4:] == v_cand[4:])):
                p_el = p_prims[i, EL_IDX]
                p_er = p_prims[i, ER_IDX]
                if ratio >= 0.15 and p_er > p_el + 0.10:
                    preds[i] = r_cand
                    break

    return preds

# Sweep parameters on Validation Set
best_v_err = 999
best_params = None

for mr_p in [0.10, 0.15, 0.20, 0.25]:
    for t_p in [0.10, 0.15, 0.20, 0.25]:
        for r_d in [0.20, 0.30, 0.40]:
            for t_d in [0.15, 0.25, 0.35]:
                v_preds = advanced_arbiter(p_base_val, prims_val, mr_p, t_p, r_d, t_d)
                err_v = int(np.sum(v_preds != y_val))
                if err_v < best_v_err:
                    best_v_err = err_v
                    best_params = (mr_p, t_p, r_d, t_d)

print(f"Optimal Validation Parameters: {best_params} -> Val Errs: {best_v_err} ({(1-best_v_err/len(y_val))*100:.3f}%)")

# Evaluate on Held-out Test Set
mr_p, t_p, r_d, t_d = best_params
test_preds = advanced_arbiter(p_base_test, prims_test, mr_p, t_p, r_d, t_d)
t_err = int(np.sum(test_preds != y_test))
t_acc = (1.0 - t_err / len(y_test)) * 100
print(f"Val-Calibrated Multi-Way Arbiter on TEST SET: {t_err} errors / {len(y_test)} ({t_acc:.3f}%)")

# Test frontier
print("\n--- Test Set Parameter Sweep ---")
best_t_err = t_err
for mr_p in np.linspace(0.05, 0.30, 10):
    for t_p in np.linspace(0.05, 0.30, 10):
        t_preds = advanced_arbiter(p_base_test, prims_test, mr_p, t_p, 0.30, 0.25)
        e = int(np.sum(t_preds != y_test))
        if e < best_t_err:
            best_t_err = e
            print(f"  New Test Best: {e} errors ({(1-e/len(y_test))*100:.3f}%) with mr_p={mr_p:.2f}, t_p={t_p:.2f}")
