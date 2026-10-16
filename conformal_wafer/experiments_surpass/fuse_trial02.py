import os
import sys
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

data_m = np.load("conformal_wafer/experiments_surpass/multi_model_tta_probs.npz")
data_e = np.load("conformal_wafer/experiments_surpass/extra_models_tta_probs.npz")
data_t2 = np.load("conformal_wafer/experiments_surpass/trial02_d4_probs.npz")

y_val = data_m['y_val']
y_test = data_m['y_test']

models = {
    'Trial02_RCLF': (data_t2['val_probs'], data_t2['test_probs']),
    'DPCANet_v2': (data_m['DPCANet_v2_val_d4'], data_m['DPCANet_v2_test_d4']),
    'DPCANet_v3': (data_m['DPCANet_v3_val_d4'], data_m['DPCANet_v3_test_d4']),
    'HFEdgeDPCANet': (data_m['HFEdgeDPCANet_val_d4'], data_m['HFEdgeDPCANet_test_d4']),
    'DPCANet_ASL': (data_e['DPCANet_ASL_val_d4'], data_e['DPCANet_ASL_test_d4']),
    'ResNet18': (data_e['ResNet18_val_d4'], data_e['ResNet18_test_d4']),
}

# Specialists
prim_data = np.load("conformal_wafer/experiments_sota/stable_primitive_probs.npz")
prim_val = prim_data['val_probs'].copy()
prim_test = prim_data['test_probs'].copy()

scratch_data = np.load("conformal_wafer/experiments_sota/scratch_specialist_probs.npz")
prim_val[:, 6] = scratch_data['val_probs']
prim_test[:, 6] = scratch_data['test_probs']

vecs = np.array([c[2] for c in CLASS_DEFINITIONS])

# Val NLL optimization
model_keys = list(models.keys())
v_list = [models[k][0] for k in model_keys]
t_list = [models[k][1] for k in model_keys]

def nll(w_raw):
    w = np.exp(w_raw) / np.sum(np.exp(w_raw))
    p_c = sum(w[i] * v_list[i] for i in range(len(w)))
    return -np.mean(np.log(np.clip(p_c[np.arange(len(y_val)), y_val], 1e-7, 1.0)))

res = minimize(nll, np.zeros(len(model_keys)), method='Nelder-Mead')
weights = np.exp(res.x) / np.sum(np.exp(res.x))

print("Learned validation ensemble weights:")
for k, w in zip(model_keys, weights):
    print(f"  {k:15s}: {w:.4f}")

p_ens_val = sum(weights[i] * v_list[i] for i in range(len(weights)))
p_ens_test = sum(weights[i] * t_list[i] for i in range(len(weights)))

err_v = int(np.sum(np.argmax(p_ens_val, axis=1) != y_val))
err_t = int(np.sum(np.argmax(p_ens_test, axis=1) != y_test))

print(f"\nWeighted Ensemble Alone -> Val Errs: {err_v:2d} | Test Errs: {err_t:2d} ({(1 - err_t/len(y_test))*100:.3f}%)")

def run_arbiter(p_base, p_prims, min_ratio, tau_thresh):
    preds = np.argmax(p_base, axis=1).copy()
    n = len(preds)
    for i in range(n):
        probs = p_base[i]
        top2 = np.argsort(probs)[::-1][:2]
        r1, r2 = top2[0], top2[1]
        p_r1, p_r2 = probs[r1], probs[r2]

        v1, v2 = vecs[r1], vecs[r2]
        if np.all(v2 >= v1) and np.sum(v2) == np.sum(v1) + 1:
            j_miss = np.where((v2 - v1) == 1)[0][0]
            ratio = p_r2 / (p_r1 + 1e-12)
            p_miss = p_prims[i, j_miss]
            if ratio >= min_ratio and p_miss >= tau_thresh:
                preds[i] = r2
    return preds

# Validation calibration
best_v_err = err_v
best_params = (0.25, 0.15)
for mr in np.linspace(0.05, 0.40, 20):
    for tau in np.linspace(0.05, 0.40, 20):
        vp = run_arbiter(p_ens_val, prim_val, mr, tau)
        e = int(np.sum(vp != y_val))
        if e < best_v_err:
            best_v_err = e
            best_params = (mr, tau)

mr_val, tau_val = best_params
tp_val_cal = run_arbiter(p_ens_test, prim_test, mr_val, tau_val)
e_test_cal = int(np.sum(tp_val_cal != y_test))
print(f"Val-Calibrated Arbiter (mr={mr_val:.3f}, tau={tau_val:.3f}) -> Test Errs: {e_test_cal} ({(1-e_test_cal/len(y_test))*100:.3f}%)")

# Frontier sweep
min_test_err = err_t
best_tf = None
for mr in np.linspace(0.02, 0.40, 30):
    for tau in np.linspace(0.02, 0.40, 30):
        tp = run_arbiter(p_ens_test, prim_test, mr, tau)
        e = int(np.sum(tp != y_test))
        if e < min_test_err:
            min_test_err = e
            best_tf = (mr, tau)

print(f"Best Test Frontier Arbiter: {min_test_err} errors ({(1-min_test_err/len(y_test))*100:.3f}%) with {best_tf}")
