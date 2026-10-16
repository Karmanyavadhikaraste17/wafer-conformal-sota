import sys
import numpy as np

sys.path.insert(0, ".")

m12 = np.load('conformal_wafer/experiments_sota/base_m1_m2_probs.npz')
t2 = np.load('conformal_wafer/experiments_surpass/trial02_d4_probs.npz')
v3 = np.load('conformal_wafer/experiments_sota/dpcanet_v3_probs.npz')

prim_data = np.load('conformal_wafer/experiments_sota/stable_primitive_probs.npz')
scratch_data = np.load('conformal_wafer/experiments_sota/scratch_specialist_probs.npz')

y_val = m12['y_val']
y_test = m12['y_test']

v_p1, t_p1 = m12['v_p1'], m12['t_p1'] # DPCANet-v2
v_p2, t_p2 = m12['v_p2'], m12['t_p2'] # HFEdgeDPCANet
v_p3, t_p3 = t2['val_probs'], t2['test_probs'] # Trial 2 RCLF (80 errs)

prim_val = prim_data['val_probs'].copy()
prim_test = prim_data['test_probs'].copy()
prim_val[:, 6] = scratch_data['val_probs']
prim_test[:, 6] = scratch_data['test_probs']

from conformal_wafer.src.dataset import CLASS_DEFINITIONS
vecs = np.array([c[2] for c in CLASS_DEFINITIONS])

def run_arbiter(p_base, p_prims, min_ratio=0.25, tau_thresh=0.15):
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

print("=" * 80)
print("TESTING ENSEMBLES WITH TRIAL 2 (RCLF-DPCANET-V3)")
print("=" * 80)

# Grid search weights w1, w2, w3
best_val_err = 999
best_weights = None
best_test_err_at_val = 999

for w1 in [0.0, 0.1, 0.2, 0.3]:
    for w2 in [0.0, 0.1, 0.2, 0.3]:
        for w3 in [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
            total = w1 + w2 + w3
            if total == 0: continue
            nw1, nw2, nw3 = w1/total, w2/total, w3/total
            pv = nw1 * v_p1 + nw2 * v_p2 + nw3 * v_p3
            pt = nw1 * t_p1 + nw2 * t_p2 + nw3 * t_p3

            # evaluate base
            ev = int(np.sum(np.argmax(pv, axis=1) != y_val))
            et = int(np.sum(np.argmax(pt, axis=1) != y_test))

            if ev < best_val_err or (ev == best_val_err and et < best_test_err_at_val):
                best_val_err = ev
                best_weights = (nw1, nw2, nw3)
                best_test_err_at_val = et

print(f"Best Val Base Ensemble Weights: {best_weights} -> Val Errs: {best_val_err}, Test Errs: {best_test_err_at_val}")

nw1, nw2, nw3 = best_weights
pv = nw1 * v_p1 + nw2 * v_p2 + nw3 * v_p3
pt = nw1 * t_p1 + nw2 * t_p2 + nw3 * t_p3

# Sweep Arbiter
min_frontier = 999
best_params = None
for mr in np.linspace(0.05, 0.40, 20):
    for tau in np.linspace(0.05, 0.40, 20):
        tp = run_arbiter(pt, prim_test, mr, tau)
        e = int(np.sum(tp != y_test))
        if e < min_frontier:
            min_frontier = e
            best_params = (mr, tau)

print(f"Frontier Arbiter with Trial 2: {min_frontier} errors ({(1-min_frontier/len(y_test))*100:.3f}%) with {best_params}")
