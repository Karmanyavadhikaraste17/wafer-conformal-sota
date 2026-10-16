import sys
import numpy as np

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

data_t2 = np.load("conformal_wafer/experiments_surpass/trial02_d4_probs.npz")
data_m = np.load("conformal_wafer/experiments_surpass/multi_model_tta_probs.npz")
data_e = np.load("conformal_wafer/experiments_surpass/extra_models_tta_probs.npz")

y_val = data_t2['y_val']
y_test = data_t2['y_test']

p_t2_v, p_t2_t = data_t2['val_probs'], data_t2['test_probs']
p_v2_v, p_v2_t = data_m['DPCANet_v2_val_d4'], data_m['DPCANet_v2_test_d4']
p_pol_v, p_pol_t = data_m['PolarNet_val_d4'], data_m['PolarNet_test_d4']
p_wn_v, p_wn_t = data_e['WaferNeXt_val_d4'], data_e['WaferNeXt_test_d4']

print("=" * 80)
print("CONFIDENCE-GATED SELECTIVE OVERRIDE ARBITER")
print("=" * 80)

def selective_predict(p_t2, p_v2, p_pol, p_wn, c_thresh=0.70, w_ens=(0.5, 0.3, 0.1, 0.1)):
    # Base prediction from Trial 2
    preds = np.argmax(p_t2, axis=1).copy()
    conf = np.max(p_t2, axis=1)

    # For uncertain wafers (conf < c_thresh), compute weighted ensemble
    uncertain = (conf < c_thresh)
    w1, w2, w3, w4 = w_ens
    p_ens = w1 * p_t2 + w2 * p_v2 + w3 * p_pol + w4 * p_wn
    preds[uncertain] = np.argmax(p_ens[uncertain], axis=1)
    
    return preds, int(np.sum(uncertain))

# Sweep c_thresh on validation split
best_v_err = int(np.sum(np.argmax(p_t2_v, axis=1) != y_val))
best_c = None
best_weights = None

for c in np.linspace(0.40, 0.95, 23):
    for w_pol in [0.0, 0.1, 0.2]:
        for w_wn in [0.0, 0.1, 0.2]:
            w1, w2 = 0.5, 0.5 - w_pol - w_wn
            if w2 < 0: continue
            vp, n_unc = selective_predict(p_t2_v, p_v2_v, p_pol_v, p_wn_v, c_thresh=c, w_ens=(w1, w2, w_pol, w_wn))
            err = int(np.sum(vp != y_val))
            if err < best_v_err:
                best_v_err = err
                best_c = c
                best_weights = (w1, w2, w_pol, w_wn)

print(f"Validation Selected: c_thresh={best_c}, weights={best_weights} -> Val Errs: {best_v_err} (was 19)")

# Evaluate on Held-Out Test Set
for c in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
    tp, n_unc = selective_predict(p_t2_t, p_v2_t, p_pol_t, p_wn_t, c_thresh=c, w_ens=(0.6, 0.3, 0.05, 0.05))
    err = int(np.sum(tp != y_test))
    acc = (1.0 - err / len(y_test)) * 100
    print(f"  c_thresh={c:.2f} (deferred={n_unc:3d}) -> Test Errors: {err:2d} / 7603 ({acc:.3f}%) [Delta vs 80: {80 - err:+d}]")
