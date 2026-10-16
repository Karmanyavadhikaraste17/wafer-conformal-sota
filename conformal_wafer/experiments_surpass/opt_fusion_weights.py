import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

data_t2 = np.load('conformal_wafer/experiments_surpass/trial02_d4_probs.npz')
data_m = np.load('conformal_wafer/experiments_surpass/multi_model_tta_probs.npz')
data_e = np.load('conformal_wafer/experiments_surpass/extra_models_tta_probs.npz')

# 8-view temperature scaled logits if available
data_views = np.load('conformal_wafer/experiments_surpass/trial02_8views_logits.npz')
p_t2_v = np.mean(softmax(data_views['val_views'] / 0.5, axis=-1), axis=1) # 21 val errs
p_t2_t = np.mean(softmax(data_views['test_views'] / 0.5, axis=-1), axis=1) # 78 test errs

p_pol_v = data_m['PolarNet_val_d4']
p_pol_t = data_m['PolarNet_test_d4']

p_wn_v = data_e['WaferNeXt_val_d4']
p_wn_t = data_e['WaferNeXt_test_d4']

p_v2_v = data_m['DPCANet_v2_val_d4']
p_v2_t = data_m['DPCANet_v2_test_d4']

y_val = data_t2['y_val']
y_test = data_t2['y_test']

print("=" * 80)
print("NELDER-MEAD VAL-CALIBRATED FUSION OF SPECIALIZED GEOMETRIC STREAMS")
print("=" * 80)

# We want to find w = [w_t2, w_pol, w_wn, w_v2] minimizing validation NLL
models_v = [p_t2_v, p_pol_v, p_wn_v, p_v2_v]
models_t = [p_t2_t, p_pol_t, p_wn_t, p_v2_t]
names = ['Trial02_Cartesian', 'PolarNet_Cylindrical', 'WaferNeXt_LargeKernel', 'DPCANet_v2']

def loss_fn(w_raw):
    w = softmax(w_raw)
    p_ens = sum(w[i] * models_v[i] for i in range(len(w)))
    p_true = p_ens[np.arange(len(y_val)), y_val]
    nll = -np.mean(np.log(np.clip(p_true, 1e-7, 1.0)))
    return nll

res = minimize(loss_fn, [2.0, -1.0, -1.0, 0.0], method='Nelder-Mead')
weights = softmax(res.x)

print("Learned Weights (Strictly Calibrated on Validation Set):")
for name, w in zip(names, weights):
    print(f"  {name:25s}: {w*100:6.2f}%")

p_val_ens = sum(weights[i] * models_v[i] for i in range(len(weights)))
p_test_ens = sum(weights[i] * models_t[i] for i in range(len(weights)))

e_val = int(np.sum(np.argmax(p_val_ens, axis=1) != y_val))
e_test = int(np.sum(np.argmax(p_test_ens, axis=1) != y_test))

print("\n" + "=" * 80)
print(f"Validation Errors: {e_val:2d} / 3042 ({100*(1-e_val/len(y_val)):.3f}%) [was 19]")
print(f"HELD-OUT TEST ERRORS: {e_test:2d} / 7603 ({100*(1-e_test/len(y_test)):.3f}%) [was 80, SOTA is 66]")
print(f"Delta vs Published SOTA (66 errors): {66 - e_test:+d} errors!")
print("=" * 80)
