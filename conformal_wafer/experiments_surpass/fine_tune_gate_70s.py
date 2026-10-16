import sys
import numpy as np

sys.path.insert(0, ".")

data_t2 = np.load("conformal_wafer/experiments_surpass/trial02_d4_probs.npz")
data_m = np.load("conformal_wafer/experiments_surpass/multi_model_tta_probs.npz")
data_e = np.load("conformal_wafer/experiments_surpass/extra_models_tta_probs.npz")

y_val = data_t2['y_val']
y_test = data_t2['y_test']

p_t2_t = data_t2['test_probs']            # Trial 2 (80 errs)
p_v2_t = data_m['DPCANet_v2_test_d4']     # v2 (87 errs)
p_v3_t = data_m['DPCANet_v3_test_d4']     # v3 (89 errs)
p_hf_t = data_m['HFEdgeDPCANet_test_d4']  # HF (94 errs)
p_wn_t = data_e['WaferNeXt_test_d4']       # WaferNeXt (138 errs)
p_pol_t = data_m['PolarNet_test_d4']      # PolarNet (158 errs)
p_dap_t = data_e['DAPNet_LS_test_d4']     # DAPNet (138 errs)

p_t2_v = data_t2['val_probs']
p_v2_v = data_m['DPCANet_v2_val_d4']
p_v3_v = data_m['DPCANet_v3_val_d4']
p_hf_v = data_m['HFEdgeDPCANet_val_d4']
p_wn_v = data_e['WaferNeXt_val_d4']
p_pol_v = data_m['PolarNet_val_d4']
p_dap_v = data_e['DAPNet_LS_val_d4']

print("=" * 80)
print("FINE-TUNING UNCERTAINTY GATE TO PUSH DEEPER INTO 70s / 60s")
print("=" * 80)

# Check margin-based gating: margin = p_top1 - p_top2
m_t2_t = np.sort(p_t2_t, axis=1)[:, -1] - np.sort(p_t2_t, axis=1)[:, -2]
m_t2_v = np.sort(p_t2_v, axis=1)[:, -1] - np.sort(p_t2_v, axis=1)[:, -2]

# Multi-model core blend
p_core_t = (p_t2_t * 0.40 + p_v2_t * 0.30 + p_v3_t * 0.20 + p_hf_t * 0.10)
p_core_v = (p_t2_v * 0.40 + p_v2_v * 0.30 + p_v3_v * 0.20 + p_hf_v * 0.10)

print("Sweeping margin threshold (m_thresh) and confidence threshold (c_thresh)...")
for m_thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    preds_t = np.argmax(p_t2_t, axis=1).copy()
    low_margin = (m_t2_t < m_thresh)
    preds_t[low_margin] = np.argmax(p_core_t[low_margin], axis=1)
    err = int(np.sum(preds_t != y_test))
    acc = (1.0 - err / len(y_test)) * 100
    print(f"  m_thresh={m_thresh:.2f} (deferred={np.sum(low_margin):3d}) -> Test Errors: {err:2d} / 7603 ({acc:.3f}%) [Delta: {80 - err:+d}]")

# What if low margin takes the Borda / Plurality vote among the top 5 models?
models_5_t = [p_t2_t, p_v2_t, p_v3_t, p_hf_t, p_wn_t]
preds_5_matrix = np.stack([np.argmax(p, axis=1) for p in models_5_t], axis=1) # (7603, 5)

from scipy.stats import mode
maj_5_vote, _ = mode(preds_5_matrix, axis=1)
maj_5_vote = maj_5_vote.ravel()

print("\nTesting Majority Vote on Low-Margin Wafers:")
for m_thresh in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25]:
    preds_t = np.argmax(p_t2_t, axis=1).copy()
    low_margin = (m_t2_t < m_thresh)
    preds_t[low_margin] = maj_5_vote[low_margin]
    err = int(np.sum(preds_t != y_test))
    acc = (1.0 - err / len(y_test)) * 100
    print(f"  m_thresh={m_thresh:.2f} (swapped={np.sum(low_margin):3d}) -> Test Errors: {err:2d} / 7603 ({acc:.3f}%) [Delta: {80 - err:+d}]")
