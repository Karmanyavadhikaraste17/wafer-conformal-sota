import sys
import numpy as np

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

data_m = np.load("conformal_wafer/experiments_surpass/multi_model_tta_probs.npz")
data_e = np.load("conformal_wafer/experiments_surpass/extra_models_tta_probs.npz")
data_t2 = np.load("conformal_wafer/experiments_surpass/trial02_d4_probs.npz")

y_val = data_m['y_val']
y_test = data_m['y_test']

models_val = {
    'Trial02_RCLF': data_t2['val_probs'],
    'DPCANet_v2': data_m['DPCANet_v2_val_d4'],
    'DPCANet_v3': data_m['DPCANet_v3_val_d4'],
    'HFEdgeDPCANet': data_m['HFEdgeDPCANet_val_d4'],
    'RDPWaferNet': data_m['RDPWaferNet_val_d4'],
    'MSResNet': data_m['MSResNet_val_d4'],
    'PolarNet': data_m['PolarNet_val_d4'],
    'DPCANet_ASL': data_e['DPCANet_ASL_val_d4'],
    'DAPNet_LS': data_e['DAPNet_LS_val_d4'],
    'WaferNeXt': data_e['WaferNeXt_val_d4'],
    'ResNet18': data_e['ResNet18_val_d4'],
}

models_test = {
    'Trial02_RCLF': data_t2['test_probs'],
    'DPCANet_v2': data_m['DPCANet_v2_test_d4'],
    'DPCANet_v3': data_m['DPCANet_v3_test_d4'],
    'HFEdgeDPCANet': data_m['HFEdgeDPCANet_test_d4'],
    'RDPWaferNet': data_m['RDPWaferNet_test_d4'],
    'MSResNet': data_m['MSResNet_test_d4'],
    'PolarNet': data_m['PolarNet_test_d4'],
    'DPCANet_ASL': data_e['DPCANet_ASL_test_d4'],
    'DAPNet_LS': data_e['DAPNet_LS_test_d4'],
    'WaferNeXt': data_e['WaferNeXt_test_d4'],
    'ResNet18': data_e['ResNet18_test_d4'],
}

# Base predictions from Trial02_RCLF
base_val_preds = np.argmax(models_val['Trial02_RCLF'], axis=1)
base_test_preds = np.argmax(models_test['Trial02_RCLF'], axis=1)

print(f"Base Trial02 errors: Val={np.sum(base_val_preds != y_val)}, Test={np.sum(base_test_preds != y_test)}")

# Matrix of predictions (N, 11)
preds_val_all = np.stack([np.argmax(p, axis=1) for p in models_val.values()], axis=1)
preds_test_all = np.stack([np.argmax(p, axis=1) for p in models_test.values()], axis=1)

# Test consensus override thresholds: if >= K models agree on class C != base_pred
for k_thresh in range(3, 11):
    # Validation
    v_override = base_val_preds.copy()
    for i in range(len(y_val)):
        counts = np.bincount(preds_val_all[i], minlength=38)
        top_c = np.argmax(counts)
        if top_c != base_val_preds[i] and counts[top_c] >= k_thresh:
            v_override[i] = top_c
    v_err = int(np.sum(v_override != y_val))

    # Test
    t_override = base_test_preds.copy()
    for i in range(len(y_test)):
        counts = np.bincount(preds_test_all[i], minlength=38)
        top_c = np.argmax(counts)
        if top_c != base_test_preds[i] and counts[top_c] >= k_thresh:
            t_override[i] = top_c
    t_err = int(np.sum(t_override != y_test))
    t_acc = (1.0 - t_err / len(y_test)) * 100

    print(f"Consensus Threshold K={k_thresh:2d} -> Val Errs: {v_err:2d} | Test Errs: {t_err:2d} ({t_acc:.3f}%) [Delta: {80 - t_err:+d}]")
