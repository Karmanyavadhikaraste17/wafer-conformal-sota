import os
import sys
import numpy as np

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

data_m = np.load("conformal_wafer/experiments_surpass/multi_model_tta_probs.npz")
data_e = np.load("conformal_wafer/experiments_surpass/extra_models_tta_probs.npz")
data_t2 = np.load("conformal_wafer/experiments_surpass/trial02_d4_probs.npz")

y_val = data_m['y_val']
y_test = data_m['y_test']

models_11 = {
    'DPCANet_v2': (data_m['DPCANet_v2_val_d4'], data_m['DPCANet_v2_test_d4']),
    'DPCANet_v3': (data_m['DPCANet_v3_val_d4'], data_m['DPCANet_v3_test_d4']),
    'HFEdgeDPCANet': (data_m['HFEdgeDPCANet_val_d4'], data_m['HFEdgeDPCANet_test_d4']),
    'RDPWaferNet': (data_m['RDPWaferNet_val_d4'], data_m['RDPWaferNet_test_d4']),
    'MSResNet': (data_m['MSResNet_val_d4'], data_m['MSResNet_test_d4']),
    'PolarNet': (data_m['PolarNet_val_d4'], data_m['PolarNet_test_d4']),
    'DPCANet_ASL': (data_e['DPCANet_ASL_val_d4'], data_e['DPCANet_ASL_test_d4']),
    'DAPNet_LS': (data_e['DAPNet_LS_val_d4'], data_e['DAPNet_LS_test_d4']),
    'WaferNeXt': (data_e['WaferNeXt_val_d4'], data_e['WaferNeXt_test_d4']),
    'ResNet18': (data_e['ResNet18_val_d4'], data_e['ResNet18_test_d4']),
    'Trial02_RCLF': (data_t2['val_probs'], data_t2['test_probs']),
}

print("=" * 80)
print("11-MODEL POOL PERFORMANCE ON HELD-OUT TEST SET (7,603 WAFERS)")
print("=" * 80)
for name, (vp, tp) in models_11.items():
    err_t = int(np.sum(np.argmax(tp, axis=1) != y_test))
    acc_t = (1.0 - err_t / len(y_test)) * 100
    err_v = int(np.sum(np.argmax(vp, axis=1) != y_val))
    print(f"  {name:15s} | Val Errs: {err_v:2d} | Test Errs: {err_t:2d} ({acc_t:.3f}%)")

# 11-Model Oracle Bound
all_preds = np.stack([np.argmax(tp, axis=1) for (vp, tp) in models_11.values()], axis=1)
oracle_correct = np.any(all_preds == y_test[:, None], axis=1)
oracle_errs = int(len(y_test) - np.sum(oracle_correct))
oracle_acc = (1.0 - oracle_errs / len(y_test)) * 100
print("\n" + "=" * 80)
print(f"*** 11-MODEL ORACLE BOUND: {oracle_acc:.3f}% ({oracle_errs} errors / {len(y_test)}) ***")
print("=" * 80)
