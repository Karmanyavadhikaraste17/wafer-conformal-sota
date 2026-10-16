import sys
import numpy as np
from collections import Counter

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

data_m = np.load("conformal_wafer/experiments_surpass/multi_model_tta_probs.npz")
data_e = np.load("conformal_wafer/experiments_surpass/extra_models_tta_probs.npz")
data_t2 = np.load("conformal_wafer/experiments_surpass/trial02_d4_probs.npz")

y_test = data_m['y_test']
NAMES = [c[1] for c in CLASS_DEFINITIONS]
VECS = np.array([c[2] for c in CLASS_DEFINITIONS])

models = {
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
    'Trial02_RCLF': data_t2['test_probs'],
}

preds = {name: np.argmax(prob, axis=1) for name, prob in models.items()}
t2_preds = preds['Trial02_RCLF']

# Oracle mask: at least one model got it right
all_preds_matrix = np.stack(list(preds.values()), axis=1) # (7603, 11)
oracle_correct = np.any(all_preds_matrix == y_test[:, None], axis=1) # (7603,)

# 80 error mask for Trial02_RCLF
t2_errors = (t2_preds != y_test)

# Wafers that Trial02 got WRONG, but Oracle got RIGHT!
recoverable_indices = np.where(t2_errors & oracle_correct)[0]
print(f"Total recoverable wafers from Oracle: {len(recoverable_indices)} wafers!")

print("\n" + "=" * 80)
print("RECOVERABLE WAFERS BREAKDOWN:")
print("=" * 80)
for idx in recoverable_indices:
    y_true = y_test[idx]
    y_pred_t2 = t2_preds[idx]
    
    # Which models got it right?
    correct_models = [m for m, p in preds.items() if p[idx] == y_true]
    print(f"Wafer {idx:4d} | True: {NAMES[y_true]:15s} | Trial2 Pred: {NAMES[y_pred_t2]:15s} | Correct Models ({len(correct_models)}): {', '.join(correct_models)}")

print("\n" + "=" * 80)
print("WHICH MODELS RECOVER THE MOST ERRORS?")
print("=" * 80)
model_recovery_counts = Counter()
for idx in recoverable_indices:
    for m, p in preds.items():
        if p[idx] == y_test[idx]:
            model_recovery_counts[m] += 1

for m, cnt in model_recovery_counts.most_common():
    print(f"  {m:15s}: recovered {cnt:2d} / {len(recoverable_indices)} errors")
