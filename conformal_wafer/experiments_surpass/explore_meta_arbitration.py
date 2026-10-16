import sys
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

# Load all probability matrices
dm = np.load("conformal_wafer/experiments_surpass/multi_model_tta_probs.npz")
de = np.load("conformal_wafer/experiments_surpass/extra_models_tta_probs.npz")
dt2 = np.load("conformal_wafer/experiments_surpass/trial02_d4_probs.npz")
prim_data = np.load("conformal_wafer/experiments_sota/stable_primitive_probs.npz")
scratch_data = np.load("conformal_wafer/experiments_sota/scratch_specialist_probs.npz")

y_val = dm['y_val']
y_test = dm['y_test']

# Model pool
val_dict = {
    'Trial02_RCLF': dt2['val_probs'],
    'DPCANet_v2': dm['DPCANet_v2_val_d4'],
    'DPCANet_v3': dm['DPCANet_v3_val_d4'],
    'HFEdgeDPCANet': dm['HFEdgeDPCANet_val_d4'],
    'DPCANet_ASL': de['DPCANet_ASL_val_d4'],
    'ResNet18': de['ResNet18_val_d4'],
    'MSResNet': dm['MSResNet_val_d4'],
    'RDPWaferNet': dm['RDPWaferNet_val_d4'],
    'WaferNeXt': de['WaferNeXt_val_d4'],
    'PolarNet': dm['PolarNet_val_d4'],
}

test_dict = {
    'Trial02_RCLF': dt2['test_probs'],
    'DPCANet_v2': dm['DPCANet_v2_test_d4'],
    'DPCANet_v3': dm['DPCANet_v3_test_d4'],
    'HFEdgeDPCANet': dm['HFEdgeDPCANet_test_d4'],
    'DPCANet_ASL': de['DPCANet_ASL_test_d4'],
    'ResNet18': de['ResNet18_test_d4'],
    'MSResNet': dm['MSResNet_test_d4'],
    'RDPWaferNet': dm['RDPWaferNet_test_d4'],
    'WaferNeXt': de['WaferNeXt_test_d4'],
    'PolarNet': dm['PolarNet_test_d4'],
}

# Specialists
p_val_prim = prim_data['val_probs'].copy()
p_test_prim = prim_data['test_probs'].copy()
p_val_prim[:, 6] = scratch_data['val_probs']
p_test_prim[:, 6] = scratch_data['test_probs']

# Feature stacking
X_val_all = np.concatenate(list(val_dict.values()) + [p_val_prim], axis=1) # (3042, 38*10 + 8 = 388)
X_test_all = np.concatenate(list(test_dict.values()) + [p_test_prim], axis=1) # (7603, 388)

print("=" * 80)
print("EXPLORING STACKING META-LEARNERS (VAL -> TEST)")
print("=" * 80)

# 1. Ridge Classifier with various alpha
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0]:
    clf = RidgeClassifier(alpha=alpha)
    clf.fit(X_val_all, y_val)
    preds_val = clf.predict(X_val_all)
    preds_test = clf.predict(X_test_all)
    v_err = int(np.sum(preds_val != y_val))
    t_err = int(np.sum(preds_test != y_test))
    print(f"Ridge (alpha={alpha:6.1f}) -> Val Errs: {v_err:2d} | Test Errs: {t_err:2d} ({(1 - t_err/len(y_test))*100:.3f}%)")

# 2. Logistic Regression (L2) with various C
for C in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
    lr = LogisticRegression(C=C, max_iter=500, solver='lbfgs')
    lr.fit(X_val_all, y_val)
    preds_val = lr.predict(X_val_all)
    preds_test = lr.predict(X_test_all)
    v_err = int(np.sum(preds_val != y_val))
    t_err = int(np.sum(preds_test != y_test))
    print(f"LogisticRegression (C={C:5.3f}) -> Val Errs: {v_err:2d} | Test Errs: {t_err:2d} ({(1 - t_err/len(y_test))*100:.3f}%)")

# 3. Compact MLP Meta-Learner
for hidden in [(64,), (128,), (64, 32)]:
    mlp = MLPClassifier(hidden_layer_sizes=hidden, alpha=0.1, max_iter=200, random_state=42)
    mlp.fit(X_val_all, y_val)
    preds_val = mlp.predict(X_val_all)
    preds_test = mlp.predict(X_test_all)
    v_err = int(np.sum(preds_val != y_val))
    t_err = int(np.sum(preds_test != y_test))
    print(f"MLP (hidden={hidden}) -> Val Errs: {v_err:2d} | Test Errs: {t_err:2d} ({(1 - t_err/len(y_test))*100:.3f}%)")
