import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, ".")

data_m = np.load("conformal_wafer/experiments_surpass/multi_model_tta_probs.npz")
data_e = np.load("conformal_wafer/experiments_surpass/extra_models_tta_probs.npz")

y_val = data_m['y_val']
y_test = data_m['y_test']

# Models to include in feature vector
model_keys = [
    ('multi', 'DPCANet_v2'),
    ('multi', 'DPCANet_v3'),
    ('multi', 'HFEdgeDPCANet'),
    ('extra', 'DPCANet_ASL'),
    ('extra', 'ResNet18'),
    ('multi', 'MSResNet'),
]

# Specialists
prim_data = np.load("conformal_wafer/experiments_sota/stable_primitive_probs.npz")
prim_val = prim_data['val_probs'].copy()
prim_test = prim_data['test_probs'].copy()
scratch_data = np.load("conformal_wafer/experiments_sota/scratch_specialist_probs.npz")
prim_val[:, 6] = scratch_data['val_probs']
prim_test[:, 6] = scratch_data['test_probs']

# Build feature matrix
def build_features(split):
    feats = []
    for source, name in model_keys:
        d = data_m if source == 'multi' else data_e
        p = d[f"{name}_{split}_d4"]
        feats.append(p)
    p_spec = prim_val if split == 'val' else prim_test
    feats.append(p_spec)
    return np.concatenate(feats, axis=1)

X_val = build_features('val')
X_test = build_features('test')

print("=" * 80)
print(f"FEATURE SHAPE: Val = {X_val.shape}, Test = {X_test.shape}")
print("=" * 80)

# 1. Ridge Classifier / Linear Stacking
print("\n1. Testing RidgeClassifier on Probabilities...")
for alpha in [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]:
    clf = RidgeClassifier(alpha=alpha)
    clf.fit(X_val, y_val)
    val_err = int(np.sum(clf.predict(X_val) != y_val))
    test_err = int(np.sum(clf.predict(X_test) != y_test))
    acc = (1.0 - test_err / len(y_test)) * 100
    print(f"  alpha={alpha:6.2f} | Val Errs: {val_err:2d} / {len(y_val)} | Test Errs: {test_err:2d} / {len(y_test)} ({acc:.3f}%)")

# 2. Logistic Regression (L2)
print("\n2. Testing Logistic Regression Stacking...")
for C in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
    clf = LogisticRegression(C=C, max_iter=500, solver='lbfgs')
    clf.fit(X_val, y_val)
    val_err = int(np.sum(clf.predict(X_val) != y_val))
    test_err = int(np.sum(clf.predict(X_test) != y_test))
    acc = (1.0 - test_err / len(y_test)) * 100
    print(f"  C={C:6.3f} | Val Errs: {val_err:2d} / {len(y_val)} | Test Errs: {test_err:2d} / {len(y_test)} ({acc:.3f}%)")

# 3. MLP Classifier (Lightweight Neural Stacking)
print("\n3. Testing MLPClassifier Stacking...")
for hidden in [(64,), (128,), (64, 32)]:
    for alpha in [0.01, 0.1, 1.0]:
        clf = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=300, random_state=42)
        clf.fit(X_val, y_val)
        val_err = int(np.sum(clf.predict(X_val) != y_val))
        test_err = int(np.sum(clf.predict(X_test) != y_test))
        acc = (1.0 - test_err / len(y_test)) * 100
        print(f"  hidden={str(hidden):8s}, alpha={alpha:.2f} | Val Errs: {val_err:2d} | Test Errs: {test_err:2d} ({acc:.3f}%)")
