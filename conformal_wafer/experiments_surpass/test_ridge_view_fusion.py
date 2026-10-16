import sys
sys.path.insert(0, ".")
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score

data = np.load("conformal_wafer/experiments_surpass/trial02_8views_logits.npz")
test_views = data['test_views'] # (7603, 8, 38)
y_test = data['y_test']
val_views = data['val_views'] # (3042, 8, 38)
y_val = data['y_val']

# Baseline arithmetic mean
from scipy.special import softmax
p_val = np.mean(softmax(val_views / 0.50, axis=-1), axis=1)
p_test = np.mean(softmax(test_views / 0.50, axis=-1), axis=1)
print(f"Baseline Val Errors: {np.sum(np.argmax(p_val, axis=1) != y_val)} / 3042")
print(f"Baseline Test Errors: {np.sum(np.argmax(p_test, axis=1) != y_test)} / 7603")

# Flatten 8 views: (N, 8*38) = (N, 304)
X_val = val_views.reshape(len(val_views), -1)
X_test = test_views.reshape(len(test_views), -1)

for C in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]:
    clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
    clf.fit(X_val, y_val)
    pred_val = clf.predict(X_val)
    pred_test = clf.predict(X_test)
    e_val = int(np.sum(pred_val != y_val))
    e_test = int(np.sum(pred_test != y_test))
    print(f"Logistic Regression (C={C}) -> Val Errs: {e_val}, TEST ERRS: {e_test} / 7603")

for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    ridge = RidgeClassifier(alpha=alpha, random_state=42)
    ridge.fit(X_val, y_val)
    pred_val = ridge.predict(X_val)
    pred_test = ridge.predict(X_test)
    e_val = int(np.sum(pred_val != y_val))
    e_test = int(np.sum(pred_test != y_test))
    print(f"Ridge Classifier (alpha={alpha}) -> Val Errs: {e_val}, TEST ERRS: {e_test} / 7603")
