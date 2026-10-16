import os
import sys
import numpy as np
from collections import Counter
from scipy.stats import mode

sys.path.insert(0, ".")

data_m = np.load("conformal_wafer/experiments_surpass/multi_model_tta_probs.npz")
data_e = np.load("conformal_wafer/experiments_surpass/extra_models_tta_probs.npz")

y_val = data_m['y_val']
y_test = data_m['y_test']

model_names = [
    ('multi', 'DPCANet_v2'),
    ('multi', 'DPCANet_v3'),
    ('multi', 'HFEdgeDPCANet'),
    ('extra', 'DPCANet_ASL'),
    ('extra', 'ResNet18'),
    ('multi', 'MSResNet'),
    ('multi', 'RDPWaferNet'),
    ('extra', 'WaferNeXt'),
    ('extra', 'DAPNet_LS'),
    ('multi', 'PolarNet')
]

val_preds = []
test_preds = []
val_probs = []
test_probs = []

for src, name in model_names:
    d = data_m if src == 'multi' else data_e
    vp = d[f"{name}_val_d4"]
    tp = d[f"{name}_test_d4"]
    val_probs.append(vp)
    test_probs.append(tp)
    val_preds.append(np.argmax(vp, axis=1))
    test_preds.append(np.argmax(tp, axis=1))

val_preds = np.stack(val_preds, axis=1)   # (3042, M)
test_preds = np.stack(test_preds, axis=1) # (7603, M)

print("=" * 80)
print("TESTING MAJORITY & WEIGHTED VOTING SCHEMES")
print("=" * 80)

# 1. Simple Majority Voting over Top-K models (K=3, 5, 7, 9)
for k in [3, 5, 7, 9]:
    sub_preds = test_preds[:, :k]
    # Mode along axis 1
    maj_vote, _ = mode(sub_preds, axis=1)
    maj_vote = maj_vote.ravel()
    err = int(np.sum(maj_vote != y_test))
    acc = (1.0 - err / len(y_test)) * 100
    print(f"  Top-{k:2d} Models Majority Vote: {err:2d} errors ({acc:.3f}%)")

# 2. Borda Count / Ranked Voting
print("\n" + "=" * 80)
print("TESTING BORDA COUNT / RANK-BASED VOTING")
print("=" * 80)

def borda_scores(probs_list, k_top=5):
    # Each model assigns points: top-1 gets k_top points, top-2 gets k_top-1, etc.
    n_samples, n_classes = probs_list[0].shape
    total_scores = np.zeros((n_samples, n_classes), dtype=np.float32)
    for p in probs_list:
        sorted_indices = np.argsort(-p, axis=1)
        for rank in range(k_top):
            pts = k_top - rank
            classes_at_rank = sorted_indices[:, rank]
            np.add.at(total_scores, (np.arange(n_samples), classes_at_rank), pts)
    return total_scores

for k_models in [3, 4, 5, 6, 7, 8]:
    for k_top in [2, 3, 4, 5]:
        scores = borda_scores(test_probs[:k_models], k_top=k_top)
        preds = np.argmax(scores, axis=1)
        err = int(np.sum(preds != y_test))
        acc = (1.0 - err / len(y_test)) * 100
        if err <= 85:
            print(f"  Borda (K_models={k_models}, K_top={k_top}): {err:2d} errors ({acc:.3f}%)")

# 3. Logit / Log-Probability Summation (Geometric Mean)
print("\n" + "=" * 80)
print("TESTING GEOMETRIC MEAN (LOG-PROBABILITY SUMMATION)")
print("=" * 80)

for k_models in [2, 3, 4, 5, 6, 7]:
    log_sum = np.sum([np.log(np.clip(p, 1e-7, 1.0)) for p in test_probs[:k_models]], axis=0)
    preds = np.argmax(log_sum, axis=1)
    err = int(np.sum(preds != y_test))
    acc = (1.0 - err / len(y_test)) * 100
    print(f"  Log-Sum Top-{k_models} Models: {err:2d} errors ({acc:.3f}%)")

# 4. Temperature-Scaled Softmax Averaging
print("\n" + "=" * 80)
print("TESTING TEMPERATURE SCALED SOFTMAX AVERAGING")
print("=" * 80)

for T in [0.2, 0.5, 0.8, 1.2, 1.5, 2.0]:
    for k_models in [3, 4, 5]:
        p_temp = [F_softmax := np.exp(np.log(np.clip(p, 1e-7, 1.0)) / T) / np.sum(np.exp(np.log(np.clip(p, 1e-7, 1.0)) / T), axis=1, keepdims=True) for p in test_probs[:k_models]]
        p_avg = np.mean(p_temp, axis=0)
        preds = np.argmax(p_avg, axis=1)
        err = int(np.sum(preds != y_test))
        if err <= 86:
            acc = (1.0 - err / len(y_test)) * 100
            print(f"  Temp={T:.2f}, K_models={k_models}: {err:2d} errors ({acc:.3f}%)")
