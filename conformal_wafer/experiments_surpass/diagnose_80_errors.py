import os
import sys
import numpy as np
import pandas as pd
from collections import Counter

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

VECS = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.int64)
NAMES = [c[1] for c in CLASS_DEFINITIONS]

m12 = np.load('conformal_wafer/experiments_sota/base_m1_m2_probs.npz')
v3 = np.load('conformal_wafer/experiments_sota/dpcanet_v3_probs.npz')
prim_data = np.load('conformal_wafer/experiments_sota/stable_primitive_probs.npz')
scratch_data = np.load('conformal_wafer/experiments_sota/scratch_specialist_probs.npz')

y_val = m12['y_val']
y_test = m12['y_test']

v_p1, t_p1 = m12['v_p1'], m12['t_p1']
v_p2, t_p2 = m12['v_p2'], m12['t_p2']
v_p3, t_p3 = v3['p_val'], v3['p_test']

# Tri-ensemble
p_tri_test = 0.222 * t_p1 + 0.111 * t_p2 + 0.667 * t_p3

prim_test = prim_data['test_probs'].copy()
prim_test[:, 6] = scratch_data['test_probs']

# Arbiter that gets 80 errors
def run_arbiter(p_base, p_prims, min_ratio=0.25, tau_thresh=0.15):
    preds = np.argmax(p_base, axis=1).copy()
    n = len(preds)
    for i in range(n):
        probs = p_base[i]
        top2 = np.argsort(probs)[::-1][:2]
        r1, r2 = top2[0], top2[1]
        p_r1, p_r2 = probs[r1], probs[r2]

        v1, v2 = VECS[r1], VECS[r2]
        if np.all(v2 >= v1) and np.sum(v2) == np.sum(v1) + 1:
            j_miss = np.where((v2 - v1) == 1)[0][0]
            ratio = p_r2 / (p_r1 + 1e-12)
            p_miss = p_prims[i, j_miss]
            if ratio >= min_ratio and p_miss >= tau_thresh:
                preds[i] = r2
    return preds

preds_80 = run_arbiter(p_tri_test, prim_test, min_ratio=0.25, tau_thresh=0.15)
err_mask = (preds_80 != y_test)
err_indices = np.where(err_mask)[0]
print(f"Total errors: {len(err_indices)} / {len(y_test)}")

# Error analysis
print("\n" + "=" * 80)
print("TAXONOMY OF THE 80 ERRORS:")
print("=" * 80)

tax = Counter()
error_list = []
for idx in err_indices:
    y_t = y_test[idx]
    y_p = preds_80[idx]
    v_t = VECS[y_t]
    v_p = VECS[y_p]
    depth_t = np.sum(v_t)
    depth_p = np.sum(v_p)

    name_t = NAMES[y_t]
    name_p = NAMES[y_p]

    # Category
    if depth_t == 1 and depth_p == 1:
        cat = "Singleton Confusion"
    elif np.all(v_t >= v_p) and depth_t == depth_p + 1:
        cat = "Under-prediction (Omission of 1 primitive)"
    elif np.all(v_p >= v_t) and depth_p == depth_t + 1:
        cat = "Over-prediction (Hallucination of 1 primitive)"
    elif depth_t > 1 and depth_p > 1:
        cat = "Compound-Compound Cross Confusion"
    else:
        cat = "Other Misclassification"

    tax[cat] += 1
    error_list.append((idx, depth_t, depth_p, name_t, name_p, cat))

for cat, count in tax.most_common():
    print(f"  {cat:45s}: {count:2d} ({count/len(err_indices)*100:.1f}%)")

print("\n" + "=" * 80)
print("TOP 15 FREQUENT ERROR PAIRS (True -> Pred):")
print("=" * 80)
pair_counts = Counter((item[3], item[4]) for item in error_list)
for (nt, np_name), c in pair_counts.most_common(15):
    print(f"  {nt:25s} -> {np_name:25s} : {c:2d} wafers")
