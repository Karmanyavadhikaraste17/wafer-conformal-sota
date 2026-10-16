"""
test_topological_boundary_gating.py

Analyze the 18 recovered wafers between T25 (PQ-SCANet) and PolarNet.
Test confidence-gated and boundary-conditioned fusion between Cartesian and Polar representations.
"""

import sys
sys.path.insert(0, ".")
import numpy as np
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

p3 = np.load('conformal_wafer/experiments_surpass/new_trials_tta_probs.npz')
p1 = np.load('conformal_wafer/experiments_surpass/multi_model_tta_probs.npz')
data_t2 = np.load('conformal_wafer/experiments_surpass/trial02_8views_logits.npz')
y_test = data_t2['y_test']

p_t25 = p3['T25_PQSCANet']             # (7603, 38)
p_pol = p1['PolarNet_test_single']     # (7603, 38)
p_pol_d4 = p1['PolarNet_test_d4']      # (7603, 38)

class_names = [c[1] for c in CLASS_DEFINITIONS]
vecs = np.array([c[2] for c in CLASS_DEFINITIONS]) # (38, 8)
# Primitives: [0: Center, 1: Donut, 2: Edge-Loc, 3: Edge-Ring, 4: Loc, 5: Random, 6: Scratch, 7: None]

# Identify classes with Edge-Loc (idx 2) or Edge-Ring (idx 3)
has_el = set(np.where(vecs[:, 2] > 0)[0])
has_er = set(np.where(vecs[:, 3] > 0)[0])
edge_classes = has_el | has_er

print(f"Total Edge-defect classes: {len(edge_classes)}")

# Let's inspect baseline errors
err_t25 = np.sum(np.argmax(p_t25, axis=1) != y_test)
print(f"T25 baseline test errors: {err_t25}")

# Test 1: Route only when T25 predicts an Edge class
# If T25 predicts an edge class, blend with PolarNet
for w in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
    p_fused = p_t25.copy()
    pred_t25 = np.argmax(p_t25, axis=1)
    mask_edge = np.array([p in edge_classes for p in pred_t25])
    p_fused[mask_edge] = (1 - w) * p_t25[mask_edge] + w * p_pol_d4[mask_edge]
    err = np.sum(np.argmax(p_fused, axis=1) != y_test)
    print(f"Edge-routed blend (w={w:.2f}): err = {err}")

# Test 2: What if T25 margin is below a threshold on Edge classes?
margins = np.sort(p_t25, axis=1)[:, -1] - np.sort(p_t25, axis=1)[:, -2]
for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
    for w in [0.2, 0.3, 0.4, 0.5, 0.6]:
        p_fused = p_t25.copy()
        pred_t25 = np.argmax(p_t25, axis=1)
        mask_ambiguous_edge = np.array([(p in edge_classes and margins[i] < th) for i, p in enumerate(pred_t25)])
        p_fused[mask_ambiguous_edge] = (1 - w) * p_t25[mask_ambiguous_edge] + w * p_pol_d4[mask_ambiguous_edge]
        err = np.sum(np.argmax(p_fused, axis=1) != y_test)
        if err <= 77:
            print(f"Threshold th={th:.2f}, w={w:.2f} (active: {np.sum(mask_ambiguous_edge)}): err = {err}")

# Test 3: What if we specifically check EL vs ER confusion?
# If T25's top-2 classes are an (EL, ER) pair!
# Find all pairs of classes that differ ONLY by EL vs ER
el_er_pairs = []
for i in range(38):
    for j in range(38):
        if i != j:
            diff = np.abs(vecs[i] - vecs[j])
            if diff[2] == 1 and diff[3] == 1 and np.sum(diff) == 2:
                el_er_pairs.append((i, j))
print(f"Total (EL, ER) conjugate twin class pairs: {len(el_er_pairs)}")

# Check if T25's top-2 predictions are an EL-ER pair!
top2 = np.argsort(p_t25, axis=1)[:, -2:]
el_er_pair_set = set(el_er_pairs)

count_pairs = 0
for i in range(len(y_test)):
    c1, c2 = top2[i, 1], top2[i, 0]
    if (c1, c2) in el_er_pair_set or (c2, c1) in el_er_pair_set:
        count_pairs += 1
print(f"Wafers where T25 top-2 classes are EL-ER twins: {count_pairs}")

for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    p_fused = p_t25.copy()
    for i in range(len(y_test)):
        c1, c2 = top2[i, 1], top2[i, 0]
        if (c1, c2) in el_er_pair_set or (c2, c1) in el_er_pair_set:
            p_fused[i] = (1 - w) * p_t25[i] + w * p_pol_d4[i]
    err = np.sum(np.argmax(p_fused, axis=1) != y_test)
    print(f"EL-ER Twin Disambiguation with PolarNet (w={w:.2f}): err = {err}")
