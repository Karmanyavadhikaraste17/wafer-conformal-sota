import sys
import numpy as np

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

data_m = np.load("conformal_wafer/experiments_surpass/multi_model_tta_probs.npz")
data_e = np.load("conformal_wafer/experiments_surpass/extra_models_tta_probs.npz")
data_t2 = np.load("conformal_wafer/experiments_surpass/trial02_d4_probs.npz")

y_val = data_m['y_val']
y_test = data_m['y_test']
NAMES = [c[1] for c in CLASS_DEFINITIONS]
VECS = np.array([c[2] for c in CLASS_DEFINITIONS]) # 8 primitives: Center(0), Donut(1), Edge_Loc(2), Edge_Ring(3), Loc(4), Near_Full(5), Scratch(6), Random(7)

EL_IDX = 2
ER_IDX = 3

v_t2 = data_t2['val_probs']
t_t2 = data_t2['test_probs']

v_polar = data_m['PolarNet_val_d4']
t_polar = data_m['PolarNet_test_d4']

v_wnext = data_e['WaferNeXt_val_d4']
t_wnext = data_e['WaferNeXt_test_d4']

v_v2 = data_m['DPCANet_v2_val_d4']
t_v2 = data_m['DPCANet_v2_test_d4']

# Let's inspect on validation set first:
pred_val_t2 = np.argmax(v_t2, axis=1)
pred_val_polar = np.argmax(v_polar, axis=1)
pred_val_wnext = np.argmax(v_wnext, axis=1)

print(f"Validation Errors: Trial2={np.sum(pred_val_t2 != y_val)}, Polar={np.sum(pred_val_polar != y_val)}")

# Let's see: When Trial 2 and Polar disagree, what happens on Validation?
val_disagree = np.where(pred_val_t2 != pred_val_polar)[0]
print(f"Validation disagreements: {len(val_disagree)}")

# Specifically, where Trial2 predicts EL/ER and Polar predicts the other:
el_er_val_cases = []
for idx in val_disagree:
    c_t2 = pred_val_t2[idx]
    c_pol = pred_val_polar[idx]
    vec_t2 = VECS[c_t2]
    vec_pol = VECS[c_pol]
    
    # Check if they differ ONLY by EL vs ER (or have EL vs ER difference)
    if (vec_t2[EL_IDX] != vec_pol[EL_IDX]) or (vec_t2[ER_IDX] != vec_pol[ER_IDX]):
        el_er_val_cases.append((idx, y_val[idx], c_t2, c_pol))

print(f"Validation EL/ER conflict cases: {len(el_er_val_cases)}")
for idx, yt, c_t2, c_pol in el_er_val_cases:
    print(f"Val Wafer {idx:4d} | True: {NAMES[yt]:15s} | Trial2: {NAMES[c_t2]:15s} | Polar: {NAMES[c_pol]:15s} | Polar Conf: {v_polar[idx, c_pol]:.3f} | T2 Conf: {v_t2[idx, c_t2]:.3f}")
