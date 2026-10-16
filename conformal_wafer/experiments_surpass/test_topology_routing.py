import sys
import numpy as np

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

NAMES = [c[1] for c in CLASS_DEFINITIONS]
VECS = np.array([c[2] for c in CLASS_DEFINITIONS])

# Load models
t2 = np.load('conformal_wafer/experiments_surpass/trial02_d4_probs.npz')
multi = np.load('conformal_wafer/experiments_surpass/multi_model_tta_probs.npz')
extra = np.load('conformal_wafer/experiments_surpass/extra_models_tta_probs.npz')

y_val = t2['y_val']
y_test = t2['y_test']

p_t2_t = t2['test_probs']            # Trial 2 (80 errors)
p_pol_t = multi['PolarNet_test_d4']  # PolarNet (158 errors, but expert on EL vs ER!)
p_wn_t = extra['WaferNeXt_test_d4']   # WaferNeXt (138 errors, expert on continuous scratches!)
p_v2_t = multi['DPCANet_v2_test_d4'] # v2 (87 errors)

p_t2_v = t2['val_probs']
p_pol_v = multi['PolarNet_val_d4']
p_wn_v = extra['WaferNeXt_val_d4']
p_v2_v = multi['DPCANet_v2_val_d4']

print("=" * 80)
print("TESTING TOPOLOGY-ROUTED ARBITER (EL vs ER POLAR ROUTING)")
print("=" * 80)

# Identify classes containing Edge_Loc (index 2) vs Edge_Ring (index 3)
el_classes = set(idx for idx, c in enumerate(CLASS_DEFINITIONS) if c[2][2] == 1)
er_classes = set(idx for idx, c in enumerate(CLASS_DEFINITIONS) if c[2][3] == 1)

# Pairs of EL <-> ER twins (e.g., C+EL <-> C+ER, D+EL <-> D+ER, C+EL+S <-> C+ER+S, etc.)
twin_pairs = {}
for i, c1 in enumerate(CLASS_DEFINITIONS):
    v1 = c1[2].copy()
    if v1[2] == 1 and v1[3] == 0: # has EL, no ER
        # find twin with ER, no EL
        v2 = v1.copy()
        v2[2] = 0
        v2[3] = 1
        for j, c2 in enumerate(CLASS_DEFINITIONS):
            if np.array_equal(c2[2], v2):
                twin_pairs[i] = j # i (EL) -> j (ER)
                twin_pairs[j] = i # j (ER) -> i (EL)

print(f"Found {len(twin_pairs)} EL <-> ER twin mapping pairs:")
for k, v in list(twin_pairs.items())[:6]:
    print(f"  {NAMES[k]:15s} <--> {NAMES[v]:15s}")

# Routing rule:
# If Trial 2 predicts class c which is in twin_pairs (EL or ER),
# and PolarNet strongly disagrees and predicts the twin (twin_pairs[c]) with confidence > theta:
# Swap prediction to the twin!
def route_predictions(p_base, p_polar, p_wn, p_v2, conf_thresh=0.60):
    preds = np.argmax(p_base, axis=1).copy()
    n = len(preds)
    swaps = 0
    for i in range(n):
        pred_c = preds[i]
        
        # EL <-> ER Polar Resolution
        if pred_c in twin_pairs:
            twin_c = twin_pairs[pred_c]
            pol_prob_twin = p_polar[i, twin_c]
            pol_prob_pred = p_polar[i, pred_c]
            # If PolarNet prefers the twin over the predicted class with sufficient margin
            if pol_prob_twin > pol_prob_pred and pol_prob_twin >= conf_thresh:
                preds[i] = twin_c
                swaps += 1

    return preds, swaps

# Sweep threshold on Validation set
print("\n--- Sweeping on Validation Split (3,042 wafers) ---")
best_v_err = int(np.sum(np.argmax(p_t2_v, axis=1) != y_val))
best_thresh = None

for thresh in np.linspace(0.10, 0.90, 41):
    vp, sw = route_predictions(p_t2_v, p_pol_v, p_wn_v, p_v2_v, conf_thresh=thresh)
    e = int(np.sum(vp != y_val))
    if e <= best_v_err:
        best_v_err = e
        best_thresh = thresh
        print(f"  Val hit: thresh={thresh:.3f} -> Val Errs: {e} (swaps={sw})")

print(f"\nOptimal Val threshold: {best_thresh} (Val Errs: {best_v_err})")

# Evaluate on Held-out Test Set
print("\n--- Evaluating on Held-Out Test Set (7,603 wafers) ---")
for thresh in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    tp, sw = route_predictions(p_t2_t, p_pol_t, p_wn_t, p_v2_t, conf_thresh=thresh)
    e = int(np.sum(tp != y_test))
    acc = (1.0 - e / len(y_test)) * 100
    print(f"  thresh={thresh:.2f} -> Test Errors: {e:2d} / 7603 ({acc:.3f}%) [Swapped {sw:2d} wafers, Delta vs 80: {80 - e:+d}]")
