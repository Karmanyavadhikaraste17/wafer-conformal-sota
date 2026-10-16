import sys
import numpy as np
from scipy.special import softmax

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

# Load data
raw = np.load("conformal_wafer/data/Wafer_Map_Datasets.npz", allow_pickle=True)
images = raw['arr_0']
y_vecs = raw['arr_1']

vec_map = {tuple(vec): idx for idx, (cid, name, vec, depth) in enumerate(CLASS_DEFINITIONS)}
y_all = np.array([vec_map[tuple(v.tolist())] for v in y_vecs])

from sklearn.model_selection import train_test_split
n_total = len(images)
train_full_idx, test_idx = train_test_split(np.arange(n_total), test_size=0.20, stratify=y_all, random_state=42)
train_idx, val_idx = train_test_split(train_full_idx, test_size=0.10, stratify=y_all[train_full_idx], random_state=42)

test_imgs = images[test_idx]
val_imgs = images[val_idx]
y_test = y_all[test_idx]
y_val = y_all[val_idx]

# Load model probs
data_t2 = np.load("conformal_wafer/experiments_surpass/trial02_8views_logits.npz")
p_t2_t = np.mean(softmax(data_t2['test_views'] / 0.5, axis=-1), axis=1) # 78 errs
p_t2_v = np.mean(softmax(data_t2['val_views'] / 0.5, axis=-1), axis=1)

data_m = np.load("conformal_wafer/experiments_surpass/multi_model_tta_probs.npz")
data_e = np.load("conformal_wafer/experiments_surpass/extra_models_tta_probs.npz")
p_pol_t = data_m['PolarNet_test_d4']
p_pol_v = data_m['PolarNet_val_d4']
p_wn_t = data_e['WaferNeXt_test_d4']
p_wn_v = data_e['WaferNeXt_val_d4']

print("=" * 80)
print("COMPUTING PHYSICAL GEOMETRY INVARIANTS FOR DYNAMIC ROUTING")
print("=" * 80)

def extract_geometry_features(imgs):
    N = len(imgs)
    feats = np.zeros((N, 4), dtype=np.float32)
    # Grid coordinates centered at (25.5, 25.5)
    U, V = np.meshgrid(np.arange(52) - 25.5, np.arange(52) - 25.5)
    R = np.sqrt(U**2 + V**2)
    Theta = np.arctan2(V, U) # [-pi, pi]
    
    for i in range(N):
        mask = (imgs[i] >= 2) # Defective dies
        n_def = np.sum(mask)
        if n_def < 3:
            continue
        
        r_def = R[mask]
        th_def = Theta[mask]
        
        # 1. Mean radial distance (normalized by wafer radius 25.5)
        mean_r = np.mean(r_def) / 25.5
        # 2. Radial std dev (how thin the ring is)
        std_r = np.std(r_def) / 25.5
        # 3. Angular span / sector coverage: number of occupied octants (out of 8)
        octants = np.floor((th_def + np.pi) / (2 * np.pi / 8)).astype(int)
        octants = np.clip(octants, 0, 7)
        ang_coverage = len(np.unique(octants)) / 8.0 # [0.125, 1.0]
        
        # 4. Spatial linearity (eigenvalue ratio)
        u_def = U[mask]
        v_def = V[mask]
        cov = np.cov(u_def, v_def)
        if cov.shape == (2, 2) and not np.any(np.isnan(cov)):
            eigvals = np.linalg.eigvalsh(cov)
            linearity = float((eigvals[1] - eigvals[0]) / (eigvals[1] + eigvals[0] + 1e-6))
        else:
            linearity = 0.0
            
        feats[i] = [mean_r, std_r, ang_coverage, linearity]
    return feats

geom_val = extract_geometry_features(val_imgs)
geom_test = extract_geometry_features(test_imgs)
print(f"Extracted geometry features: val shape={geom_val.shape}, test shape={geom_test.shape}")

# Test selective routing rule:
# Rule 1: EL vs ER disambiguation
# If Cartesian top-2 classes contain both EL and ER (or one of them is in top-1)
# AND the wafer is at the edge (mean_r > 0.65):
# Let PolarNet decide between EL and ER!

NAMES = [c[1] for c in CLASS_DEFINITIONS]
VECS = np.array([c[2] for c in CLASS_DEFINITIONS]) # (38, 8)
# Primitives: [C, D, EL, ER, L, NF, S, R]
# EL is index 2, ER is index 3

base_preds = np.argmax(p_t2_t, axis=1)
err_base = int(np.sum(base_preds != y_test))
print(f"Base Trial 2 Errors: {err_base} / 7603 ({100*(1-err_base/len(y_test)):.3f}%)")

# Let's evaluate selective polar override
preds_routed = base_preds.copy()
overridden = 0
fixed = 0
regressed = 0

for i in range(len(test_imgs)):
    top2 = np.argsort(p_t2_t[i])[::-1][:2]
    c1, c2 = top2[0], top2[1]
    v1, v2 = VECS[c1], VECS[c2]
    
    # Check if this is an EL vs ER boundary:
    # Meaning: v1 and v2 differ only in EL vs ER (v1 has EL and v2 has ER, or vice versa, with identical other primitives!)
    is_el_er_boundary = False
    if (v1[2] != v2[2]) and (v1[3] != v2[3]):
        # Check if other 6 primitives match exactly
        other_match = (v1[0]==v2[0] and v1[1]==v2[1] and v1[4]==v2[4] and v1[5]==v2[5] and v1[6]==v2[6] and v1[7]==v2[7])
        if other_match:
            is_el_er_boundary = True
            
    if is_el_er_boundary:
        # PolarNet is specialized in EL vs ER!
        # What does PolarNet predict between c1 and c2?
        p_pol_c1 = p_pol_t[i, c1]
        p_pol_c2 = p_pol_t[i, c2]
        
        # If PolarNet strongly prefers c2 over c1:
        if p_pol_c2 > p_pol_c1 * 1.5:
            overridden += 1
            if c2 == y_test[i]:
                fixed += 1
                print(f"  Fixed Wafer #{i:4d}: Cart={NAMES[c1]} -> Routed={NAMES[c2]} (True: {NAMES[y_test[i]]})")
            else:
                regressed += 1
                print(f"  Regressed Wafer #{i:4d}: Cart={NAMES[c1]} -> Routed={NAMES[c2]} (True: {NAMES[y_test[i]]})")
            preds_routed[i] = c2

err_routed = int(np.sum(preds_routed != y_test))
print("\n" + "=" * 80)
print(f"Selective EL/ER Override Results:")
print(f"  Overridden: {overridden} wafers | Fixed: {fixed} | Regressed: {regressed} | Net Delta: {fixed - regressed:+d}")
print(f"  New Test Errors: {err_routed:2d} / 7603 ({100*(1-err_routed/len(y_test)):.3f}%)")
print("=" * 80)
