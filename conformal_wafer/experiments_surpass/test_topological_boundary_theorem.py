import sys
sys.path.insert(0, ".")
import numpy as np
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

# Load raw wafer data
DATA_PATH = "conformal_wafer/data/Wafer_Map_Datasets.npz"
raw = np.load(DATA_PATH, allow_pickle=True)
images = raw['arr_0'] # (38015, 52, 52)
y_vecs = raw['arr_1']

vec_map = {tuple(vec): idx for idx, (cid, name, vec, depth) in enumerate(CLASS_DEFINITIONS)}
y = np.zeros(len(y_vecs), dtype=np.int64)
for i in range(len(y_vecs)):
    y[i] = vec_map[tuple(y_vecs[i].tolist())]

from sklearn.model_selection import train_test_split
n_total = len(images)
train_full_idx, test_idx = train_test_split(np.arange(n_total), test_size=0.20, stratify=y, random_state=42)

test_images = images[test_idx]
y_test = y[test_idx]

# Recoverable indices from diagnose_78_errors.py:
# 1194: D+EL+S -> D+ER+S
# 1508: D+EL+L -> D+ER+L
# 2977: ER+L+S -> EL+L+S
# 4169: D+EL+S -> D+ER+S
# 6399: Edge_Loc -> Edge_Ring

boundary_samples = [1194, 1508, 2977, 4169, 6399]

print("=" * 70)
print(f"{'Idx':<6} | {'True Label':<16} | {'Edge Dies (r>0.85)':<20} | {'Max Arc Run (deg)':<18}")
print("=" * 70)

# Coordinate grid
ys, xs = np.ogrid[:52, :52]
center = (25.5, 25.5)
radius = np.sqrt((xs - center[0])**2 + (ys - center[1])**2) / 25.5
angle = np.arctan2(ys - center[1], xs - center[0]) # [-pi, pi]

for idx in boundary_samples:
    img = test_images[idx]
    true_lbl = CLASS_DEFINITIONS[y_test[idx]][1]
    
    # Defect dies on perimeter (r between 0.80 and 1.00)
    edge_defects = (img >= 2) & (radius >= 0.80) & (radius <= 1.05)
    n_edge_defects = np.sum(edge_defects)
    
    # Calculate angular distribution: bin angles into 36 bins of 10 degrees
    edge_angles = angle[edge_defects]
    bins = np.linspace(-np.pi, np.pi, 37)
    counts, _ = np.histogram(edge_angles, bins=bins)
    occupied_bins = (counts > 0).astype(int)
    
    # Calculate maximum contiguous run of occupied bins
    double_bins = np.concatenate([occupied_bins, occupied_bins])
    max_run = 0
    cur_run = 0
    for b in double_bins:
        if b == 1:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    max_run = min(max_run, 36)
    max_arc_deg = max_run * 10
    
    print(f"{idx:<6} | {true_lbl:<16} | {n_edge_defects:<20} | {max_arc_deg} deg / 360 deg")
