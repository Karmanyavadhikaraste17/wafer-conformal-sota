import sys
sys.path.insert(0, ".")
import numpy as np
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

DATA_PATH = "conformal_wafer/data/Wafer_Map_Datasets.npz"
raw = np.load(DATA_PATH, allow_pickle=True)
images = raw['arr_0']
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

boundary_samples = [1194, 1508, 2977, 4169, 6399]

print("=" * 80)
print(f"{'Idx':<6} | {'True Label':<16} | {'Total Perimeter Dies':<22} | {'Perimeter Density':<18} | {'Angular Concentration (Std/Mean)'}")
print("=" * 80)

ys, xs = np.ogrid[:52, :52]
center = (25.5, 25.5)
radius = np.sqrt((xs - center[0])**2 + (ys - center[1])**2) / 25.5
angle = np.arctan2(ys - center[1], xs - center[0]) # [-pi, pi]

for idx in boundary_samples:
    img = test_images[idx]
    true_lbl = CLASS_DEFINITIONS[y_test[idx]][1]
    
    perimeter_mask = (radius >= 0.82) & (radius <= 1.02)
    perimeter_dies = (img >= 1) & perimeter_mask
    perimeter_defects = (img >= 2) & perimeter_mask
    
    total_p_dies = np.sum(perimeter_dies)
    total_p_defects = np.sum(perimeter_defects)
    density = total_p_defects / max(total_p_dies, 1)
    
    # Angular distribution in 12 sectors of 30 degrees
    bins = np.linspace(-np.pi, np.pi, 13)
    p_counts, _ = np.histogram(angle[perimeter_defects], bins=bins)
    p_total, _ = np.histogram(angle[perimeter_dies], bins=bins)
    sector_densities = p_counts / np.maximum(p_total, 1)
    
    std_mean_ratio = np.std(sector_densities) / (np.mean(sector_densities) + 1e-6)
    
    print(f"{idx:<6} | {true_lbl:<16} | {total_p_defects}/{total_p_dies} dies ({density:.2f}) | {std_mean_ratio:.3f} | {np.round(sector_densities, 2).tolist()}")
