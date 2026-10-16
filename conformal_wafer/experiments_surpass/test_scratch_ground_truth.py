"""
test_scratch_ground_truth.py
Scientifically compare raw 52x52 image features of:
1. True Scratch Wafers (1181, 1522, 4703, 5190, 6063)
2. False Scratch Wafers (63, 277, 294, 405, 413, 492)
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders

loaders = get_fast_wafer_loaders(batch_size=128)
test_ds = loaders['test'].dataset
y_test = loaders['y_test']

true_scratch_wafers = [1181, 1522, 4703, 5190, 6063]
false_scratch_wafers = [63, 277, 294, 405, 413, 492]

# Measure length, continuity, and linearity of defect clusters on these wafers!
# How to measure true scratch linearity?
# Using Hough Transform, Radon Transform, or Connected Component aspect ratio!

from scipy.ndimage import label

def analyze_scratch_physics(idx):
    x, y, _ = test_ds[idx]
    def_map = (x[2].numpy() > 0.5).astype(np.uint8) # binary defect plane (52, 52)
    
    # 1. Connected components
    lbl, num_features = label(def_map)
    max_component_size = 0
    max_eccentricity = 0.0
    
    # Analyze each connected component
    for f in range(1, num_features + 1):
        coords = np.argwhere(lbl == f)
        sz = len(coords)
        if sz > max_component_size:
            max_component_size = sz
        if sz >= 5:
            # compute covariance matrix of coordinates to measure elongation / linearity
            cov = np.cov(coords, rowvar=False)
            evals = np.linalg.eigvalsh(cov)
            if evals.min() > 0:
                aspect = np.sqrt(evals.max() / evals.min())
            else:
                aspect = 10.0
            if aspect > max_eccentricity:
                max_eccentricity = aspect
                
    # 2. Steerable line response with length 13
    k_len = 13
    center = k_len // 2
    weights = np.zeros((8, 1, k_len, k_len), dtype=np.float32)
    for i, angle_deg in enumerate(np.linspace(0, 180, 8, endpoint=False)):
        theta = np.deg2rad(angle_deg)
        for r in range(-center, center + 1):
            yy = int(round(center + r * np.sin(theta)))
            xx = int(round(center + r * np.cos(theta)))
            if 0 <= yy < k_len and 0 <= xx < k_len:
                weights[i, 0, yy, xx] = 1.0
                
    line_k = torch.tensor(weights) # (8, 1, 13, 13)
    t_def = torch.from_numpy(def_map.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resp = F.conv2d(t_def, line_k, padding=center).numpy() # (1, 8, 52, 52)
    max_line_13 = resp.max()
    
    # Orientation variance of max line
    peak_per_orient = resp[0].reshape(8, -1).max(axis=-1)
    aniso_13 = peak_per_orient.max() - np.median(peak_per_orient)
    
    return max_component_size, max_eccentricity, max_line_13, aniso_13

print("--- TRUE SCRATCH WAFERS (Ground truth has S, mean pred missed it) ---")
for idx in true_scratch_wafers:
    sz, ecc, l13, aniso = analyze_scratch_physics(idx)
    t_name = CLASS_DEFINITIONS[y_test[idx]][1]
    print(f"Wafer {idx:4d} ({t_name:10s}) | MaxComp: {sz:3d} | Eccentricity: {ecc:5.2f} | MaxLine13: {l13:4.1f} | Aniso13: {aniso:4.1f}")

print("\n--- FALSE SCRATCH WAFERS (Ground truth lacks S, a view hallucinated it) ---")
for idx in false_scratch_wafers:
    sz, ecc, l13, aniso = analyze_scratch_physics(idx)
    t_name = CLASS_DEFINITIONS[y_test[idx]][1]
    print(f"Wafer {idx:4d} ({t_name:10s}) | MaxComp: {sz:3d} | Eccentricity: {ecc:5.2f} | MaxLine13: {l13:4.1f} | Aniso13: {aniso:4.1f}")
