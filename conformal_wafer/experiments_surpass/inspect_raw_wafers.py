"""
inspect_raw_wafers.py
Inspect the raw defect pixels of the 18 gap wafers.
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

# The 18 gap indices
gap_indices = [1181, 1194, 1327, 1378, 1508, 1522, 2977, 3123, 3606, 3634, 4169, 4703, 5190, 5713, 5832, 6063, 6399, 6576]

print(f"{'Idx':<5} | {'True Name':<12} | {'True Vec':<25} | {'Total Defects':<13} | {'Max Line 9':<10} | {'Perimeter Defects':<17}")
print("-" * 90)

for idx in gap_indices:
    x, y, y_prim = test_ds[idx]
    # x is (3, 52, 52)
    def_map = x[2].numpy() # (52, 52) binary defect map
    total_def = int(def_map.sum())
    
    # Check max line of length 9 in any direction
    t_def = torch.from_numpy(def_map).unsqueeze(0).unsqueeze(0) # (1, 1, 52, 52)
    
    # 8 line kernels
    num_angles = 8
    kernel_size = 9
    center = kernel_size // 2
    weights = np.zeros((num_angles, 1, kernel_size, kernel_size), dtype=np.float32)
    for i, angle_deg in enumerate(np.linspace(0, 180, num_angles, endpoint=False)):
        theta = np.deg2rad(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        for r in range(-center, center + 1):
            yy = int(round(center + r * sin_t))
            xx = int(round(center + r * cos_t))
            if 0 <= yy < kernel_size and 0 <= xx < kernel_size:
                weights[i, 0, yy, xx] = 1.0
    line_k = torch.tensor(weights) # unnormalized count of defects along line of 9
    line_resp = F.conv2d(t_def, line_k, padding=center)
    max_line_count = int(line_resp.max().item())
    
    # Perimeter defects (radius between 20 and 26)
    yy, xx = np.ogrid[:52, :52]
    dist_from_center = np.sqrt((yy - 25.5)**2 + (xx - 25.5)**2)
    perim_mask = (dist_from_center >= 20.0) & (dist_from_center <= 25.0)
    perim_def = int((def_map * perim_mask).sum())
    
    true_name = CLASS_DEFINITIONS[y][1]
    true_vec = str(y_prim.numpy().astype(int).tolist())
    print(f"{idx:<5d} | {true_name:<12s} | {true_vec:<25s} | {total_def:<13d} | {max_line_count:<10d} | {perim_def:<17d}")
