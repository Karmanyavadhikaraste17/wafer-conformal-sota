"""
test_steerable_scratch_metric.py
Evaluate exact steerable line anisotropy and scratch detection across all 7,603 test wafers.
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, ".")

from conformal_wafer.src.dataset import CLASS_DEFINITIONS
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders

loaders = get_fast_wafer_loaders(batch_size=128)
y_test = loaders['y_test']

# 8 oriented 1D line kernels (length 9)
num_angles = 8
kernel_size = 9
weights = np.zeros((num_angles, 1, kernel_size, kernel_size), dtype=np.float32)
center = kernel_size // 2
for i, angle_deg in enumerate(np.linspace(0, 180, num_angles, endpoint=False)):
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    for r in range(-center, center + 1):
        y = int(round(center + r * sin_t))
        x = int(round(center + r * cos_t))
        if 0 <= y < kernel_size and 0 <= x < kernel_size:
            weights[i, 0, y, x] = 1.0
    s = np.sum(weights[i, 0])
    if s > 0:
        weights[i, 0] /= s

line_kernels = torch.tensor(weights, dtype=torch.float32)

# Check scratch presence in ground truth
has_scratch = np.array(['S' in CLASS_DEFINITIONS[y][1].split('+') or CLASS_DEFINITIONS[y][1] == 'Scratch' for y in y_test])
print(f"Total test wafers with Scratch: {has_scratch.sum()} / 7603")

# Compute steerable line responses on test wafers
all_scratch_scores = []
with torch.no_grad():
    for x, _, _ in loaders['test']:
        # x[:, 2:3] is binary defect channel
        p_def = x[:, 2:3] # (B, 1, 52, 52)
        resp = F.conv2d(p_def, line_kernels, padding=center) # (B, 8, 52, 52)
        
        # Max response over space for each orientation
        max_per_orient, _ = torch.max(resp.flatten(2), dim=-1) # (B, 8)
        
        # Max over all orientations
        peak_line = torch.max(max_per_orient, dim=-1)[0] # (B,)
        
        # Anisotropy: peak orientation minus median orientation
        median_line = torch.median(max_per_orient, dim=-1)[0] # (B,)
        anisotropy = peak_line - median_line # (B,)
        
        # Metric combining peak and anisotropy
        score = peak_line * anisotropy
        all_scratch_scores.append(score.numpy())

all_scratch_scores = np.concatenate(all_scratch_scores)

# Measure ROC-AUC or separability of Scratch presence
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(has_scratch, all_scratch_scores)
print(f"Steerable Scratch Score ROC-AUC on 7,603 test wafers: {auc:.4f}")

# Threshold search
for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    pred_scratch = (all_scratch_scores > thresh)
    precision = np.sum(pred_scratch & has_scratch) / (np.sum(pred_scratch) + 1e-9)
    recall = np.sum(pred_scratch & has_scratch) / np.sum(has_scratch)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    print(f"Thresh: {thresh:.2f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
