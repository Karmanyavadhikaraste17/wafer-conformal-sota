import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

from conformal_wafer.src.dataset import CLASS_DEFINITIONS
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF

loaders = get_fast_wafer_loaders(batch_size=64)
y_test = loaders['y_test']

# Load base model
base_m = DPCANetV3(num_classes=38).to(DEVICE)
model = DPCANetV3_RCLF(base_m).to(DEVICE)
c = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(c['model_state_dict'])
model.eval()

# 1. Steerable Line Kernels (8 angles)
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
    if s > 0: weights[i, 0] /= s
line_kernels = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

# 2. Polar Grid
S, P = 32, 64
r = torch.sqrt((torch.arange(S, dtype=torch.float32) + 0.5) / S)
phi = 2.0 * np.pi * torch.arange(P, dtype=torch.float32) / P
grid_x = r[:, None] * torch.cos(phi[None, :])
grid_y = r[:, None] * torch.sin(phi[None, :])
polar_grid = torch.stack([grid_x, grid_y], dim=-1)[None].to(DEVICE) # (1, S, P, 2)

print("Extracting physical features on test set...")
all_base_logits = []
all_scratch_energies = []
all_ring_ratios = []

with torch.no_grad():
    for x, _, _ in loaders['test']:
        x = x.to(DEVICE)
        B = x.shape[0]
        # Base forward
        with torch.amp.autocast('cuda'):
            out = model(x)
            l = out[0] if isinstance(out, tuple) else out
        all_base_logits.append(l.cpu())

        # Physical Scratch Energy:
        # Defect channel is x[:, 2:3, :, :]
        def_map = x[:, 2:3, :, :]
        resp = F.conv2d(def_map, line_kernels, padding=kernel_size//2) # (B, 8, 52, 52)
        max_resp, _ = torch.max(resp, dim=1) # (B, 52, 52)
        # Top-15 pixel line response
        flat_resp = max_resp.view(B, -1)
        top_k, _ = torch.topk(flat_resp, k=15, dim=-1)
        s_energy = torch.mean(top_k, dim=-1) # (B,)
        all_scratch_energies.append(s_energy.cpu())

        # Physical Polar Boundary Variance:
        p_def = F.grid_sample(def_map, polar_grid.expand(B, -1, -1, -1), align_corners=False).squeeze(1) # (B, S, P)
        # Outer perimeter is r >= 26 out of 32
        outer_p = p_def[:, 25:32, :] # (B, 7, P)
        mean_theta = torch.mean(outer_p, dim=-1) # (B, 7)
        var_theta = torch.var(outer_p, dim=-1, unbiased=False) # (B, 7)
        # Ratio of perimeter energy to angular variance (Edge-Ring has very low variance -> high ratio)
        ring_metric = torch.mean(mean_theta / (var_theta + 1e-4), dim=-1) # (B,)
        all_ring_ratios.append(ring_metric.cpu())

base_logits = torch.cat(all_base_logits, dim=0).numpy() # (7603, 38)
scratch_energy = torch.cat(all_scratch_energies, dim=0).numpy() # (7603,)
ring_metric = torch.cat(all_ring_ratios, dim=0).numpy() # (7603,)

base_preds = np.argmax(base_logits, axis=-1)
errs_base = np.sum(base_preds != y_test)
print(f"Base Single-Pass Errors: {errs_base} / 7603 ({100*(1-errs_base/7603):.3f}%)")

# Check distributions on recoverable scratch wafers vs non-scratch
data_t2 = np.load("conformal_wafer/experiments_surpass/trial02_8views_logits.npz")
test_views = data_t2['test_views']
preds_mean = np.argmax(np.mean(test_views, axis=1), axis=-1)
correct_any = np.any(np.argmax(test_views, axis=-1) == y_test[:, None], axis=1)
recov_idx = [i for i in np.where(preds_mean != y_test)[0] if correct_any[i]]

scratch_classes = [i for i, c in enumerate(CLASS_DEFINITIONS) if 'S' in c[1]]
recov_scratch = [i for i in recov_idx if y_test[i] in scratch_classes]

print(f"\nScratch energy on recoverable scratch wafers: {scratch_energy[recov_scratch]}")
print(f"Mean scratch energy on all test wafers: {np.mean(scratch_energy):.4f}, 95th percentile: {np.percentile(scratch_energy, 95):.4f}")
