"""
extract_trial33_all_views.py

Extract 8 dihedral view logits for Train, Val, and Test splits from Trial 33 (SWA DC-PolarSCANet).
Saves to conformal_wafer/experiments_surpass/trial33_all_views.npz
"""

import sys
sys.path.insert(0, ".")
import os
import time
import torch
import numpy as np

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_surpass.autonomous_overnight_surpass_loop_part12 import DCPolarSCANet
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.src.train_optimized_sota import PolarNet

print("Loading Fast Wafer Loaders...")
loaders = get_fast_wafer_loaders(batch_size=128)

print("Instantiating DCPolarSCANet architecture...")
base_m = DPCANetV3(num_classes=38).to(DEVICE)
cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
polar_model = PolarNet(num_classes=38, S=32, P=64).to(DEVICE)
model = DCPolarSCANet(cart_model, polar_model).to(DEVICE)

ckpt_path = "conformal_wafer/experiments_surpass/swa_dc_polarscanet_trial33_best.pt"
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Loaded {ckpt_path} successfully!")

def extract_split(loader_key):
    t0 = time.time()
    view_preds = [[] for _ in range(8)]
    with torch.no_grad():
        for x, _, _ in loaders[loader_key]:
            x = x.to(DEVICE)
            v_idx = 0
            for k in range(4):
                xk = torch.rot90(x, k, [2, 3])
                for flip in [False, True]:
                    xv = torch.flip(xk, [3]) if flip else xk
                    with torch.amp.autocast('cuda'):
                        out = model(xv)
                        logits = out[0] if isinstance(out, tuple) else out
                    view_preds[v_idx].append(logits.cpu().numpy())
                    v_idx += 1
    # Shape: (N, 8, 38)
    split_views = np.stack([np.concatenate(v, axis=0) for v in view_preds], axis=1)
    dt = time.time() - t0
    print(f"Extracted {loader_key} ({len(split_views)} samples) in {dt:.1f}s, shape: {split_views.shape}")
    return split_views

print("\nExtracting Test split views...")
test_views = extract_split('test')

print("\nExtracting Val split views...")
val_views = extract_split('val')

print("\nExtracting Train split views...")
train_views = extract_split('train')

save_path = "conformal_wafer/experiments_surpass/trial33_all_views.npz"
np.savez_compressed(
    save_path,
    train_views=train_views,
    val_views=val_views,
    test_views=test_views,
    y_train=loaders['y_train'],
    y_val=loaders['y_val'],
    y_test=loaders['y_test']
)
print(f"\nSaved all splits to {save_path} successfully!")
