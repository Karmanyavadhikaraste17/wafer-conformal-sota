import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.55)

from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.experiments_surpass.autonomous_overnight_surpass_loop_part4 import CylindriWaferNet
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

loaders = get_fast_wafer_loaders(batch_size=64)
y_test = loaders['y_test']

# Evaluate CylindriWaferNet (Trial 18)
base_m = DPCANetV3(num_classes=38).to(DEVICE)
cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
model18 = CylindriWaferNet(cart_model).to(DEVICE)
c18 = torch.load("conformal_wafer/experiments_surpass/cylindriwafer_trial18_best.pt", map_location=DEVICE, weights_only=False)
model18.load_state_dict(c18['model_state_dict'])
model18.eval()

def get_d4_probs(model):
    all_p = []
    with torch.no_grad():
        for b_idx, (x, y, _) in enumerate(loaders['test']):
            x = x.to(DEVICE)
            B = len(x)
            p_sum = np.zeros((B, 38), dtype=np.float32)
            for k in range(4):
                xk = torch.rot90(x, k, [2, 3])
                for flip in [False, True]:
                    xv = torch.flip(xk, [3]) if flip else xk
                    with torch.amp.autocast('cuda', enabled=True):
                        out = model(xv)
                        l = out[0] if isinstance(out, tuple) else out
                    p_sum += F.softmax(l / 0.50, dim=1).cpu().numpy()
            all_p.append(p_sum / 8.0)
    return np.concatenate(all_p, axis=0)

print("Extracting D4-TTA probabilities for CylindriWaferNet (Trial 18)...")
p18 = get_d4_probs(model18)
err18 = int(np.sum(np.argmax(p18, axis=1) != y_test))
print(f"Trial 18 D4-TTA Errors: {err18}")

# Load previous probabilities
d_t2 = np.load("conformal_wafer/experiments_surpass/trial02_d4_probs.npz")['test_probs']
d_t3 = np.load("conformal_wafer/experiments_surpass/trial03_d4_probs.npz")['test_probs']
d_t4 = np.load("conformal_wafer/experiments_surpass/trial04_d4_probs.npz")['test_probs']
d_t5 = np.load("conformal_wafer/experiments_surpass/trial05_d4_probs.npz")['test_probs']
d_t6 = np.load("conformal_wafer/experiments_surpass/trial06_d4_probs.npz")['test_probs']
d_t7 = np.load("conformal_wafer/experiments_surpass/trial07_d4_probs.npz")['test_probs']
d_t8 = np.load("conformal_wafer/experiments_surpass/trial08_d4_probs.npz")['test_probs']

all_preds = np.stack([
    np.argmax(d_t2, axis=1),
    np.argmax(d_t3, axis=1),
    np.argmax(d_t4, axis=1),
    np.argmax(d_t5, axis=1),
    np.argmax(d_t6, axis=1),
    np.argmax(d_t7, axis=1),
    np.argmax(d_t8, axis=1),
    np.argmax(p18, axis=1),
], axis=1) # (7603, 8)

oracle_mask = np.any(all_preds == y_test[:, None], axis=1)
oracle_err = int(np.sum(~oracle_mask))
print(f"NEW ORACLE BOUND across 8 surpass models: {oracle_err} ERRORS ({100*(1-oracle_err/len(y_test)):.3f}%) [Delta vs SOTA: {66 - oracle_err:+d}]")

np.savez_compressed("conformal_wafer/experiments_surpass/trial18_d4_probs.npz", test_probs=p18)
