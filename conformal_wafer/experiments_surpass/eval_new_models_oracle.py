import os
import sys
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from conformal_wafer.src.dataset import CLASS_DEFINITIONS
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_surpass.autonomous_overnight_surpass_loop_part8 import PQSCANet, GaborLineHighwayNet, GrandMasterNet
from conformal_wafer.experiments_surpass.autonomous_overnight_surpass_loop_part9 import PICGNet
from conformal_wafer.experiments_surpass.autonomous_overnight_surpass_loop_part7 import HiResPeakNet, CylindriHiResNet
from conformal_wafer.experiments_surpass.autonomous_overnight_surpass_loop_part10 import HMoENet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)
    torch.backends.cudnn.benchmark = False
loaders = get_fast_wafer_loaders(batch_size=64)
test_loader = loaders['test']
y_test = loaders['y_test']

def get_d4_probs(model):
    model.eval()
    all_p = []
    with torch.no_grad():
        for x, _, _ in test_loader:
            x = x.to(DEVICE)
            B = len(x)
            p_sum = np.zeros((B, 38), dtype=np.float32)
            for k in range(4):
                xk = torch.rot90(x, k, [2, 3])
                for flip in [False, True]:
                    xv = torch.flip(xk, [3]) if flip else xk
                    with autocast('cuda', enabled=True):
                        out = model(xv)
                        l = out[0] if isinstance(out, tuple) else out
                    p_sum += F.softmax(l / 0.50, dim=1).cpu().numpy()
            all_p.append(p_sum / 8.0)
    return np.concatenate(all_p, axis=0)

base_m = DPCANetV3(num_classes=38).to(DEVICE)
cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
cart_model.load_state_dict(c_cart['model_state_dict'])

new_probs = {}

models_to_eval = [
    ("T25_PQSCANet", PQSCANet, "conformal_wafer/experiments_surpass/pq_scanet_trial25_best.pt"),
    ("T28_PICGNet", PICGNet, "conformal_wafer/experiments_surpass/picg_trial28_best.pt"),
    ("T22_HiResPeak", HiResPeakNet, "conformal_wafer/experiments_surpass/hires_peaknet_trial22_best.pt"),
    ("T27_DistillGrand", GrandMasterNet, "conformal_wafer/experiments_surpass/distill_grandmaster_trial27_best.pt"),
    ("T29_HMoENet", HMoENet, "conformal_wafer/experiments_surpass/hmoenet_trial29_best.pt"),
    ("T24_GaborHwy", GaborLineHighwayNet, "conformal_wafer/experiments_surpass/gabor_highway_trial24_best.pt"),
]

for name, cls, path in models_to_eval:
    if os.path.exists(path):
        print(f"Extracting D4 TTA probs for {name}...")
        torch.cuda.empty_cache()
        m = cls(cart_model).to(DEVICE)
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        m.load_state_dict(ckpt['model_state_dict'])
        p = get_d4_probs(m)
        err = int(np.sum(np.argmax(p, axis=1) != y_test))
        print(f"  {name} -> {err} errors ({(1-err/7603)*100:.3f}%)")
        new_probs[name] = p
        del m
        torch.cuda.empty_cache()

np.savez_compressed("conformal_wafer/experiments_surpass/new_trials_tta_probs.npz", **new_probs)
print("Saved new trials TTA probabilities successfully!")
