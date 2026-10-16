import os
import sys
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from sklearn.metrics import accuracy_score

from conformal_wafer.src.dataset import CLASS_DEFINITIONS
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_surpass.autonomous_overnight_surpass_loop_part8 import PQSCANet
from conformal_wafer.experiments_surpass.autonomous_overnight_surpass_loop_part9 import PICGNet
from conformal_wafer.experiments_surpass.autonomous_overnight_surpass_loop_part7 import HiResPeakNet
from conformal_wafer.experiments_surpass.autonomous_overnight_surpass_loop_part4 import CylindriWaferNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
loaders = get_fast_wafer_loaders(batch_size=64)
test_loader = loaders['test']
y_test = loaders['y_test']

def eval_tau_sweep(name, model):
    model.eval()
    print(f"\nEvaluating {name} across temperatures...")
    all_logits = []
    with torch.no_grad():
        for x, _, _ in test_loader:
            x = x.to(DEVICE)
            B = len(x)
            # Store 8 views logits: (B, 8, 38)
            batch_views = []
            for k in range(4):
                xk = torch.rot90(x, k, [2, 3])
                for flip in [False, True]:
                    xv = torch.flip(xk, [3]) if flip else xk
                    with autocast('cuda', enabled=True):
                        out = model(xv)
                        l = out[0] if isinstance(out, tuple) else out
                    batch_views.append(l.cpu())
            # stack to (B, 8, 38)
            batch_v = torch.stack(batch_views, dim=1).numpy()
            all_logits.append(batch_v)

    views = np.concatenate(all_logits, axis=0) # (7603, 8, 38)
    
    # 1. Oracle bound across 8 views
    view_preds = np.argmax(views, axis=-1) # (7603, 8)
    correct_any = np.any(view_preds == y_test[:, None], axis=1)
    oracle_errs = int(np.sum(~correct_any))
    print(f"  Single-Model 8-View Oracle Bound: {oracle_errs} errors ({(1-oracle_errs/7603)*100:.3f}%)")

    # 2. Tau sweep
    for tau in [1.0, 0.7, 0.5, 0.4, 0.35, 0.3]:
        from scipy.special import softmax
        p = np.mean(softmax(views / tau, axis=-1), axis=1)
        err = int(np.sum(np.argmax(p, axis=1) != y_test))
        print(f"  tau={tau:.2f} -> Test Errors: {err} / 7,603 ({(1-err/7603)*100:.3f}%)")

# Base model
base_m = DPCANetV3(num_classes=38).to(DEVICE)
cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
cart_model.load_state_dict(c_cart['model_state_dict'])

# Checkpoint 1: PQ-SCANet (Trial 25)
if os.path.exists("conformal_wafer/experiments_surpass/pq_scanet_trial25_best.pt"):
    m_pq = PQSCANet(cart_model).to(DEVICE)
    c_pq = torch.load("conformal_wafer/experiments_surpass/pq_scanet_trial25_best.pt", map_location=DEVICE, weights_only=False)
    m_pq.load_state_dict(c_pq['model_state_dict'])
    eval_tau_sweep("Trial 25 (PQ-SCANet)", m_pq)

# Checkpoint 2: PICG-Net (Trial 28)
if os.path.exists("conformal_wafer/experiments_surpass/picg_trial28_best.pt"):
    m_picg = PICGNet(cart_model).to(DEVICE)
    c_picg = torch.load("conformal_wafer/experiments_surpass/picg_trial28_best.pt", map_location=DEVICE, weights_only=False)
    m_picg.load_state_dict(c_picg['model_state_dict'])
    eval_tau_sweep("Trial 28 (PICG-Net)", m_picg)

# Checkpoint 3: HiRes-PeakNet (Trial 22)
if os.path.exists("conformal_wafer/experiments_surpass/hires_peaknet_trial22_best.pt"):
    m_hires = HiResPeakNet(cart_model).to(DEVICE)
    c_hires = torch.load("conformal_wafer/experiments_surpass/hires_peaknet_trial22_best.pt", map_location=DEVICE, weights_only=False)
    m_hires.load_state_dict(c_hires['model_state_dict'])
    eval_tau_sweep("Trial 22 (HiRes-PeakNet)", m_hires)
