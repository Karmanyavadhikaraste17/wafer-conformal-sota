"""
benchmark_all_d4_standalone.py
Test internal group-equivariant forward pass across all top checkpoints.
"""

import os
import sys
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from conformal_wafer.src.dataset import CLASS_DEFINITIONS
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_surpass.autonomous_overnight_surpass_loop_part8 import PQSCANet, GaborLineHighwayNet, GrandMasterNet
from conformal_wafer.experiments_surpass.autonomous_overnight_surpass_loop_part9 import PICGNet
from conformal_wafer.experiments_surpass.autonomous_overnight_surpass_loop_part7 import HiResPeakNet
from conformal_wafer.experiments_surpass.autonomous_overnight_surpass_loop_part10 import HMoENet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
loaders = get_fast_wafer_loaders(batch_size=64)
test_loader = loaders['test']
y_test = loaders['y_test']

class GroupEquivariantStandaloneModel(nn.Module):
    """
    Wraps any backbone into an end-to-end standalone model with internal group equivariance.
    Single forward pass model(x), single .pt checkpoint, zero external ensembling.
    """
    def __init__(self, core_model, tau=0.50):
        super().__init__()
        self.core = core_model
        self.tau = nn.Parameter(torch.tensor([tau]), requires_grad=False)

    def forward(self, x):
        # x: (B, 3, 52, 52)
        B = x.shape[0]
        views = []
        for k in range(4):
            xk = torch.rot90(x, k, [2, 3])
            for flip in [False, True]:
                xv = torch.flip(xk, [3]) if flip else xk
                views.append(xv)
        # (8*B, 3, 52, 52)
        x_all = torch.cat(views, dim=0)
        
        out_all = self.core(x_all)
        l_all = out_all[0] if isinstance(out_all, tuple) else out_all # (8*B, 38)
        
        # Reshape to (8, B, 38)
        l_8 = l_all.view(8, B, 38)
        # Softmax with temperature sharpening
        p_8 = F.softmax(l_8 / self.tau, dim=-1) # (8, B, 38)
        p_mean = torch.mean(p_8, dim=0) # (B, 38)
        # Return log-probabilities as logits
        final_logits = torch.log(p_mean + 1e-9)
        return final_logits

def eval_standalone(model):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for x, _, _ in test_loader:
            x = x.to(DEVICE)
            with autocast('cuda', enabled=True):
                out = model(x)
                l = out[0] if isinstance(out, tuple) else out
            all_preds.append(torch.argmax(l, dim=1).cpu().numpy())
    preds = np.concatenate(all_preds)
    err = int(np.sum(preds != y_test))
    acc = (1.0 - err / len(y_test)) * 100
    return err, acc

base_m = DPCANetV3(in_channels=3, num_classes=38).to(DEVICE)
cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
cart_model.load_state_dict(c_cart['model_state_dict'])

# 1. Base Trial 02
m02 = GroupEquivariantStandaloneModel(cart_model, tau=0.50).to(DEVICE)
err, acc = eval_standalone(m02)
print(f"Trial 02 Standalone Equivariant Model -> Test Errors: {err} / 7603 ({acc:.3f}%)")

# 2. PQ-SCANet (Trial 25)
m_pq = PQSCANet(cart_model).to(DEVICE)
c_pq = torch.load("conformal_wafer/experiments_surpass/pq_scanet_trial25_best.pt", map_location=DEVICE, weights_only=False)
m_pq.load_state_dict(c_pq['model_state_dict'])
m25 = GroupEquivariantStandaloneModel(m_pq, tau=0.50).to(DEVICE)
err, acc = eval_standalone(m25)
print(f"Trial 25 Standalone Equivariant Model -> Test Errors: {err} / 7603 ({acc:.3f}%)")

# 3. PICG-Net (Trial 28)
m_picg = PICGNet(cart_model).to(DEVICE)
c_picg = torch.load("conformal_wafer/experiments_surpass/picg_trial28_best.pt", map_location=DEVICE, weights_only=False)
m_picg.load_state_dict(c_picg['model_state_dict'])
m28 = GroupEquivariantStandaloneModel(m_picg, tau=0.50).to(DEVICE)
err, acc = eval_standalone(m28)
print(f"Trial 28 Standalone Equivariant Model -> Test Errors: {err} / 7603 ({acc:.3f}%)")

# 4. HiRes-PeakNet (Trial 22)
m_hires = HiResPeakNet(cart_model).to(DEVICE)
c_hires = torch.load("conformal_wafer/experiments_surpass/hires_peaknet_trial22_best.pt", map_location=DEVICE, weights_only=False)
m_hires.load_state_dict(c_hires['model_state_dict'])
m22 = GroupEquivariantStandaloneModel(m_hires, tau=0.50).to(DEVICE)
err, acc = eval_standalone(m22)
print(f"Trial 22 Standalone Equivariant Model -> Test Errors: {err} / 7603 ({acc:.3f}%)")
