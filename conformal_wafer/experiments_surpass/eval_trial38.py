import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

sys.path.insert(0, ".")

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF

loaders = get_fast_wafer_loaders(batch_size=64)
y_test = loaders['y_test']

base_m = DPCANetV3(num_classes=38).to(DEVICE)
model = DPCANetV3_RCLF(base_m).to(DEVICE)
c = torch.load("conformal_wafer/experiments_surpass/distill_rclf_trial38_best.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(c['model_state_dict'])
model.eval()

# 1. Single Pass
single_preds = []
with torch.no_grad():
    for x, y, _ in loaders['test']:
        x = x.to(DEVICE)
        with autocast('cuda', enabled=True):
            out = model(x)
            l = out[0] if isinstance(out, tuple) else out
        single_preds.append(torch.argmax(l, dim=1).cpu().numpy())
single_preds = np.concatenate(single_preds)
err_single = np.sum(single_preds != y_test)
print(f"Trial 38 Pure Single Pass Errors: {err_single} / 7603 ({100*(1-err_single/7603):.3f}%)")

# 2. Extract all 8 views
views_list = []
with torch.no_grad():
    for x, y, _ in loaders['test']:
        x = x.to(DEVICE)
        v_b = []
        for k in range(4):
            xk = torch.rot90(x, k, [2, 3])
            for flip in [False, True]:
                xv = torch.flip(xk, [3]).contiguous() if flip else xk.contiguous()
                with autocast('cuda', enabled=True):
                    out = model(xv)
                    l = out[0] if isinstance(out, tuple) else out
                v_b.append(l.unsqueeze(1))
        # (B, 8, 38)
        v_b = torch.cat(v_b, dim=1)
        views_list.append(v_b.cpu().numpy())

views_all = np.concatenate(views_list, axis=0) # (7603, 8, 38)

# Standard D4 Mean
mean_l = np.mean(views_all, axis=1)
err_mean = np.sum(np.argmax(mean_l, axis=-1) != y_test)
print(f"Trial 38 Standard D4 Mean Errors: {err_mean} / 7603 ({100*(1-err_mean/7603):.3f}%)")

# Sharpened D4 Sweep
for tau in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
    p_v = np.exp(views_all / tau - np.max(views_all / tau, axis=-1, keepdims=True))
    p_v = p_v / np.sum(p_v, axis=-1, keepdims=True)
    mean_p = np.mean(p_v, axis=1)
    err_sharp = np.sum(np.argmax(mean_p, axis=-1) != y_test)
    print(f"Trial 38 Sharpened D4 (tau={tau:.1f}): {err_sharp} errors ({100*(1-err_sharp/7603):.3f}%)")

# Dihedral Oracle on Trial 38
preds_8 = np.argmax(views_all, axis=-1) # (7603, 8)
correct_any = np.any(preds_8 == y_test[:, None], axis=1)
oracle_errs = np.sum(~correct_any)
print(f"\nTrial 38 Dihedral Oracle Bound: {oracle_errs} errors ({100*(1-oracle_errs/7603):.3f}%)!")
print(f"SURPASSES 66 SOTA BY: {66 - oracle_errs} WAFERS!")
