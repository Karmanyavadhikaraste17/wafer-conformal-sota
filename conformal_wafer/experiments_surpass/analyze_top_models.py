"""
analyze_top_models.py
Analyze predictions and confusion of top checkpoints on the 7,603 test wafers.
"""

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
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

loaders = get_fast_wafer_loaders(batch_size=128)
y_test = loaders['y_test']

def eval_ckpt(path):
    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    model = DPCANetV3_RCLF(base_m).to(DEVICE)
    c = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(c['model_state_dict'])
    model.eval()
    
    all_logits = []
    with torch.no_grad():
        for x, y, _ in loaders['test']:
            x = x.to(DEVICE)
            with autocast('cuda', enabled=True):
                out = model(x)
                l = out[0] if isinstance(out, tuple) else out
            all_logits.append(l.cpu())
    logits = torch.cat(all_logits, dim=0).numpy()
    preds = np.argmax(logits, axis=1)
    errs = np.sum(preds != y_test)
    print(f"{os.path.basename(path)} -> Test Errors: {errs} / {len(y_test)} ({100*(1-errs/len(y_test)):.3f}%)")
    return logits, preds

print("Evaluating Trial 38:")
l38, p38 = eval_ckpt("conformal_wafer/experiments_surpass/distill_rclf_trial38_best.pt")

print("Evaluating Trial 02:")
l02, p02 = eval_ckpt("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt")

# Analyze misclassified wafers
err_idx_38 = set(np.where(p38 != y_test)[0])
err_idx_02 = set(np.where(p02 != y_test)[0])

print(f"\nErrors in Trial 38: {len(err_idx_38)}")
print(f"Errors in Trial 02: {len(err_idx_02)}")
print(f"Intersection of errors: {len(err_idx_38 & err_idx_02)}")
print(f"Union of errors: {len(err_idx_38 | err_idx_02)}")
print(f"Trial 38 correct where Trial 02 was wrong: {len(err_idx_02 - err_idx_38)}")
print(f"Trial 02 correct where Trial 38 was wrong: {len(err_idx_38 - err_idx_02)}")

# Let's check mean of logits between Trial 38 and Trial 02
p_joint = np.argmax(0.5 * l38 + 0.5 * l02, axis=1)
err_joint = np.sum(p_joint != y_test)
print(f"\nJoint Logits (Trial 38 + Trial 02) Errors: {err_joint} / 7603 ({100*(1-err_joint/7603):.3f}%)")
