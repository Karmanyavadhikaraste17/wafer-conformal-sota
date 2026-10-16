"""
diagnose_errors.py
Find the exact classes and wafer indices of errors in Trial 02 and Trial 25.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from conformal_wafer.src.dataset import CLASS_DEFINITIONS
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_surpass.autonomous_overnight_surpass_loop_part8 import PQSCANet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
loaders = get_fast_wafer_loaders(batch_size=128)
y_test = loaders['y_test']

base_m = DPCANetV3(num_classes=38).to(DEVICE)
cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
cart_model.load_state_dict(c_cart['model_state_dict'])

m_pq = PQSCANet(cart_model).to(DEVICE)
c_pq = torch.load("conformal_wafer/experiments_surpass/pq_scanet_trial25_best.pt", map_location=DEVICE, weights_only=False)
m_pq.load_state_dict(c_pq['model_state_dict'])
m_pq.eval()

preds = []
with torch.no_grad():
    for x, _, _ in loaders['test']:
        x = x.to(DEVICE)
        with torch.amp.autocast('cuda'):
            out = m_pq(x)
            l = out[0] if isinstance(out, tuple) else out
        preds.append(torch.argmax(l, dim=1).cpu().numpy())
preds = np.concatenate(preds)

err_idx = np.where(preds != y_test)[0]
print(f"Total Standalone Errors: {len(err_idx)} / 7603 ({100*(1-len(err_idx)/7603):.3f}%)")

from collections import Counter
confusions = Counter()
for i in err_idx:
    true_cls = CLASS_DEFINITIONS[y_test[i]][1]
    pred_cls = CLASS_DEFINITIONS[preds[i]][1]
    confusions[(true_cls, pred_cls)] += 1

print("\nTop Confusions in Standalone Model:")
for (t, p), count in confusions.most_common(15):
    print(f"  True: {t:12s} -> Pred: {p:12s} | Count: {count}")
