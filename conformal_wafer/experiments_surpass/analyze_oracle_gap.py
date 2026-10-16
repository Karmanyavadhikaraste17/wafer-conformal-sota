"""
analyze_oracle_gap.py
Inspect the 18 wafers that separate the 78-error standalone model from the 60-error oracle bound.
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, ".")

from conformal_wafer.src.dataset import CLASS_DEFINITIONS
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
loaders = get_fast_wafer_loaders(batch_size=128)
y_test = loaders['y_test']

base_m = DPCANetV3(num_classes=38).to(DEVICE)
cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
cart_model.load_state_dict(c_cart['model_state_dict'])
cart_model.eval()

all_logits_8 = []
with torch.no_grad():
    for x, _, _ in loaders['test']:
        x = x.to(DEVICE)
        B = x.shape[0]
        views = []
        for k in range(4):
            xk = torch.rot90(x, k, [2, 3])
            for flip in [False, True]:
                xv = torch.flip(xk, [3]) if flip else xk
                views.append(xv)
        x_all = torch.cat(views, dim=0)
        with torch.amp.autocast('cuda'):
            logits_all = cart_model(x_all)
            if isinstance(logits_all, tuple):
                logits_all = logits_all[0]
        logits_8 = logits_all.view(8, B, 38).permute(1, 0, 2).cpu()
        all_logits_8.append(logits_8)

all_logits_8 = torch.cat(all_logits_8, dim=0) # (7603, 8, 38)
probs_8 = F.softmax(all_logits_8 / 0.50, dim=-1) # (7603, 8, 38)
mean_prob = torch.mean(probs_8, dim=1) # (7603, 38)
preds_mean = torch.argmax(mean_prob, dim=-1).numpy()

# Find the 18 gap wafers
gap_indices = []
for i in range(len(y_test)):
    view_preds = torch.argmax(all_logits_8[i], dim=-1).numpy()
    if preds_mean[i] != y_test[i] and y_test[i] in view_preds:
        gap_indices.append(i)

print(f"Total Gap Wafers: {len(gap_indices)}")
print("-" * 80)
for idx in gap_indices:
    true_label = y_test[idx]
    pred_label = preds_mean[idx]
    true_name = CLASS_DEFINITIONS[true_label][1]
    pred_name = CLASS_DEFINITIONS[pred_label][1]
    
    # View distribution
    view_preds = torch.argmax(all_logits_8[idx], dim=-1).numpy()
    num_correct_views = np.sum(view_preds == true_label)
    
    # Probabilities
    true_probs = probs_8[idx, :, true_label].numpy()
    pred_probs = probs_8[idx, :, pred_label].numpy()
    
    print(f"Wafer {idx:4d} | True: {true_name:12s} ({true_label:2d}) vs Pred: {pred_name:12s} ({pred_label:2d}) | Correct Views: {num_correct_views}/8")
    print(f"   True Probs across 8 views: {[round(float(p), 3) for p in true_probs]}")
    print(f"   Pred Probs across 8 views: {[round(float(p), 3) for p in pred_probs]}")
