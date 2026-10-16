"""
check_wafernext_on_recoverable.py
Check WaferNeXt predictions on the 20 recoverable wafers.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS
from conformal_wafer.src.train_wafernext import WaferNeXt
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
loaders = get_fast_wafer_loaders(batch_size=128)
y_test = loaders['y_test']

# Load WaferNeXt
m_wn = WaferNeXt(dims=[40, 64, 120, 192]).to(DEVICE)
c_wn = torch.load("conformal_wafer/checkpoints/wafernext_compact_best.pt", map_location=DEVICE, weights_only=False)
m_wn.load_state_dict(c_wn['model_state_dict'])
m_wn.eval()

# Extract WaferNeXt test predictions
all_wn_preds = []
with torch.no_grad():
    for x, _, _ in loaders['test']:
        x = x.to(DEVICE)
        lwn = m_wn(x)
        all_wn_preds.append(lwn.argmax(dim=-1).cpu().numpy())
wn_preds = np.concatenate(all_wn_preds)

# Recoverable indices from Trial 2
data = np.load('conformal_wafer/experiments_surpass/trial02_8views_logits.npz')
test_views = data['test_views']
mean_logits = np.mean(test_views, axis=1)
pred_mean = np.argmax(mean_logits, axis=-1)
mean_err_indices = np.where(pred_mean != y_test)[0]
recoverable_indices = [idx for idx in mean_err_indices if y_test[idx] in np.argmax(test_views[idx], axis=-1)]

print(f"WaferNeXt Overall Test Accuracy: {100*(1 - np.sum(wn_preds != y_test)/7603):.3f}% ({np.sum(wn_preds != y_test)} errors)")
print("\nWaferNeXt predictions on the 20 recoverable wafers:")
print(f"{'Idx':<5} | {'True Name':<12} | {'T2 Mean Pred':<12} | {'WaferNeXt Pred':<12} | {'WN Correct?'}")
print("-" * 65)

wn_correct_count = 0
for idx in recoverable_indices:
    true_label = y_test[idx]
    true_name = CLASS_DEFINITIONS[true_label][1]
    t2_pred_name = CLASS_DEFINITIONS[pred_mean[idx]][1]
    wn_pred_name = CLASS_DEFINITIONS[wn_preds[idx]][1]
    is_corr = (wn_preds[idx] == true_label)
    if is_corr: wn_correct_count += 1
    print(f"{idx:<5d} | {true_name:<12s} | {t2_pred_name:<12s} | {wn_pred_name:<12s} | {is_corr}")

print(f"\nWaferNeXt correctly solved {wn_correct_count} / {len(recoverable_indices)} of the recoverable errors!")
