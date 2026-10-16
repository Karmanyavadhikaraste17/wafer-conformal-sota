"""
check_polarnet_on_recoverable.py
Check PolarNet predictions on the 20 recoverable wafers.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS
from conformal_wafer.src.train_optimized_sota import PolarNet
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
loaders = get_fast_wafer_loaders(batch_size=128)
y_test = loaders['y_test']

# Load PolarNet
m_polar = PolarNet(num_classes=38).to(DEVICE)
c_polar = torch.load("conformal_wafer/checkpoints/polarnet-v2_fast80_best.pt", map_location=DEVICE, weights_only=False)
m_polar.load_state_dict(c_polar['model_state_dict'])
m_polar.eval()

# Extract PolarNet test predictions
all_polar_preds = []
with torch.no_grad():
    for x, _, _ in loaders['test']:
        x = x.to(DEVICE)
        lp = m_polar(x)
        all_polar_preds.append(lp.argmax(dim=-1).cpu().numpy())
polar_preds = np.concatenate(all_polar_preds)

# Recoverable indices from Trial 2
data = np.load('conformal_wafer/experiments_surpass/trial02_8views_logits.npz')
test_views = data['test_views']
mean_logits = np.mean(test_views, axis=1)
pred_mean = np.argmax(mean_logits, axis=-1)
mean_err_indices = np.where(pred_mean != y_test)[0]
recoverable_indices = [idx for idx in mean_err_indices if y_test[idx] in np.argmax(test_views[idx], axis=-1)]

print(f"PolarNet Overall Test Accuracy: {100*(1 - np.sum(polar_preds != y_test)/7603):.3f}% ({np.sum(polar_preds != y_test)} errors)")
print("\nPolarNet predictions on the 20 recoverable wafers:")
print(f"{'Idx':<5} | {'True Name':<12} | {'T2 Mean Pred':<12} | {'PolarNet Pred':<12} | {'Polar Correct?'}")
print("-" * 65)

polar_correct_count = 0
for idx in recoverable_indices:
    true_label = y_test[idx]
    true_name = CLASS_DEFINITIONS[true_label][1]
    t2_pred_name = CLASS_DEFINITIONS[pred_mean[idx]][1]
    pol_pred_name = CLASS_DEFINITIONS[polar_preds[idx]][1]
    is_corr = (polar_preds[idx] == true_label)
    if is_corr: polar_correct_count += 1
    print(f"{idx:<5d} | {true_name:<12s} | {t2_pred_name:<12s} | {pol_pred_name:<12s} | {is_corr}")

print(f"\nPolarNet correctly solved {polar_correct_count} / {len(recoverable_indices)} of the recoverable errors!")
