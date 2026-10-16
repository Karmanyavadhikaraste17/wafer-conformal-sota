import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

from conformal_wafer.src.dataset import CLASS_DEFINITIONS

# Load test logits from Trial 2
data_t2 = np.load("conformal_wafer/experiments_surpass/trial02_8views_logits.npz")
test_views = data_t2['test_views'] # (7603, 8, 38)
y_test = data_t2['y_test']

# Canonical View 0 logits
z0 = test_views[:, 0, :]
preds_0 = np.argmax(z0, axis=-1)
errs_0 = np.sum(preds_0 != y_test)
print(f"Trial 2 Canonical View 0 Errors: {errs_0} / 7603 ({100*(1-errs_0/7603):.3f}%)")

# Mean D4 logits
z_mean = np.mean(test_views, axis=1)
preds_mean = np.argmax(z_mean, axis=-1)
errs_mean = np.sum(preds_mean != y_test)
print(f"Trial 2 D4 Mean Errors: {errs_mean} / 7603 ({100*(1-errs_mean/7603):.3f}%)")

# Dihedral Oracle
preds_all = np.argmax(test_views, axis=-1)
correct_any = np.any(preds_all == y_test[:, None], axis=1)
oracle_errs = np.sum(~correct_any)
print(f"Trial 2 Dihedral Oracle: {oracle_errs} / 7603 ({100*(1-oracle_errs/7603):.3f}%)")

# Find the exact indices of the 80 mean errors
mean_err_idx = np.where(preds_mean != y_test)[0]
recov_idx = [i for i in mean_err_idx if correct_any[i]]
print(f"\nTotal Errors in D4 Mean: {len(mean_err_idx)}")
print(f"Recoverable Wafers across orientations: {len(recov_idx)} / {len(mean_err_idx)}")

# Identify which classes are involved in recoverable wafers
scratch_classes = [i for i, c in enumerate(CLASS_DEFINITIONS) if 'S' in c[1]]
er_el_classes = [i for i, c in enumerate(CLASS_DEFINITIONS) if 'ER' in c[1] or 'EL' in c[1]]

recov_scratch = [i for i in recov_idx if y_test[i] in scratch_classes]
recov_boundary = [i for i in recov_idx if y_test[i] in er_el_classes and i not in recov_scratch]
print(f"Recoverable Scratch Wafers: {len(recov_scratch)}")
print(f"Recoverable Boundary (ER/EL) Wafers: {len(recov_boundary)}")
print(f"Other Recoverable: {len(recov_idx) - len(recov_scratch) - len(recov_boundary)}")
