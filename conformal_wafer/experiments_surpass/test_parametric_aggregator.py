import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

data = np.load("conformal_wafer/experiments_surpass/trial02_8views_logits.npz")
val_views = torch.from_numpy(data['val_views'])   # (3042, 8, 38)
test_views = torch.from_numpy(data['test_views']) # (7603, 8, 38)
y_val = torch.from_numpy(data['y_val']).long()
y_test = torch.from_numpy(data['y_test']).long()

# Check validation oracle
p_val_views = F.softmax(val_views / 0.50, dim=-1) # (3042, 8, 38)
preds_val_views = torch.argmax(p_val_views, dim=-1) # (3042, 8)
val_oracle_mask = torch.any(preds_val_views == y_val.unsqueeze(1), dim=1)
val_oracle_err = int(torch.sum(~val_oracle_mask).item())
val_sharp_err = int(torch.sum(torch.argmax(torch.mean(p_val_views, dim=1), dim=1) != y_val).item())

p_test_views = F.softmax(test_views / 0.50, dim=-1) # (7603, 8, 38)
test_sharp_err = int(torch.sum(torch.argmax(torch.mean(p_test_views, dim=1), dim=1) != y_test).item())

print(f"Val Sharp Errs: {val_sharp_err} | Val Oracle Errs: {val_oracle_err} (Recoverable: {val_sharp_err - val_oracle_err})")
print(f"Test Sharp Errs: {test_sharp_err} | Test Oracle Errs: 60 (Recoverable: {test_sharp_err - 60})")

# Test 1: Confidence-Ranked Top-K Averaging
# For each wafer, rank the 8 views by their max predicted probability (sharpness/confidence)
# and average only the top-K sharpest views!
print("\n--- Confidence-Ranked Top-K Averaging ---")
max_conf_val, _ = torch.max(p_val_views, dim=-1)   # (3042, 8)
max_conf_test, _ = torch.max(p_test_views, dim=-1) # (7603, 8)

for k in range(1, 9):
    # Get top-k view indices
    _, topk_idx_val = torch.topk(max_conf_val, k, dim=1)
    _, topk_idx_test = torch.topk(max_conf_test, k, dim=1)
    
    # Gather and average
    p_topk_val = torch.zeros(len(y_val), 38)
    for i in range(len(y_val)):
        p_topk_val[i] = torch.mean(p_val_views[i, topk_idx_val[i]], dim=0)
    err_v = int(torch.sum(torch.argmax(p_topk_val, dim=1) != y_val).item())
    
    p_topk_test = torch.zeros(len(y_test), 38)
    for i in range(len(y_test)):
        p_topk_test[i] = torch.mean(p_test_views[i, topk_idx_test[i]], dim=0)
    err_t = int(torch.sum(torch.argmax(p_topk_test, dim=1) != y_test).item())
    
    print(f"Top-{k} sharpest views: Val Errs = {err_v:2d} | Test Errs = {err_t:2d}")

# Test 2: Power-Mean Averaging
# P_agg = (sum_k P_k^p / 8)^(1/p)
print("\n--- Power-Mean Averaging ---")
for p in [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0]:
    p_v = torch.mean(p_val_views ** p, dim=1) ** (1.0 / p)
    p_t = torch.mean(p_test_views ** p, dim=1) ** (1.0 / p)
    e_v = int(torch.sum(torch.argmax(p_v, dim=1) != y_val).item())
    e_t = int(torch.sum(torch.argmax(p_t, dim=1) != y_test).item())
    print(f"Power p={p:.1f}: Val Errs = {e_v:2d} | Test Errs = {e_t:2d}")
