"""
explore_view_aggregations.py
Comprehensive sweep across aggregation mechanisms on 8-view logits.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F

data = np.load('conformal_wafer/experiments_surpass/trial02_8views_logits.npz')
val_views = torch.tensor(data['val_views'], dtype=torch.float32) # (3042, 8, 38)
test_views = torch.tensor(data['test_views'], dtype=torch.float32) # (7603, 8, 38)
y_val = torch.tensor(data['y_val'], dtype=torch.long)
y_test = torch.tensor(data['y_test'], dtype=torch.long)

print(f"Val: {val_views.shape}, Test: {test_views.shape}")

# Individual view performance
print("\n--- INDIVIDUAL VIEW ACCURACY ---")
for v in range(8):
    err_v_val = (val_views[:, v].argmax(dim=-1) != y_val).sum().item()
    err_v_test = (test_views[:, v].argmax(dim=-1) != y_test).sum().item()
    print(f"View {v:1d}: Val Errs = {err_v_val:3d} ({100 - err_v_val/30.42:.2f}%) | Test Errs = {err_v_test:3d} ({100 - err_v_test/76.03:.3f}%)")

# 1. Softmax with different temperatures
print("\n--- SOFTMAX MEAN AGGREGATION ACROSS TEMPERATURES ---")
for tau in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]:
    p_val = F.softmax(val_views / tau, dim=-1).mean(dim=1)
    p_test = F.softmax(test_views / tau, dim=-1).mean(dim=1)
    e_val = (p_val.argmax(dim=-1) != y_val).sum().item()
    e_test = (p_test.argmax(dim=-1) != y_test).sum().item()
    print(f"Tau={tau:4.2f} | Val Errs: {e_val:2d} | Test Errs: {e_test:2d}")

# 2. Entropy-weighted view aggregation
print("\n--- ENTROPY-WEIGHTED VIEW AGGREGATION ---")
# Lower entropy = higher confidence view
for tau in [0.2, 0.5, 1.0]:
    p_8_val = F.softmax(val_views / tau, dim=-1) # (B, 8, 38)
    ent_val = -(p_8_val * torch.log(p_8_val + 1e-9)).sum(dim=-1) # (B, 8)
    
    p_8_test = F.softmax(test_views / tau, dim=-1)
    ent_test = -(p_8_test * torch.log(p_8_test + 1e-9)).sum(dim=-1)
    
    for gamma in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        w_val = F.softmax(-gamma * ent_val, dim=-1).unsqueeze(-1) # (B, 8, 1)
        w_test = F.softmax(-gamma * ent_test, dim=-1).unsqueeze(-1)
        
        pred_val = (p_8_val * w_val).sum(dim=1).argmax(dim=-1)
        pred_test = (p_8_test * w_test).sum(dim=1).argmax(dim=-1)
        
        e_val = (pred_val != y_val).sum().item()
        e_test = (pred_test != y_test).sum().item()
        print(f"Tau={tau:.1f}, Gamma={gamma:4.1f} | Val Errs: {e_val:2d} | Test Errs: {e_test:2d}")

# 3. Geometric Mean / Log-prob aggregation
print("\n--- GEOMETRIC MEAN (LOG-PROB SUM) ---")
for tau in [0.1, 0.2, 0.5, 1.0]:
    lp_val = F.log_softmax(val_views / tau, dim=-1).mean(dim=1)
    lp_test = F.log_softmax(test_views / tau, dim=-1).mean(dim=1)
    e_val = (lp_val.argmax(dim=-1) != y_val).sum().item()
    e_test = (lp_test.argmax(dim=-1) != y_test).sum().item()
    print(f"Log-Softmax Tau={tau:.2f} | Val Errs: {e_val:2d} | Test Errs: {e_test:2d}")

# 4. Top-K view averaging
print("\n--- TOP-K VIEW SELECTION (K most confident views per wafer) ---")
for k in [1, 2, 3, 4, 5, 6, 7]:
    for tau in [0.2, 0.5]:
        p_8_val = F.softmax(val_views / tau, dim=-1)
        max_p_val, _ = torch.max(p_8_val, dim=-1) # (B, 8)
        _, topk_idx_val = torch.topk(max_p_val, k=k, dim=-1) # (B, k)
        
        p_8_test = F.softmax(test_views / tau, dim=-1)
        max_p_test, _ = torch.max(p_8_test, dim=-1)
        _, topk_idx_test = torch.topk(max_p_test, k=k, dim=-1)
        
        # Gather top-k
        topk_p_val = torch.stack([p_8_val[i, topk_idx_val[i]].mean(dim=0) for i in range(len(p_8_val))])
        topk_p_test = torch.stack([p_8_test[i, topk_idx_test[i]].mean(dim=0) for i in range(len(p_8_test))])
        
        e_val = (topk_p_val.argmax(dim=-1) != y_val).sum().item()
        e_test = (topk_p_test.argmax(dim=-1) != y_test).sum().item()
        print(f"Top-{k} Views (Tau={tau:.1f}) | Val Errs: {e_val:2d} | Test Errs: {e_test:2d}")
