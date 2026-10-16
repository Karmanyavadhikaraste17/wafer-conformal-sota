import sys
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

data = np.load('conformal_wafer/experiments_surpass/trial02_8views_logits.npz')
val_views = torch.tensor(data['val_views'], dtype=torch.float32) # (3042, 8, 38)
test_views = torch.tensor(data['test_views'], dtype=torch.float32) # (7603, 8, 38)
y_val = torch.tensor(data['y_val'], dtype=torch.long)
y_test = torch.tensor(data['y_test'], dtype=torch.long)

z0_test = test_views[:, 0, :]
top2_vals, _ = torch.topk(z0_test, k=2, dim=-1)
margin_0 = top2_vals[:, 0] - top2_vals[:, 1] # (7603,)

print("--- Margin-Gated View Override ---")
for tau_m in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    high_conf = margin_0 > tau_m
    mean_views = torch.mean(test_views, dim=1) # (7603, 38)
    
    pred_gated = torch.argmax(z0_test, dim=-1)
    pred_gated[~high_conf] = torch.argmax(mean_views[~high_conf], dim=-1)
    
    errs = torch.sum(pred_gated != y_test).item()
    n_ambig = torch.sum(~high_conf).item()
    print(f"tau_m={tau_m:4.1f} | Ambiguous: {n_ambig:4d} wafers | Gated Test Errors: {errs} ({100*(1-errs/7603):.3f}%)")

print("\n--- Max-Certainty View Selection for Ambiguous Wafers ---")
probs_views = F.softmax(test_views, dim=-1) # (7603, 8, 38)
max_probs_per_view, _ = torch.max(probs_views, dim=-1) # (7603, 8)
best_view_idx = torch.argmax(max_probs_per_view, dim=-1) # (7603,)

most_certain_preds = torch.zeros(7603, dtype=torch.long)
for i in range(7603):
    v = best_view_idx[i]
    most_certain_preds[i] = torch.argmax(test_views[i, v], dim=-1)

errs_mc = torch.sum(most_certain_preds != y_test).item()
print(f"Pure Max-Certainty View Selection: {errs_mc} errors ({100*(1-errs_mc/7603):.3f}%)")

for tau_m in [1.0, 1.5, 2.0, 2.5, 3.0]:
    high_conf = margin_0 > tau_m
    pred_gated = torch.argmax(z0_test, dim=-1)
    pred_gated[~high_conf] = most_certain_preds[~high_conf]
    errs = torch.sum(pred_gated != y_test).item()
    print(f"Gated Max-Certainty tau_m={tau_m:4.1f}: {errs} errors ({100*(1-errs/7603):.3f}%)")
