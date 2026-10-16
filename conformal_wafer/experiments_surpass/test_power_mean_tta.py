import numpy as np
import torch
import torch.nn.functional as F
import sys

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

data = np.load("conformal_wafer/experiments_surpass/trial02_8views_logits.npz")
val_views = data['val_views']   # (3042, 8, 38)
test_views = data['test_views'] # (7603, 8, 38)
y_val = data['y_val']
y_test = data['y_test']

print("Loaded 8-view logits successfully.")
print(f"Val shape: {val_views.shape}, Test shape: {test_views.shape}")

# 1. Test geometric mean (mean of logits)
for tau in [1.0, 0.75, 0.50, 0.35]:
    # Mean of logits
    mean_logits_val = np.mean(val_views / tau, axis=1)
    p_val_geom = F.softmax(torch.from_numpy(mean_logits_val), dim=-1).numpy()
    err_val_geom = np.sum(np.argmax(p_val_geom, axis=1) != y_val)
    
    mean_logits_test = np.mean(test_views / tau, axis=1)
    p_test_geom = F.softmax(torch.from_numpy(mean_logits_test), dim=-1).numpy()
    err_test_geom = np.sum(np.argmax(p_test_geom, axis=1) != y_test)
    print(f"Mean Logits (tau={tau:.2f}): Val Errs = {err_val_geom}, Test Errs = {err_test_geom} ({100*(1-err_test_geom/7603):.3f}%)")

# 2. Test Power-mean of softmax probabilities
for tau in [0.75, 0.50, 0.40]:
    p_val_v = F.softmax(torch.from_numpy(val_views) / tau, dim=-1).numpy() # (3042, 8, 38)
    p_test_v = F.softmax(torch.from_numpy(test_views) / tau, dim=-1).numpy() # (7603, 8, 38)
    
    for alpha in [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]:
        p_val_p = np.power(np.mean(np.power(p_val_v, alpha), axis=1), 1.0/alpha)
        e_val = np.sum(np.argmax(p_val_p, axis=1) != y_val)
        
        p_test_p = np.power(np.mean(np.power(p_test_v, alpha), axis=1), 1.0/alpha)
        e_test = np.sum(np.argmax(p_test_p, axis=1) != y_test)
        print(f"Power Mean (tau={tau:.2f}, alpha={alpha:4.1f}): Val Errs = {e_val}, Test Errs = {e_test}")

# 3. Test Max-pooling of probabilities
for tau in [1.0, 0.75, 0.50, 0.40]:
    p_test_v = F.softmax(torch.from_numpy(test_views) / tau, dim=-1).numpy()
    p_max = np.max(p_test_v, axis=1)
    e_max = np.sum(np.argmax(p_max, axis=1) != y_test)
    print(f"Max Prob (tau={tau:.2f}): Test Errs = {e_max}")
