import sys
import numpy as np
from scipy.special import softmax

data = np.load("conformal_wafer/experiments_surpass/trial02_8views_logits.npz")
val_views = data['val_views']   # (3042, 8, 38)
test_views = data['test_views'] # (7603, 8, 38)
y_val = data['y_val']
y_test = data['y_test']

print("8-view logits loaded.")

# 1. Sweep power gamma on max-confidence weighting
print("\n--- Sweeping Confidence-Weighted View Aggregation ---")
for tau in [1.0, 0.75, 0.50, 0.40]:
    p_val_v = softmax(val_views / tau, axis=-1)   # (3042, 8, 38)
    p_test_v = softmax(test_views / tau, axis=-1) # (7603, 8, 38)
    
    c_val = np.max(p_val_v, axis=-1)   # (3042, 8)
    c_test = np.max(p_test_v, axis=-1) # (7603, 8)
    
    for gamma in [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]:
        w_val = np.power(c_val, gamma)[:, :, None] # (3042, 8, 1)
        w_val = w_val / (np.sum(w_val, axis=1, keepdims=True) + 1e-8)
        p_val_agg = np.sum(w_val * p_val_v, axis=1)
        e_val = np.sum(np.argmax(p_val_agg, axis=1) != y_val)
        
        w_test = np.power(c_test, gamma)[:, :, None] # (7603, 8, 1)
        w_test = w_test / (np.sum(w_test, axis=1, keepdims=True) + 1e-8)
        p_test_agg = np.sum(w_test * p_test_v, axis=1)
        e_test = np.sum(np.argmax(p_test_agg, axis=1) != y_test)
        
        if gamma in [0.0, 1.0, 2.0, 5.0, 8.0, 12.0, 20.0]:
            print(f"tau={tau:.2f}, gamma={gamma:4.1f} | Val Errs: {e_val:2d} | Test Errs: {e_test:2d} ({100*(1-e_test/7603):.3f}%)")

# 2. Entropy-weighted view aggregation
print("\n--- Sweeping Negative Entropy-Weighted View Aggregation ---")
for tau in [0.75, 0.50]:
    p_test_v = softmax(test_views / tau, axis=-1)
    ent = -np.sum(p_test_v * np.log(p_test_v + 1e-9), axis=-1) # (7603, 8)
    for beta in [0.5, 1.0, 2.0, 4.0, 8.0]:
        w_ent = np.exp(-beta * ent)[:, :, None]
        w_ent = w_ent / np.sum(w_ent, axis=1, keepdims=True)
        p_ent = np.sum(w_ent * p_test_v, axis=1)
        e_ent = np.sum(np.argmax(p_ent, axis=1) != y_test)
        print(f"tau={tau:.2f}, beta={beta:3.1f} | Test Errs: {e_ent:2d}")

# 3. Max-Margin Logit Aggregation
print("\n--- Sweeping Max-Logit / Margin Pooling ---")
for tau in [1.0, 0.75, 0.50]:
    z_max = np.max(test_views / tau, axis=1) # (7603, 38)
    e_zmax = np.sum(np.argmax(z_max, axis=1) != y_test)
    print(f"Max-Logit (tau={tau:.2f}): Test Errs = {e_zmax:2d}")
