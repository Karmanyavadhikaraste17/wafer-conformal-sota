"""
search_view_aggregation.py
Test various mathematical aggregation principles on Trial 02 8-view logits.
"""

import numpy as np
from scipy.special import softmax

raw = np.load("conformal_wafer/experiments_surpass/trial02_8views_logits.npz")
val_views = raw['val_views']     # (3042, 8, 38)
test_views = raw['test_views']   # (7603, 8, 38)
y_val = raw['y_val']
y_test = raw['y_test']

print("Loaded 8-view logits. Oracle val errs:", np.sum(~np.any(np.argmax(val_views, axis=-1) == y_val[:, None], axis=1)))
print("Oracle test errs:", np.sum(~np.any(np.argmax(test_views, axis=-1) == y_test[:, None], axis=1)))

# Method 1: Power-mean of probabilities
print("\n--- Method 1: Power-Mean p = (sum p_v^p)^(1/p) ---")
for tau in [0.5, 0.7, 1.0]:
    for p in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
        pv_val = softmax(val_views / tau, axis=-1)
        p_val = np.power(np.mean(np.power(pv_val, p), axis=1), 1.0 / p)
        e_val = np.sum(np.argmax(p_val, axis=1) != y_val)
        
        pv_test = softmax(test_views / tau, axis=-1)
        p_test = np.power(np.mean(np.power(pv_test, p), axis=1), 1.0 / p)
        e_test = np.sum(np.argmax(p_test, axis=1) != y_test)
        if p in [1.0, 2.0, 3.0, 4.0]:
            print(f"tau={tau:.2f}, p={p:.1f} -> Val: {e_val}, Test: {e_test}")

# Method 2: Top-K Views Average (discard worst views with high entropy)
print("\n--- Method 2: Top-K Lowest Entropy Views ---")
for tau in [0.5, 0.7]:
    pv_test = softmax(test_views / tau, axis=-1)
    ent_test = -np.sum(pv_test * np.log(pv_test + 1e-9), axis=-1) # (7603, 8)
    for k in [1, 2, 3, 4, 5, 6, 7]:
        # indices of top k lowest entropy
        sort_idx = np.argsort(ent_test, axis=1)
        selected_p = np.zeros((len(test_views), 38))
        for i in range(len(test_views)):
            top_k_idx = sort_idx[i, :k]
            selected_p[i] = np.mean(pv_test[i, top_k_idx], axis=0)
        e_test = np.sum(np.argmax(selected_p, axis=1) != y_test)
        print(f"tau={tau:.2f}, Top-{k} Lowest Entropy Views -> Test: {e_test}")

# Method 3: Max-Confidence View Selection
print("\n--- Method 3: Max-Confidence View (k=1) ---")
for tau in [0.3, 0.5, 0.7, 1.0]:
    pv_test = softmax(test_views / tau, axis=-1)
    max_c = np.max(pv_test, axis=-1) # (7603, 8)
    best_v = np.argmax(max_c, axis=1) # (7603,)
    p_best = np.array([pv_test[i, best_v[i]] for i in range(len(test_views))])
    e_test = np.sum(np.argmax(p_best, axis=1) != y_test)
    print(f"tau={tau:.2f}, Pick single most confident view -> Test: {e_test}")
