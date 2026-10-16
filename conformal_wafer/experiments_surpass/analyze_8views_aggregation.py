import numpy as np
from scipy.special import softmax

data = np.load("conformal_wafer/experiments_surpass/trial02_8views_logits.npz")
print("Keys in trial02_8views_logits.npz:", data.files)

test_views = data['test_views'] # shape (7603, 8, 38)
y_test = data['y_test']
val_views = data['val_views']
y_val = data['y_val']

print(f"test_views shape: {test_views.shape}, y_test shape: {y_test.shape}")

# 1. Baseline arithmetic mean
p_mean = np.mean(softmax(test_views, axis=-1), axis=1)
err_mean = int(np.sum(np.argmax(p_mean, axis=1) != y_test))
print(f"Arithmetic Mean (tau=1.0): {err_mean} errors")

# 2. Temperature scan on arithmetic mean
for tau in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    p = np.mean(softmax(test_views / tau, axis=-1), axis=1)
    err = int(np.sum(np.argmax(p, axis=1) != y_test))
    print(f"Arithmetic Mean (tau={tau:.1f}): {err} errors")

# 3. Generalized Power Mean: (1/8 sum p_v^p)^(1/p)
for tau in [0.5, 1.0]:
    probs = softmax(test_views / tau, axis=-1)
    for p_power in [1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0]:
        p_agg = np.power(np.mean(np.power(probs, p_power), axis=1), 1.0 / p_power)
        err = int(np.sum(np.argmax(p_agg, axis=1) != y_test))
        print(f"Power Mean (tau={tau}, p={p_power}): {err} errors")

# 4. Max-Pooling across views: max_v p_{v, c}
for tau in [0.5, 1.0]:
    probs = softmax(test_views / tau, axis=-1)
    p_max = np.max(probs, axis=1)
    err = int(np.sum(np.argmax(p_max, axis=1) != y_test))
    print(f"Max Prob across views (tau={tau}): {err} errors")

# 5. Max Logit across views: max_v z_{v, c}
z_max = np.max(test_views, axis=1)
err_zmax = int(np.sum(np.argmax(z_max, axis=1) != y_test))
print(f"Max Logit across views: {err_zmax} errors")

# 6. Entropy-Weighted Averaging: weight view v by exp(-alpha * Entropy(v))
for tau in [0.5, 1.0]:
    probs = softmax(test_views / tau, axis=-1)
    # entropy: shape (N, 8)
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=-1)
    for alpha in [0.5, 1.0, 2.0, 5.0, 10.0]:
        weights = np.exp(-alpha * entropy) # (N, 8)
        weights /= np.sum(weights, axis=1, keepdims=True)
        p_ew = np.sum(probs * weights[:, :, None], axis=1)
        err = int(np.sum(np.argmax(p_ew, axis=1) != y_test))
        print(f"Entropy-Weighted (tau={tau}, alpha={alpha}): {err} errors")

# 7. Top-K View Selection: Average only the top K views with lowest entropy
for tau in [0.5, 1.0]:
    probs = softmax(test_views / tau, axis=-1)
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=-1)
    for K in [1, 2, 3, 4, 5, 6]:
        idx_sorted = np.argsort(entropy, axis=1) # (N, 8), lowest entropy first
        # take top K views
        p_topk = np.zeros_like(probs[:, 0, :])
        for i in range(len(test_views)):
            top_k_indices = idx_sorted[i, :K]
            p_topk[i] = np.mean(probs[i, top_k_indices, :], axis=0)
        err = int(np.sum(np.argmax(p_topk, axis=1) != y_test))
        print(f"Top-{K} lowest entropy views (tau={tau}): {err} errors")
