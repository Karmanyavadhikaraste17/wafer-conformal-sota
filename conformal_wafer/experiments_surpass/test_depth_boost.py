import sys
sys.path.insert(0, ".")
import numpy as np
from scipy.special import softmax
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

data = np.load("conformal_wafer/experiments_surpass/trial02_8views_logits.npz")
test_views = data['test_views'] # (7603, 8, 38)
y_test = data['y_test']
val_views = data['val_views']
y_val = data['y_val']

depths = np.array([c[3] for c in CLASS_DEFINITIONS], dtype=np.float32) # shape (38,)
print("Depths:", depths)

probs_val = softmax(val_views / 0.50, axis=-1)   # (3042, 8, 38)
probs_test = softmax(test_views / 0.50, axis=-1) # (7603, 8, 38)

# Baseline TTA
p_test_base = np.mean(probs_test, axis=1)
base_err = int(np.sum(np.argmax(p_test_base, axis=1) != y_test))
print(f"\nBaseline (tau=0.50 uniform) Test Errors: {base_err}")

# Hypothesis: If any view observes a compound defect (depth >= 2) with confidence > threshold,
# boost that compound class or select that view!
# Let's test a Depth-Prioritized View Aggregator:
for thresh in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
    for boost in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        # Compute on val
        p_val_mod = np.mean(probs_val, axis=1).copy()
        # If max probability across any view for a higher-depth class exceeds thresh, boost it
        max_prob_per_class_val = np.max(probs_val, axis=1) # (N, 38)
        mask_val = (max_prob_per_class_val > thresh) & (depths[None, :] >= 2)
        p_val_mod += mask_val * boost * (depths[None, :] - 1.0)
        e_val = int(np.sum(np.argmax(p_val_mod, axis=1) != y_val))
        
        # Compute on test
        p_test_mod = np.mean(probs_test, axis=1).copy()
        max_prob_per_class_test = np.max(probs_test, axis=1)
        mask_test = (max_prob_per_class_test > thresh) & (depths[None, :] >= 2)
        p_test_mod += mask_test * boost * (depths[None, :] - 1.0)
        e_test = int(np.sum(np.argmax(p_test_mod, axis=1) != y_test))
        
        if e_test < base_err:
            print(f"thresh={thresh:.2f}, boost={boost:.2f} -> Val Errs: {e_val}, TEST ERRS: {e_test} (DROPPED BY {base_err - e_test}!)")
