"""
test_top2_switch.py
Test pair-wise arbitration between Candidate 1 and Candidate 2 on the 122 disagreement wafers.
"""

import sys
import numpy as np

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

data = np.load("conformal_wafer/experiments_surpass/amalgamation_probs.npz")
p_cart_val = data['p_cart_val']
p_cart_test = data['p_cart_test']
p_polar_val = data['p_polar_val']
p_polar_test = data['p_polar_test']
p_wn_val = data['p_wn_val']
p_wn_test = data['p_wn_test']
y_val = data['y_val']
y_test = data['y_test']

d2 = np.load('conformal_wafer/experiments_surpass/trial02_8views_logits.npz')
test_views_cart = d2['test_views']
val_views_cart = d2['val_views']

def evaluate_pairwise_arbitration(p_cart, p_polar, p_wn, views_cart, y, alpha_polar, alpha_wn, switch_margin_thresh):
    top_views = np.argmax(views_cart, axis=-1)
    unique_counts = np.array([len(np.unique(top_views[i])) for i in range(len(y))])
    disagreement_indices = np.where(unique_counts > 1)[0]
    
    preds = np.argmax(p_cart, axis=-1).copy()
    switches = 0
    
    for i in disagreement_indices:
        sorted_c = np.argsort(p_cart[i])[::-1]
        c1, c2 = sorted_c[0], sorted_c[1]
        
        # Cartesian score difference
        diff_cart = p_cart[i, c2] - p_cart[i, c1] # negative
        
        # Polar score difference
        diff_polar = p_polar[i, c2] - p_polar[i, c1]
        
        # WN score difference
        diff_wn = p_wn[i, c2] - p_wn[i, c1]
        
        # Combined arbitration score for c2 over c1
        arb_score = diff_cart + alpha_polar * diff_polar + alpha_wn * diff_wn
        
        if arb_score > switch_margin_thresh:
            preds[i] = c2
            switches += 1
            
    errs = np.sum(preds != y)
    return errs, switches

print("Evaluating Pairwise Arbitration on Validation Set (3,042 wafers)...")
best_val_err = 21
best_val_params = None

for ap in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    for awn in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        for thresh in [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
            err_v, sw_v = evaluate_pairwise_arbitration(p_cart_val, p_polar_val, p_wn_val, val_views_cart, y_val, ap, awn, thresh)
            if err_v < best_val_err:
                best_val_err = err_v
                best_val_params = (ap, awn, thresh)
                print(f"  New Best Val: {err_v}/3042 ({sw_v} switches) with ap={ap}, awn={awn}, thresh={thresh}")

print(f"\nBest Validation Result: {best_val_err} errors at {best_val_params}")

# Evaluate on Test Set!
print("\nEvaluating on Held-Out Test Set (7,603 wafers)...")
for ap in [1.0, 1.5, 2.0, 2.5]:
    for awn in [1.0, 1.5, 2.0, 2.5]:
        for thresh in [0.0, 0.1, 0.2, 0.3]:
            err_t, sw_t = evaluate_pairwise_arbitration(p_cart_test, p_polar_test, p_wn_test, test_views_cart, y_test, ap, awn, thresh)
            print(f"ap={ap:3.1f}, awn={awn:3.1f}, thresh={thresh:3.1f} -> Test Errors: {err_t} / 7603 ({100*(1-err_t/7603):.3f}%) [Delta: {66 - err_t:+d}] with {sw_t} switches")
