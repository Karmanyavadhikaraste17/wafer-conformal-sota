import numpy as np
from scipy.special import softmax

# Load Trial 2 8 views logits
data = np.load("conformal_wafer/experiments_surpass/trial02_8views_logits.npz")
test_views = data['test_views'] # (7603, 8, 38)
y_test = data['y_test']
val_views = data['val_views']
y_val = data['y_val']

probs = softmax(test_views / 0.50, axis=-1) # (7603, 8, 38)
p_tta = np.mean(probs, axis=1) # (7603, 38)
y_pred_tta = np.argmax(p_tta, axis=1)

err_idx = np.where(y_pred_tta != y_test)[0] # 78 errors

view_preds = np.argmax(probs, axis=-1) # (7603, 8)
view_correct = (view_preds == y_test[:, None])
oracle_correct = np.any(view_correct, axis=1)

recoverable = [i for i in err_idx if oracle_correct[i]] # 18 wafers
print(f"Total TTA (tau=0.50) errors: {len(err_idx)}")
print(f"Recoverable wafers by at least one view: {len(recoverable)}")

# For each recoverable wafer, examine what makes the winning view unique:
print("\n" + "=" * 90)
print(f"{'Idx':<6} | {'True Cls':<12} | {'TTA Pred':<12} | {'Win View':<10} | {'Win Margin':<12} | {'Avg Margin':<12} | {'Win MaxLogit':<12} | {'Avg MaxLogit'}")
print("=" * 90)

for i in recoverable:
    true_c = y_test[i]
    pred_c = y_pred_tta[i]
    
    # winning view index
    win_v = np.where(view_correct[i])[0][0]
    
    # logits across views
    z_win = test_views[i, win_v]
    z_sorted = np.sort(z_win)
    win_margin = z_sorted[-1] - z_sorted[-2]
    win_max = z_sorted[-1]
    
    all_margins = []
    all_maxs = []
    for v in range(8):
        zv = np.sort(test_views[i, v])
        all_margins.append(zv[-1] - zv[-2])
        all_maxs.append(zv[-1])
        
    avg_margin = np.mean(all_margins)
    avg_max = np.mean(all_maxs)
    
    print(f"{i:<6} | {true_c:<12} | {pred_c:<12} | {win_v:<10} | {win_margin:<12.2f} | {avg_margin:<12.2f} | {win_max:<12.2f} | {avg_max:.2f}")
