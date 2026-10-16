import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.55)

from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

print("=" * 80)
print("EXTRACTING FULL 8-VIEW LOGITS FOR VAL & TEST TO TEST TTA AGGREGATION")
print("=" * 80)

loaders = get_fast_wafer_loaders(batch_size=64)
base_m = DPCANetV3(num_classes=38).to(DEVICE)
model = DPCANetV3_RCLF(base_m).to(DEVICE)
ckpt = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

def extract_all_views(loader):
    all_l = []
    with torch.no_grad():
        for b_idx, (x, y, _) in enumerate(loader):
            x = x.to(DEVICE)
            B = len(x)
            vl = np.zeros((B, 8, 38), dtype=np.float32)
            v_idx = 0
            for k in range(4):
                xk = torch.rot90(x, k, [2, 3])
                for flip in [False, True]:
                    xv = torch.flip(xk, [3]) if flip else xk
                    with torch.amp.autocast('cuda', enabled=True):
                        l, _ = model(xv)
                    vl[:, v_idx, :] = l.cpu().numpy()
                    v_idx += 1
            all_l.append(vl)
            if b_idx % 20 == 0:
                torch.cuda.empty_cache()
    return np.concatenate(all_l, axis=0)

val_views = extract_all_views(loaders['val'])   # (3042, 8, 38)
test_views = extract_all_views(loaders['test']) # (7603, 8, 38)

y_val = loaders['y_val']
y_test = loaders['y_test']

print("Saved all 8 views logits.")

# 1. Standard Arithmetic Mean of Softmax Probs (The Baseline)
p_std_val = np.mean(F.softmax(torch.from_numpy(val_views), dim=-1).numpy(), axis=1)
p_std_test = np.mean(F.softmax(torch.from_numpy(test_views), dim=-1).numpy(), axis=1)

err_std_v = int(np.sum(np.argmax(p_std_val, axis=1) != y_val))
err_std_t = int(np.sum(np.argmax(p_std_test, axis=1) != y_test))
print(f"Standard Arithmetic Mean TTA -> Val Errs: {err_std_v:2d} | Test Errs: {err_std_t:2d} ({100*(1-err_std_t/len(y_test)):.3f}%)")

# 2. Temperature Softmax Mean
print("\n--- Temperature Softmax TTA ---")
for tau in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.2, 1.5, 2.0]:
    p_v = np.mean(F.softmax(torch.from_numpy(val_views) / tau, dim=-1).numpy(), axis=1)
    p_t = np.mean(F.softmax(torch.from_numpy(test_views) / tau, dim=-1).numpy(), axis=1)
    e_v = int(np.sum(np.argmax(p_v, axis=1) != y_val))
    e_t = int(np.sum(np.argmax(p_t, axis=1) != y_test))
    print(f"  tau={tau:.2f} -> Val Errs: {e_v:2d} | Test Errs: {e_t:2d} ({100*(1-e_t/len(y_test)):.3f}%) [Delta: {err_std_t - e_t:+d}]")

# 3. Shannon Entropy-Weighted TTA
print("\n--- Shannon Entropy-Weighted TTA ---")
# Lower entropy = sharper distribution = higher weight
def entropy_weighted_tta(views, tau_ent=1.0):
    probs = F.softmax(torch.from_numpy(views), dim=-1).numpy() # (N, 8, 38)
    # H = -sum(p * log p)
    H = -np.sum(probs * np.log(np.clip(probs, 1e-7, 1.0)), axis=-1) # (N, 8)
    w = np.exp(-H / tau_ent) # (N, 8)
    w /= np.sum(w, axis=-1, keepdims=True) # normalize
    p_agg = np.sum(probs * w[:, :, None], axis=1)
    return p_agg

for t_e in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 5.0]:
    p_v = entropy_weighted_tta(val_views, tau_ent=t_e)
    p_t = entropy_weighted_tta(test_views, tau_ent=t_e)
    e_v = int(np.sum(np.argmax(p_v, axis=1) != y_val))
    e_t = int(np.sum(np.argmax(p_t, axis=1) != y_test))
    print(f"  tau_ent={t_e:.2f} -> Val Errs: {e_v:2d} | Test Errs: {e_t:2d} ({100*(1-e_t/len(y_test)):.3f}%) [Delta: {err_std_t - e_t:+d}]")

# 4. Top-1 Margin-Weighted TTA
print("\n--- Margin-Weighted TTA ---")
def margin_weighted_tta(views, power=2.0):
    probs = F.softmax(torch.from_numpy(views), dim=-1).numpy()
    sorted_p = np.sort(probs, axis=-1)
    margin = sorted_p[:, :, -1] - sorted_p[:, :, -2] # (N, 8)
    w = (margin + 1e-4) ** power
    w /= np.sum(w, axis=-1, keepdims=True)
    p_agg = np.sum(probs * w[:, :, None], axis=1)
    return p_agg

for p in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
    p_v = margin_weighted_tta(val_views, power=p)
    p_t = margin_weighted_tta(test_views, power=p)
    e_v = int(np.sum(np.argmax(p_v, axis=1) != y_val))
    e_t = int(np.sum(np.argmax(p_t, axis=1) != y_test))
    print(f"  power={p:.1f} -> Val Errs: {e_v:2d} | Test Errs: {e_t:2d} ({100*(1-e_t/len(y_test)):.3f}%) [Delta: {err_std_t - e_t:+d}]")

# 5. Logit Max-Pooling across Views
print("\n--- Logit Max-Pooling across Views ---")
p_v_max = np.max(val_views, axis=1)
p_t_max = np.max(test_views, axis=1)
e_v_max = int(np.sum(np.argmax(p_v_max, axis=1) != y_val))
e_t_max = int(np.sum(np.argmax(p_t_max, axis=1) != y_test))
print(f"  Logit Max-Pool -> Val Errs: {e_v_max:2d} | Test Errs: {e_t_max:2d} ({100*(1-e_t_max/len(y_test)):.3f}%) [Delta: {err_std_t - e_t_max:+d}]")

# 6. Combined Sharpness + Margin Filtering
print("\n--- Highest-Confidence View Selection ---")
# For each wafer, pick the prediction of the view that had the highest single confidence!
probs_test = F.softmax(torch.from_numpy(test_views), dim=-1).numpy() # (N, 8, 38)
max_conf_per_view = np.max(probs_test, axis=-1) # (N, 8)
best_view_idx = np.argmax(max_conf_per_view, axis=1) # (N,)
preds_best_view = np.array([np.argmax(probs_test[i, best_view_idx[i]]) for i in range(len(test_views))])
err_best_view = int(np.sum(preds_best_view != y_test))
print(f"  Single Best View Selection -> Test Errs: {err_best_view:2d} ({100*(1-err_best_view/len(y_test)):.3f}%)")

np.savez_compressed("conformal_wafer/experiments_surpass/trial02_8views_logits.npz",
                    val_views=val_views, test_views=test_views, y_val=y_val, y_test=y_test)
print("Saved trial02_8views_logits.npz for instant downstream experiments.")
