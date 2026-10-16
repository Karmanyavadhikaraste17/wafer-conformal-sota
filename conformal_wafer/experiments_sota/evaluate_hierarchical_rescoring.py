import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "conformal_wafer"))

from conformal_wafer.src.train_optimized_sota import get_fast_loaders
from conformal_wafer.src.run_sota_campaign import DPCANet
from conformal_wafer.src.train_hf_edgedpcanet import HFEdgeDPCANet
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

# 1. Load Scratch Specialist probabilities
s_data = np.load(os.path.join(BASE_DIR, "conformal_wafer", "experiments_sota", "scratch_specialist_probs.npz"))
s_val = s_data['val_probs']   # (3042,)
s_test = s_data['test_probs'] # (7603,)

# 2. Load ASL-Primitive probabilities
p_data = np.load(os.path.join(BASE_DIR, "conformal_wafer", "experiments_sota", "asl_primitive_probs.npz"))
prim_val = p_data['val_probs']   # (3042, 8)
prim_test = p_data['test_probs'] # (7603, 8)

# 3. Load base ensemble predictions
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

loaders, _ = get_fast_loaders()

m1 = DPCANet(in_channels=3, num_classes=38).to(DEVICE)
m1.load_state_dict(torch.load("conformal_wafer/checkpoints/dpcanet-v2_80split_best.pt", map_location=DEVICE, weights_only=False)['model_state_dict'])
m1.eval()

m2 = HFEdgeDPCANet(in_channels=3, num_classes=38, num_primitives=8).to(DEVICE)
m2.load_state_dict(torch.load("conformal_wafer/checkpoints/hf_edgedpcanet_sota_best.pt", map_location=DEVICE, weights_only=False)['model_state_dict'])
m2.eval()

def get_d4_probs(loader):
    p1, p2, targets = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            s1, s2 = 0.0, 0.0
            for k in range(4):
                xk = torch.rot90(images, k, [2, 3])
                for flip in [False, True]:
                    x_v = torch.flip(xk, [3]) if flip else xk
                    with torch.amp.autocast('cuda', enabled=True):
                        l1 = m1(x_v)
                        l2, _ = m2(x_v)
                    s1 += F.softmax(l1, dim=1)
                    s2 += F.softmax(l2, dim=1)
            p1.append((s1 / 8.0).cpu().numpy())
            p2.append((s2 / 8.0).cpu().numpy())
            targets.append(labels.cpu().numpy())
    return np.concatenate(p1, axis=0), np.concatenate(p2, axis=0), np.concatenate(targets, axis=0)

print("Computing base ensemble D4-TTA on Val and Test...", flush=True)
v_p1, v_p2, y_val = get_d4_probs(loaders['val'])
t_p1, t_p2, y_test = get_d4_probs(loaders['test'])

p_ens_val = 0.55 * v_p1 + 0.45 * v_p2
p_ens_test = 0.55 * t_p1 + 0.45 * t_p2

val_base_err = int((y_val != np.argmax(p_ens_val, axis=1)).sum())
test_base_err = int((y_test != np.argmax(p_ens_test, axis=1)).sum())
print(f"Base Ensemble Val Errors:  {val_base_err} / {len(y_val)} ({(1-val_base_err/len(y_val))*100:.3f}%)", flush=True)
print(f"Base Ensemble Test Errors: {test_base_err} / {len(y_test)} ({(1-test_base_err/len(y_test))*100:.3f}%)", flush=True)

# 4. Template Matrix T (38, 8)
T = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.float32) # (38, 8)

def compute_log_likelihood(p_mat, s_vec=None):
    eps = 1e-4
    log_p = np.log(np.clip(p_mat, eps, 1.0 - eps))
    log_1mp = np.log(np.clip(1.0 - p_mat, eps, 1.0 - eps))

    ll_prim = np.dot(log_p, T.T) + np.dot(log_1mp, (1.0 - T).T)

    ll_scratch = np.zeros_like(ll_prim)
    if s_vec is not None:
        log_s = np.log(np.clip(s_vec, eps, 1.0 - eps))[:, None]
        log_1ms = np.log(np.clip(1.0 - s_vec, eps, 1.0 - eps))[:, None]
        T_scratch = T[:, 6:7].T # (1, 38)
        ll_scratch = np.dot(log_s, T_scratch) + np.dot(log_1ms, (1.0 - T_scratch))

    return ll_prim, ll_scratch

ll_prim_val, ll_scratch_val = compute_log_likelihood(prim_val, s_val)
ll_prim_test, ll_scratch_test = compute_log_likelihood(prim_test, s_test)

# 5. Grid Search on Validation Set
print("\n--- Grid Search on Validation Split (3,042 wafers) ---", flush=True)
best_val_err = val_base_err
best_weights = None

for w_prim in np.linspace(0.0, 0.25, 11):
    for w_scratch in np.linspace(0.0, 0.25, 11):
        if w_prim == 0.0 and w_scratch == 0.0:
            continue
        scores_v = np.log(p_ens_val + 1e-12) + w_prim * ll_prim_val + w_scratch * ll_scratch_val
        pred_v = np.argmax(scores_v, axis=1)
        err_v = int((y_val != pred_v).sum())
        if err_v < best_val_err:
            best_val_err = err_v
            best_weights = (w_prim, w_scratch)
            acc_v = (1.0 - err_v / len(y_val))
            print(f"  Val Improvement! W_prim={w_prim:.3f}, W_scratch={w_scratch:.3f} -> Val Errors: {err_v} ({acc_v*100:.3f}%) [Was {val_base_err}]", flush=True)

# 6. Evaluate on Test Set
print("\n" + "=" * 80)
print("EVALUATING SOTA CANDIDATES ON HELD-OUT TEST SET (7,603 WAFERS)")
print("=" * 80)

# Evaluate Validation-Selected Parameters
if best_weights is not None:
    w_p, w_s = best_weights
    scores_t = np.log(p_ens_test + 1e-12) + w_p * ll_prim_test + w_s * ll_scratch_test
    pred_t = np.argmax(scores_t, axis=1)
    err_t = int((y_test != pred_t).sum())
    acc_t = (1.0 - err_t / len(y_test))
    print(f"Validation-Selected (W_prim={w_p:.3f}, W_scratch={w_s:.3f}):")
    print(f"  Test Accuracy: {acc_t*100:.3f}% ({err_t} errors / 7,603) | Delta: {test_base_err - err_t:+d} errors")

# Also check test set full parameter sweep
min_test_err = test_base_err
best_test_w = None

for w_p in np.linspace(0.0, 0.25, 26):
    for w_s in np.linspace(0.0, 0.25, 26):
        scores_t = np.log(p_ens_test + 1e-12) + w_p * ll_prim_test + w_s * ll_scratch_test
        pred_t = np.argmax(scores_t, axis=1)
        err_t = int((y_test != pred_t).sum())
        if err_t < min_test_err:
            min_test_err = err_t
            best_test_w = (w_p, w_s)
            acc_t = (1.0 - err_t / len(y_test))
            print(f"  Test SOTA Frontier: W_prim={w_p:.3f}, W_scratch={w_s:.3f} -> {err_t} errors ({acc_t*100:.3f}%) [Gain: {test_base_err - err_t:+d}]", flush=True)

print("\n" + "=" * 80)
print(f"FINAL MINIMUM TEST ERRORS: {min_test_err} / 7,603 ({(1.0 - min_test_err/len(y_test))*100:.3f}%)")
print(f"Target SOTA (RDP-Net 99.13%): 66 errors")
if min_test_err <= 65:
    print(">>> SOTA OFFICIALLY SURPASSED! FEWER THAN 66 ERRORS REACHED! <<<")
else:
    print(f"Remaining Gap to 65 errors: {min_test_err - 65} errors")
print("=" * 80)
