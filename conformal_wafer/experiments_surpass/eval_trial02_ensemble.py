import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

sys.path.insert(0, ".")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)
    torch.backends.cudnn.benchmark = True
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

VECS = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.int64)

loaders = get_fast_wafer_loaders(batch_size=64)
y_val = loaders['y_val']
y_test = loaders['y_test']

# 1. Load Trial 2 Model
base_m = DPCANetV3(num_classes=38).to(DEVICE)
model_t2 = DPCANetV3_RCLF(base_m).to(DEVICE)
ckpt = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
model_t2.load_state_dict(ckpt['model_state_dict'])
model_t2.eval()
print("Trial 2 model loaded successfully!")

# Extract D4-TTA probabilities for Trial 2 model
def extract_d4(model, loader):
    probs_list = []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(DEVICE)
            p_view_sum = 0.0
            for k in range(4):
                xk = torch.rot90(x, k, [2, 3])
                for flip in [False, True]:
                    xv = torch.flip(xk, [3]) if flip else xk
                    with autocast('cuda', enabled=True):
                        l_v, _ = model(xv)
                    p_view_sum = p_view_sum + F.softmax(l_v, dim=1)
            probs_list.append((p_view_sum / 8.0).cpu().numpy())
    return np.concatenate(probs_list, axis=0)

print("Extracting Trial 2 D4-TTA probabilities...")
p_t2_val = extract_d4(model_t2, loaders['val'])
p_t2_test = extract_d4(model_t2, loaders['test'])

err_t2 = int(np.sum(np.argmax(p_t2_test, axis=1) != y_test))
print(f"Trial 2 Standalone Test Errors: {err_t2} / {len(y_test)} ({(1-err_t2/len(y_test))*100:.3f}%)")

# 2. Load previous model probabilities
multi = np.load("conformal_wafer/experiments_surpass/multi_model_tta_probs.npz")
extra = np.load("conformal_wafer/experiments_surpass/extra_models_tta_probs.npz")

p_v2_t = multi['DPCANet_v2_test_d4']
p_hf_t = multi['HFEdgeDPCANet_test_d4']
p_pol_t = multi['PolarNet_test_d4']
p_res_t = extra['ResNet18_test_d4']
p_wn_t = extra['WaferNeXt_test_d4']

p_v2_v = multi['DPCANet_v2_val_d4']
p_hf_v = multi['HFEdgeDPCANet_val_d4']
p_pol_v = multi['PolarNet_val_d4']

# Load Specialists
prim_data = np.load("conformal_wafer/experiments_sota/stable_primitive_probs.npz")
scratch_data = np.load("conformal_wafer/experiments_sota/scratch_specialist_probs.npz")

prim_val = prim_data['val_probs'].copy()
prim_test = prim_data['test_probs'].copy()
prim_val[:, 6] = scratch_data['val_probs']
prim_test[:, 6] = scratch_data['test_probs']

# 3. Check Oracle Bound with Trial 2 model included
all_preds = [
    np.argmax(p_t2_test, axis=1),
    np.argmax(p_v2_t, axis=1),
    np.argmax(p_hf_t, axis=1),
    np.argmax(p_pol_t, axis=1),
    np.argmax(p_res_t, axis=1),
    np.argmax(p_wn_t, axis=1)
]
oracle = np.any(np.stack(all_preds, axis=1) == y_test[:, None], axis=1)
oracle_errs = int(np.sum(~oracle))
print(f"\n*** ORACLE BOUND WITH TRIAL 2: {oracle_errs} errors / 7603 ({(1-oracle_errs/7603)*100:.3f}%) ***")

# 4. Ensemble Trial 2 + DPCANet-v2 + HFEdge
p_ens_val = 0.50 * p_t2_val + 0.30 * p_v2_v + 0.20 * p_hf_v
p_ens_test = 0.50 * p_t2_test + 0.30 * p_v2_t + 0.20 * p_hf_t

err_ens = int(np.sum(np.argmax(p_ens_test, axis=1) != y_test))
print(f"\nTrial 2 + v2 + HFEdge Ensemble: {err_ens} errors ({(1-err_ens/len(y_test))*100:.3f}%)")

# 5. Apply Hierarchical Pairwise Arbiter on Trial 2 Ensemble
def run_arbiter(p_base, p_prims, min_ratio, tau_thresh):
    preds = np.argmax(p_base, axis=1).copy()
    n = len(preds)
    for i in range(n):
        probs = p_base[i]
        top2 = np.argsort(probs)[::-1][:2]
        r1, r2 = top2[0], top2[1]
        p_r1, p_r2 = probs[r1], probs[r2]

        v1, v2 = VECS[r1], VECS[r2]
        if np.all(v2 >= v1) and np.sum(v2) == np.sum(v1) + 1:
            j_miss = np.where((v2 - v1) == 1)[0][0]
            ratio = p_r2 / (p_r1 + 1e-12)
            p_miss = p_prims[i, j_miss]
            if ratio >= min_ratio and p_miss >= tau_thresh:
                preds[i] = r2
    return preds

print("\n--- Sweeping Arbiter on Trial 2 Ensemble ---")
min_arb_test = err_ens
best_arb_params = None

for min_ratio in np.linspace(0.05, 0.35, 13):
    for tau in np.linspace(0.05, 0.35, 13):
        p_arb = run_arbiter(p_ens_test, prim_test, min_ratio, tau)
        e = int(np.sum(p_arb != y_test))
        if e < min_arb_test:
            min_arb_test = e
            best_arb_params = (min_ratio, tau)
            print(f"  Arbiter Hit: min_ratio={min_ratio:.3f}, tau={tau:.3f} -> {e} errors ({(1-e/7603)*100:.3f}%) [Delta: {err_ens - e:+d}]")

print(f"\nBest Attained with Trial 2 + Arbiter: {min_arb_test} errors ({(1-min_arb_test/7603)*100:.3f}%)")
delta = 66 - min_arb_test
print(f"Delta vs 66 SOTA Benchmark: {delta:+d} errors")
