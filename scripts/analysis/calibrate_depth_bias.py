import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("conformal_wafer"))

from conformal_wafer.src.train_optimized_sota import get_fast_loaders
from conformal_wafer.src.run_sota_campaign import DPCANet
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

loaders, n_test = get_fast_loaders()
val_loader = loaders['val']
test_loader = loaders['test']

dpcanet = DPCANet(in_channels=3, num_classes=38).to(DEVICE)
ckpt_d = torch.load("conformal_wafer/checkpoints/dpcanet-v2_80split_best.pt", map_location=DEVICE, weights_only=False)
dpcanet.load_state_dict(ckpt_d['model_state_dict'])
dpcanet.eval()

def get_d4_probs(loader):
    all_probs, targets = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            sum_p = 0.0
            for k in range(4):
                xk = torch.rot90(images, k, [2, 3])
                for flip in [False, True]:
                    x_v = torch.flip(xk, [3]) if flip else xk
                    with torch.amp.autocast('cuda', enabled=True):
                        ld = dpcanet(x_v)
                    sum_p = sum_p + F.softmax(ld, dim=1)
            all_probs.append((sum_p / 8.0).cpu().numpy())
            targets.append(labels.cpu().numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(targets, axis=0)

print("Computing D4-TTA on Val and Test...", flush=True)
p_val, y_val = get_d4_probs(val_loader)
p_test, y_test = get_d4_probs(test_loader)

# Baseline
acc_val_base = accuracy_score(y_val, np.argmax(p_val, axis=1))
acc_test_base = accuracy_score(y_test, np.argmax(p_test, axis=1))
err_test_base = int((y_test != np.argmax(p_test, axis=1)).sum())
print(f"Baseline Val Acc:  {acc_val_base*100:.3f}% ({int((y_val != np.argmax(p_val, axis=1)).sum())} errors / {len(y_val)})", flush=True)
print(f"Baseline Test Acc: {acc_test_base*100:.3f}% ({err_test_base} errors / {len(y_test)})", flush=True)

# Depth vector
depths = np.array([c[3] for c in CLASS_DEFINITIONS], dtype=np.float32) # (38,)
# Primitives matrix
vecs = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.float32) # (38, 8)
# Primitives: [C, D, EL, ER, L, NF, S, R]
# Indices: S=6, L=4, EL=2

print("\n--- Tuning Depth & Primitive Logit Calibration on Validation Set ---", flush=True)
best_val_acc = acc_val_base
best_params = None

# Grid search on validation set
# Bonus on depths: delta_logits = beta_depth * (depth - 1)
# Bonus on specific primitives: delta_logits += beta_s * vecs[:, 6] + beta_l * vecs[:, 4] + beta_el * vecs[:, 2]

for b_depth in np.linspace(0.0, 0.4, 9):
    for b_s in np.linspace(0.0, 0.4, 9):
        for b_l in np.linspace(0.0, 0.3, 7):
            boost = b_depth * (depths - 1.0) + b_s * vecs[:, 6] + b_l * vecs[:, 4]
            # In log-probability space:
            log_p_val = np.log(p_val + 1e-12) + boost[None, :]
            pred_val = np.argmax(log_p_val, axis=1)
            acc_val = accuracy_score(y_val, pred_val)
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                best_params = (b_depth, b_s, b_l)
                err_val = int((y_val != pred_val).sum())
                print(f"  Val Improvement! b_depth={b_depth:.2f}, b_s={b_s:.2f}, b_l={b_l:.2f} -> Val Errors: {err_val} ({acc_val*100:.3f}%)", flush=True)

if best_params is not None:
    b_depth, b_s, b_l = best_params
    boost = b_depth * (depths - 1.0) + b_s * vecs[:, 6] + b_l * vecs[:, 4]
    log_p_test = np.log(p_test + 1e-12) + boost[None, :]
    pred_test = np.argmax(log_p_test, axis=1)
    err_test = int((y_test != pred_test).sum())
    acc_test = accuracy_score(y_test, pred_test)
    print(f"\n--- EVALUATING OPTIMAL PARAMS ON TEST SET ---", flush=True)
    print(f"Test Accuracy: {acc_test*100:.3f}% ({err_test} errors / 7,603)", flush=True)
    print(f"Errors reduced from {err_test_base} -> {err_test} (gain of {err_test_base - err_test} wafers!)", flush=True)
else:
    print("No parameter improved validation accuracy.", flush=True)
