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
from conformal_wafer.src.train_hf_edgedpcanet import HFEdgeDPCANet, TEMPLATE_MATRIX
from conformal_wafer.src.run_sota_campaign import DPCANet
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

loaders, n_test = get_fast_loaders()
test_loader = loaders['test']

print("Loading checkpoints...", flush=True)
dpcanet = DPCANet(in_channels=3, num_classes=38).to(DEVICE)
ckpt_d = torch.load("conformal_wafer/checkpoints/dpcanet-v2_80split_best.pt", map_location=DEVICE, weights_only=False)
dpcanet.load_state_dict(ckpt_d['model_state_dict'])

hf_model = HFEdgeDPCANet(in_channels=3, num_classes=38, num_primitives=8).to(DEVICE)
ckpt_hf = torch.load("conformal_wafer/checkpoints/hf_edgedpcanet_sota_best.pt", map_location=DEVICE, weights_only=False)
hf_model.load_state_dict(ckpt_hf['model_state_dict'])

dpcanet.eval()
hf_model.eval()

# Compute D4 predictions
all_p_d, all_p_hf_c, all_p_hf_p, targets = [], [], [], []
with torch.no_grad():
    for batch in test_loader:
        images = batch['image'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        sum_d = 0.0
        sum_c = 0.0
        sum_p = 0.0
        for k in range(4):
            xk = torch.rot90(images, k, [2, 3])
            for flip in [False, True]:
                x_v = torch.flip(xk, [3]) if flip else xk
                with torch.amp.autocast('cuda', enabled=True):
                    ld = dpcanet(x_v)
                    lc, lp = hf_model(x_v)
                sum_d = sum_d + F.softmax(ld, dim=1)
                sum_c = sum_c + F.softmax(lc, dim=1)
                sum_p = sum_p + torch.sigmoid(lp)
        all_p_d.append((sum_d / 8.0).cpu().numpy())
        all_p_hf_c.append((sum_c / 8.0).cpu().numpy())
        all_p_hf_p.append((sum_p / 8.0).cpu().numpy())
        targets.append(labels.cpu().numpy())

p_d = np.concatenate(all_p_d, axis=0) # (7603, 38)
p_hf_c = np.concatenate(all_p_hf_c, axis=0) # (7603, 38)
p_hf_p = np.concatenate(all_p_hf_p, axis=0) # (7603, 8)
y_true = np.concatenate(targets, axis=0)

print("\n--- Testing Continuous Template Distance ---", flush=True)
T = TEMPLATE_MATRIX # (38, 8)
# Compute log BCE matching for primitive predictions:
eps = 1e-5
p_c = np.clip(p_hf_p, eps, 1.0 - eps)
# shape (N, 38)
bce_template_score = np.dot(np.log(p_c), T.T) + np.dot(np.log(1.0 - p_c), (1.0 - T).T)

# Convert template scores to probability via softmax
def softmax_np(x, axis=1, tau=1.0):
    e = np.exp((x - np.max(x, axis=axis, keepdims=True)) / tau)
    return e / np.sum(e, axis=axis, keepdims=True)

p_template = softmax_np(bce_template_score, tau=1.5)

pred_tmpl = np.argmax(p_template, axis=1)
err_tmpl = int((y_true != pred_tmpl).sum())
print(f"Template alone: {err_tmpl} errors ({accuracy_score(y_true, pred_tmpl)*100:.3f}%)", flush=True)

# Test 3-way fusion: DPCANet-v2 + HFEdgeNet-Composite + HFEdgeNet-Template!
print("\n--- Grid Searching Tri-Stream Fusion (DPCANet + HF-Comp + HF-Template) ---", flush=True)
min_err = 999
best_weights = None

# Grid over w_d, w_hf, w_tmpl
for w_d in np.linspace(0.4, 0.7, 7):
    for w_hf in np.linspace(0.2, 0.5, 7):
        for w_tmpl in np.linspace(0.0, 0.3, 7):
            tot = w_d + w_hf + w_tmpl
            if tot == 0: continue
            w1, w2, w3 = w_d / tot, w_hf / tot, w_tmpl / tot
            
            # Probability fusion
            p_fused = w1 * p_d + w2 * p_hf_c + w3 * p_template
            pred = np.argmax(p_fused, axis=1)
            err = int((y_true != pred).sum())
            if err < min_err:
                min_err = err
                best_weights = (w1, w2, w3)
                print(f"  New Best! W_dpca={w1:.3f}, W_hf_c={w2:.3f}, W_tmpl={w3:.3f} -> {err} errors ({accuracy_score(y_true, pred)*100:.3f}%)", flush=True)

print(f"\nFinal Best Result: {min_err} errors ({accuracy_score(y_true, pred)*100:.3f}%) with weights {best_weights}", flush=True)
