import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("conformal_wafer"))

from conformal_wafer.src.train_optimized_sota import get_fast_loaders
from conformal_wafer.src.train_hf_edgedpcanet import HFEdgeDPCANet, TEMPLATE_MATRIX
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

loaders, n_test = get_fast_loaders()
test_loader = loaders['test']

hf_model = HFEdgeDPCANet(in_channels=3, num_classes=38, num_primitives=8).to(DEVICE)
ckpt_hf = torch.load("conformal_wafer/checkpoints/hf_edgedpcanet_sota_best.pt", map_location=DEVICE, weights_only=False)
hf_model.load_state_dict(ckpt_hf['model_state_dict'])
hf_model.eval()

all_lc, all_lp, targets = [], [], []
with torch.no_grad():
    for batch in test_loader:
        images = batch['image'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        sum_c = 0.0
        sum_p = 0.0
        for k in range(4):
            xk = torch.rot90(images, k, [2, 3])
            for flip in [False, True]:
                x_v = torch.flip(xk, [3]) if flip else xk
                with torch.amp.autocast('cuda', enabled=True):
                    lc, lp = hf_model(x_v)
                sum_c = sum_c + F.softmax(lc, dim=1)
                sum_p = sum_p + torch.sigmoid(lp)
        all_lc.append((sum_c / 8.0).cpu().numpy())
        all_lp.append((sum_p / 8.0).cpu().numpy())
        targets.append(labels.cpu().numpy())

p_comp = np.concatenate(all_lc, axis=0) # (7603, 38)
p_prim = np.concatenate(all_lp, axis=0) # (7603, 8)
y_true = np.concatenate(targets, axis=0)

prim_names = ['C', 'D', 'EL', 'ER', 'L', 'NF', 'S', 'R']
class_names = [c[1] for c in CLASS_DEFINITIONS]

# Check primitive thresholding vs composite argmax
# For each sample, match thresholded primitive vector to nearest template row
vec_to_class = {tuple(TEMPLATE_MATRIX[i].astype(int)): i for i in range(38)}

# Test primitive-derived classification
print("\n--- Testing Pure Primitive Thresholding ---")
for thresh in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
    prim_bin = (p_prim >= thresh).astype(int)
    matches = 0
    mapped_count = 0
    for i in range(len(y_true)):
        tup = tuple(prim_bin[i])
        if tup in vec_to_class:
            mapped_count += 1
            if vec_to_class[tup] == y_true[i]:
                matches += 1
    print(f"  Thresh={thresh:.2f}: {matches} / {len(y_true)} correct ({matches/len(y_true)*100:.2f}%) [Valid composite: {mapped_count}/{len(y_true)}]")

# Test distance to templates
# Continuous template cosine similarity or negative BCE distance
print("\n--- Testing Continuous Template Distance ---")
# For each sample i, find k = argmax_k [ sum_j (p_prim[i, j] * T[k, j] - (1 - p_prim[i, j]) * (1 - T[k, j])) ]
# or dot product / L2 / BCE
for metric in ['dot', 'bce', 'cosine']:
    pred_from_prim = []
    for i in range(len(y_true)):
        pi = p_prim[i]
        if metric == 'dot':
            # Score: 2 * (pi - 0.5) dot (2 * T - 1)
            scores = np.dot(2.0 * pi - 1.0, (2.0 * TEMPLATE_MATRIX - 1.0).T)
        elif metric == 'cosine':
            # Cosine similarity
            norm_pi = np.linalg.norm(pi) + 1e-7
            norm_T = np.linalg.norm(TEMPLATE_MATRIX, axis=1) + 1e-7
            scores = np.dot(pi, TEMPLATE_MATRIX.T) / (norm_pi * norm_T)
        elif metric == 'bce':
            eps = 1e-4
            p_c = np.clip(pi, eps, 1.0 - eps)
            scores = np.dot(np.log(p_c), TEMPLATE_MATRIX.T) + np.dot(np.log(1.0 - p_c), (1.0 - TEMPLATE_MATRIX).T)
        pred_from_prim.append(np.argmax(scores))
    pred_from_prim = np.array(pred_from_prim)
    err = int((y_true != pred_from_prim).sum())
    print(f"  Metric {metric:6s}: {err} errors ({accuracy_score(y_true, pred_from_prim)*100:.3f}%)")
