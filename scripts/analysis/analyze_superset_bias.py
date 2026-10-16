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
from conformal_wafer.src.run_sota_campaign import DPCANet
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

loaders, n_test = get_fast_loaders()
test_loader = loaders['test']

dpcanet = DPCANet(in_channels=3, num_classes=38).to(DEVICE)
ckpt_d = torch.load("conformal_wafer/checkpoints/dpcanet-v2_80split_best.pt", map_location=DEVICE, weights_only=False)
dpcanet.load_state_dict(ckpt_d['model_state_dict'])
dpcanet.eval()

all_probs, targets = [], []
with torch.no_grad():
    for batch in test_loader:
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

p_d = np.concatenate(all_probs, axis=0)
y_true = np.concatenate(targets, axis=0)
preds = np.argmax(p_d, axis=1)

err_idx = np.where(y_true != preds)[0]
class_names = [c[1] for c in CLASS_DEFINITIONS]
vecs = np.array([c[2] for c in CLASS_DEFINITIONS]) # (38, 8)
prim_names = ['C', 'D', 'EL', 'ER', 'L', 'NF', 'S', 'R']

superset_count = 0
subset_count = 0
other_count = 0

print("Relationship between Predicted (Rank 1) and True (Rank 2/3):")
diff_prims_missed = {}
for idx in err_idx:
    t_idx = y_true[idx]
    p_idx = preds[idx]
    
    vt = vecs[t_idx]
    vp = vecs[p_idx]
    
    # Check if True is superset of Pred (Pred missed a primitive in True)
    if np.all(vt >= vp) and np.sum(vt) > np.sum(vp):
        superset_count += 1
        missed = [prim_names[j] for j in range(8) if vt[j] == 1 and vp[j] == 0]
        for m in missed:
            diff_prims_missed[m] = diff_prims_missed.get(m, 0) + 1
    elif np.all(vp >= vt) and np.sum(vp) > np.sum(vt):
        subset_count += 1
    else:
        other_count += 1

print(f"Total Errors: {len(err_idx)}")
print(f"  True is SUPERSET of Pred (Pred MISSED a defect): {superset_count} / {len(err_idx)} ({superset_count/len(err_idx)*100:.1f}%)")
print(f"  True is SUBSET of Pred (Pred OVER-DETECTED):     {subset_count} / {len(err_idx)} ({subset_count/len(err_idx)*100:.1f}%)")
print(f"  Other (defect substitution, e.g. EL vs ER):     {other_count} / {len(err_idx)} ({other_count/len(err_idx)*100:.1f}%)")

print("\nMissing primitives when True is Superset of Pred:")
for p, cnt in sorted(diff_prims_missed.items(), key=lambda x: x[1], reverse=True):
    print(f"  {p:5s}: {cnt:2d} times")
