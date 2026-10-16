import os
import sys
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "conformal_wafer"))

from conformal_wafer.src.dataset import CLASS_DEFINITIONS

prob_path = os.path.join(BASE_DIR, "conformal_wafer", "experiments_sota", "scratch_specialist_probs.npz")
probs_data = np.load(prob_path)
s_test = probs_data['test_probs']
y_test_scratch = probs_data['y_test']

# Load base predictions
import torch
import torch.nn.functional as F
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

from conformal_wafer.src.train_optimized_sota import get_fast_loaders
from conformal_wafer.src.run_sota_campaign import DPCANet
from conformal_wafer.src.train_hf_edgedpcanet import HFEdgeDPCANet

loaders, n_test = get_fast_loaders()

m1 = DPCANet(in_channels=3, num_classes=38).to(DEVICE)
m1.load_state_dict(torch.load("conformal_wafer/checkpoints/dpcanet-v2_80split_best.pt", map_location=DEVICE, weights_only=False)['model_state_dict'])
m1.eval()

m2 = HFEdgeDPCANet(in_channels=3, num_classes=38, num_primitives=8).to(DEVICE)
m2.load_state_dict(torch.load("conformal_wafer/checkpoints/hf_edgedpcanet_sota_best.pt", map_location=DEVICE, weights_only=False)['model_state_dict'])
m2.eval()

p1, p2, targets = [], [], []
with torch.no_grad():
    for batch in loaders['test']:
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

p_ens = 0.55 * np.concatenate(p1, axis=0) + 0.45 * np.concatenate(p2, axis=0)
y_true = np.concatenate(targets, axis=0)
preds = np.argmax(p_ens, axis=1)

err_idx = np.where(y_true != preds)[0]
print(f"Total Errors in Base Ensemble: {len(err_idx)}")

names = [c[1] for c in CLASS_DEFINITIONS]
vecs = [c[2] for c in CLASS_DEFINITIONS]

print(f"{'Idx':5s} | {'True Class':14s} | {'Pred Class':14s} | {'True S?':7s} | {'Pred S?':7s} | {'Scratch Prob':12s} | {'Top-3 Preds'}")
print("-" * 90)

scratch_detected_in_missed = 0
for idx in err_idx:
    t = y_true[idx]
    p = preds[idx]
    t_name = names[t]
    p_name = names[p]
    t_has_s = vecs[t][6] == 1
    p_has_s = vecs[p][6] == 1
    s_p = s_test[idx]
    
    top3 = np.argsort(p_ens[idx])[::-1][:3]
    top3_str = ", ".join([f"{names[c]}({p_ens[idx, c]*100:.1f}%)" for c in top3])
    
    if t_has_s and not p_has_s and s_p >= 0.5:
        scratch_detected_in_missed += 1
        tag = "[OPPORTUNITY!]"
    else:
        tag = ""
        
    print(f"{idx:5d} | {t_name:14s} | {p_name:14s} | {str(t_has_s):7s} | {str(p_has_s):7s} | {s_p*100:6.2f}%      | {top3_str} {tag}")

print("-" * 90)
print(f"Total True-Scratch samples where Model Missed Scratch BUT Scratch Specialist detected S (>=50%): {scratch_detected_in_missed} / {len(err_idx)}")
