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

print(f"Total Errors: {len(err_idx)}")
print(f"{'True Class':15s} | {'Pred Class (Prob)':22s} | {'Rank of True':12s} | {'True Prob':10s} | {'Margin':8s}")
print("-" * 75)

rank_1_correct = 0
rank_2_true = 0
rank_3_true = 0
for idx in err_idx[:30]:
    t_idx = y_true[idx]
    p_idx = preds[idx]
    t_name = class_names[t_idx]
    p_name = class_names[p_idx]
    
    probs = p_d[idx]
    sorted_classes = np.argsort(probs)[::-1]
    rank_true = int(np.where(sorted_classes == t_idx)[0][0]) + 1
    
    prob_pred = probs[p_idx]
    prob_true = probs[t_idx]
    margin = prob_pred - prob_true
    
    if rank_true == 2:
        rank_2_true += 1
    elif rank_true == 3:
        rank_3_true += 1
        
    print(f"{t_name:15s} | {p_name} ({prob_pred*100:.1f}%) | Rank {rank_true:2d}       | {prob_true*100:6.1f}%    | {margin*100:5.1f}%")

all_ranks = []
for idx in err_idx:
    t_idx = y_true[idx]
    sorted_classes = np.argsort(p_d[idx])[::-1]
    all_ranks.append(int(np.where(sorted_classes == t_idx)[0][0]) + 1)

all_ranks = np.array(all_ranks)
print("-" * 75)
print(f"Across all {len(err_idx)} errors:")
print(f"  True class is Rank 2 (runner up): {(all_ranks == 2).sum()} / {len(err_idx)} ({(all_ranks == 2).mean()*100:.1f}%)")
print(f"  True class is Rank 3:             {(all_ranks == 3).sum()} / {len(err_idx)} ({(all_ranks == 3).mean()*100:.1f}%)")
print(f"  True class in Top 3:              {(all_ranks <= 3).sum()} / {len(err_idx)} ({(all_ranks <= 3).mean()*100:.1f}%)")
