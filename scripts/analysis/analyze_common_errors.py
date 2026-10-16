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
from conformal_wafer.src.train_hf_edgedpcanet import HFEdgeDPCANet
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

loaders, n_test = get_fast_loaders()
test_loader = loaders['test']

dpcanet = DPCANet(in_channels=3, num_classes=38).to(DEVICE)
ckpt_d = torch.load("conformal_wafer/checkpoints/dpcanet-v2_80split_best.pt", map_location=DEVICE, weights_only=False)
dpcanet.load_state_dict(ckpt_d['model_state_dict'])

hf_model = HFEdgeDPCANet(in_channels=3, num_classes=38, num_primitives=8).to(DEVICE)
ckpt_hf = torch.load("conformal_wafer/checkpoints/hf_edgedpcanet_sota_best.pt", map_location=DEVICE, weights_only=False)
hf_model.load_state_dict(ckpt_hf['model_state_dict'])

def get_d4_probs(model, is_dual_head=False):
    model.eval()
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
                        out = model(x_v)
                        logits = out[0] if is_dual_head else out
                    sum_p = sum_p + F.softmax(logits, dim=1)
            all_probs.append((sum_p / 8.0).cpu().numpy())
            targets.append(labels.cpu().numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(targets, axis=0)

p_dpca, y_true = get_d4_probs(dpcanet, is_dual_head=False)
p_hf, _ = get_d4_probs(hf_model, is_dual_head=True)

pred_dpca = np.argmax(p_dpca, axis=1)
pred_hf = np.argmax(p_hf, axis=1)

err_idx_dpca = set(np.where(y_true != pred_dpca)[0])
err_idx_hf = set(np.where(y_true != pred_hf)[0])
common = sorted(list(err_idx_dpca.intersection(err_idx_hf)))

class_names = [c[1] for c in CLASS_DEFINITIONS]
print(f"Total Common Errors: {len(common)}")
common_by_class = {}
for idx in common:
    t = class_names[y_true[idx]]
    p1 = class_names[pred_dpca[idx]]
    p2 = class_names[pred_hf[idx]]
    common_by_class[t] = common_by_class.get(t, []) + [(p1, p2)]

for cname, pairs in sorted(common_by_class.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"  {cname:15s} ({len(pairs):2d} errors):")
    for p1, p2 in pairs[:5]:
        print(f"      DPCANet: {p1:12s} | HFEdgeNet: {p2:12s}")
