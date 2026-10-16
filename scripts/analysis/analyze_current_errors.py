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

from conformal_wafer.src.train_optimized_sota import get_fast_loaders, evaluate_test_metrics
from conformal_wafer.src.run_sota_campaign import DPCANet
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

loaders, n_test = get_fast_loaders()
test_loader = loaders['test']

dpcanet = DPCANet(in_channels=3, num_classes=38).to(DEVICE)
ckpt_d = torch.load("conformal_wafer/checkpoints/dpcanet-v2_80split_best.pt", map_location=DEVICE, weights_only=False)
dpcanet.load_state_dict(ckpt_d['model_state_dict'])

res_dpca = evaluate_test_metrics(dpcanet, test_loader)
y_true = res_dpca['y_true']
pred_dpca = np.argmax(res_dpca['prob_d4'], axis=1)
errors_dpca = np.where(pred_dpca != y_true)[0]

class_names = [c[1] for c in CLASS_DEFINITIONS]

print("\n--- EXACT CONFUSION FOR MISCLASSIFIED SAMPLES ---")
confusion_counts = {}
for idx in errors_dpca:
    t = class_names[y_true[idx]]
    p = class_names[pred_dpca[idx]]
    pair = f"{t} -> {p}"
    confusion_counts[pair] = confusion_counts.get(pair, 0) + 1

for pair, cnt in sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
    print(f"  {pair:25s}: {cnt}")
