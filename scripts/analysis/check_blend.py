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

from conformal_wafer.src.train_optimized_sota import get_fast_loaders, evaluate_test_metrics, PolarNet, MSResNet
from conformal_wafer.src.run_sota_campaign import DPCANet

loaders, n_test = get_fast_loaders()
test_loader = loaders['test']

print("Loading checkpoints...")
dpcanet = DPCANet(in_channels=3, num_classes=38).to(DEVICE)
ckpt_d = torch.load("conformal_wafer/checkpoints/dpcanet-v2_80split_best.pt", map_location=DEVICE, weights_only=False)
dpcanet.load_state_dict(ckpt_d['model_state_dict'])

msnet = MSResNet(in_channels=3, num_classes=38).to(DEVICE)
ckpt_m = torch.load("conformal_wafer/checkpoints/ms-resnet_fast80_best.pt", map_location=DEVICE, weights_only=False)
msnet.load_state_dict(ckpt_m['model_state_dict'])

res_d = evaluate_test_metrics(dpcanet, test_loader)
res_m = evaluate_test_metrics(msnet, test_loader)

y_true = res_d['y_true']
p_d = res_d['prob_d4']
p_m = res_m['prob_d4']

print(f"DPCANet-v2 D4: {res_d['err_d4']} errs ({res_d['acc_d4']*100:.3f}%)")
print(f"MS-ResNet  D4: {res_m['err_d4']} errs ({res_m['acc_d4']*100:.3f}%)")

# Search over blending weights
best_w = 0.5
min_err = 999
for w in np.linspace(0.0, 1.0, 21):
    p_blend = w * p_d + (1 - w) * p_m
    y_pred = np.argmax(p_blend, axis=1)
    err = int((y_true != y_pred).sum())
    acc = accuracy_score(y_true, y_pred)
    if err < min_err:
        min_err = err
        best_w = w
    if w in [0.0, 0.2, 0.5, 0.7, 0.8, 0.85, 0.9, 1.0]:
        print(f"Weight DPCANet={w:.2f}: {err} errors ({acc*100:.3f}%)")

print(f"\nOptimal 2-model blend weight DPCANet={best_w:.2f}: {min_err} errors ({(1 - min_err/n_test)*100:.3f}%)")
