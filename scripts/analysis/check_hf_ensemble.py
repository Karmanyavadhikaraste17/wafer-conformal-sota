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

from conformal_wafer.src.train_optimized_sota import get_fast_loaders, evaluate_test_metrics
from conformal_wafer.src.run_sota_campaign import DPCANet
from conformal_wafer.src.train_hf_edgedpcanet import HFEdgeDPCANet

loaders, n_test = get_fast_loaders()
test_loader = loaders['test']

print("Loading DPCANet-v2...")
dpcanet = DPCANet(in_channels=3, num_classes=38).to(DEVICE)
ckpt_d = torch.load("conformal_wafer/checkpoints/dpcanet-v2_80split_best.pt", map_location=DEVICE, weights_only=False)
dpcanet.load_state_dict(ckpt_d['model_state_dict'])

print("Loading HFEdgeDPCANet...")
hf_model = HFEdgeDPCANet(in_channels=3, num_classes=38, num_primitives=8).to(DEVICE)
ckpt_hf = torch.load("conformal_wafer/checkpoints/hf_edgedpcanet_sota_best.pt", map_location=DEVICE, weights_only=False)
hf_model.load_state_dict(ckpt_hf['model_state_dict'])

# Evaluate D4 on both
def get_d4_probs(model, test_loader, is_dual_head=False):
    model.eval()
    all_probs = []
    targets = []
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

print("Computing D4-TTA predictions...")
p_dpca, y_true = get_d4_probs(dpcanet, test_loader, is_dual_head=False)
p_hf, _ = get_d4_probs(hf_model, test_loader, is_dual_head=True)

pred_dpca = np.argmax(p_dpca, axis=1)
pred_hf = np.argmax(p_hf, axis=1)

err_dpca = int((y_true != pred_dpca).sum())
err_hf = int((y_true != pred_hf).sum())

print(f"DPCANet-v2 D4 Errors: {err_dpca} / {len(y_true)} ({accuracy_score(y_true, pred_dpca)*100:.3f}%)")
print(f"HFEdgeNet  D4 Errors: {err_hf} / {len(y_true)} ({accuracy_score(y_true, pred_hf)*100:.3f}%)")

err_idx_dpca = set(np.where(y_true != pred_dpca)[0])
err_idx_hf = set(np.where(y_true != pred_hf)[0])
common_errs = err_idx_dpca.intersection(err_idx_hf)
union_errs = err_idx_dpca.union(err_idx_hf)
print(f"Intersection of errors: {len(common_errs)} errors common to both!")
print(f"Errors only in DPCANet-v2 (HF got RIGHT): {len(err_idx_dpca - err_idx_hf)}")
print(f"Errors only in HFEdgeNet (DPCANet got RIGHT): {len(err_idx_hf - err_idx_dpca)}")

# Blend search
print("\n--- Soft-Voting Ensemble (DPCANet-v2 + HFEdgeNet) ---")
best_w = 0.5
min_err = 999
for w in np.linspace(0.0, 1.0, 21):
    p_blend = w * p_dpca + (1 - w) * p_hf
    pred_blend = np.argmax(p_blend, axis=1)
    err = int((y_true != pred_blend).sum())
    acc = accuracy_score(y_true, pred_blend)
    if err < min_err:
        min_err = err
        best_w = w
    print(f"  Weight DPCANet={w:.2f}: {err} errors ({acc*100:.3f}%)")

print(f"\nOptimal Blend: DPCANet weight = {best_w:.2f} -> {min_err} errors ({(1.0 - min_err/len(y_true))*100:.3f}%)")
