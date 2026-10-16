import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

sys.path.insert(0, ".")
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)
    torch.cuda.empty_cache()
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3

def eval_trial_02():
    loaders = get_fast_wafer_loaders(batch_size=64)
    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    model = DPCANetV3_RCLF(base_m).to(DEVICE)

    ckpt = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print("Evaluating Trial 2 with 8-view D4-TTA on Val & Test...", flush=True)

    # Val D4-TTA
    v_probs = []
    with torch.no_grad():
        for x, y, _ in loaders['val']:
            x = x.to(DEVICE)
            p_cpu = np.zeros((len(x), 38), dtype=np.float32)
            for k in range(4):
                xk = torch.rot90(x, k, [2, 3])
                for flip in [False, True]:
                    xv = torch.flip(xk, [3]) if flip else xk
                    with autocast('cuda', enabled=True):
                        l, _ = model(xv)
                    p_cpu += F.softmax(l, dim=1).cpu().numpy()
            v_probs.append(p_cpu / 8.0)
    v_probs = np.concatenate(v_probs, axis=0)

    # Test D4-TTA
    t_probs = []
    with torch.no_grad():
        for x, y, _ in loaders['test']:
            x = x.to(DEVICE)
            p_cpu = np.zeros((len(x), 38), dtype=np.float32)
            for k in range(4):
                xk = torch.rot90(x, k, [2, 3])
                for flip in [False, True]:
                    xv = torch.flip(xk, [3]) if flip else xk
                    with autocast('cuda', enabled=True):
                        l, _ = model(xv)
                    p_cpu += F.softmax(l, dim=1).cpu().numpy()
            t_probs.append(p_cpu / 8.0)
    t_probs = np.concatenate(t_probs, axis=0)

    y_val = loaders['y_val']
    y_test = loaders['y_test']

    val_errs = int(np.sum(np.argmax(v_probs, axis=1) != y_val))
    test_errs = int(np.sum(np.argmax(t_probs, axis=1) != y_test))

    print(f"Trial 2 D4-TTA Val Errors: {val_errs} / {len(y_val)} ({(1 - val_errs/len(y_val))*100:.3f}%)", flush=True)
    print(f"Trial 2 D4-TTA Test Errors: {test_errs} / {len(y_test)} ({(1 - test_errs/len(y_test))*100:.3f}%)", flush=True)

    # Save probabilities
    np.savez_compressed("conformal_wafer/experiments_surpass/trial02_d4_probs.npz",
                        val_probs=v_probs, test_probs=t_probs, y_val=y_val, y_test=y_test)
    print("Saved trial02_d4_probs.npz successfully.", flush=True)

if __name__ == '__main__':
    eval_trial_02()
