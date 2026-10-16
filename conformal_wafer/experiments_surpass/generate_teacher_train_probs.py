import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Generating teacher probabilities on device: {DEVICE}", flush=True)

from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.src.train_optimized_sota import PolarNet
from conformal_wafer.src.train_wafernext import WaferNeXt

OUT_DIR = "conformal_wafer/experiments_surpass"
os.makedirs(OUT_DIR, exist_ok=True)

# Fast non-augmented loader for extracting teacher probs
class EvalTensorDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors
    def __len__(self):
        return len(self.tensors)
    def __getitem__(self, idx):
        return self.tensors[idx]

def extract_teacher_probs():
    loaders = get_fast_wafer_loaders(batch_size=256)
    train_tensors = loaders['train'].dataset.tensors # (27370, 3, 52, 52)
    val_tensors = loaders['val'].dataset.tensors     # (3042, 3, 52, 52)
    test_tensors = loaders['test'].dataset.tensors   # (7603, 3, 52, 52)

    eval_train_loader = DataLoader(EvalTensorDataset(train_tensors), batch_size=256, shuffle=False)
    eval_val_loader = DataLoader(EvalTensorDataset(val_tensors), batch_size=256, shuffle=False)
    eval_test_loader = DataLoader(EvalTensorDataset(test_tensors), batch_size=256, shuffle=False)

    # 1. Load Teachers
    m_v3 = DPCANetV3(num_classes=38).to(DEVICE)
    c_v3 = torch.load('conformal_wafer/experiments_sota/dpcanet_v3_best.pt', map_location=DEVICE, weights_only=False)
    m_v3.load_state_dict(c_v3['model_state_dict'])
    m_v3.eval()

    m_pol = PolarNet(num_classes=38).to(DEVICE)
    c_pol = torch.load('conformal_wafer/checkpoints/polarnet-v2_fast80_best.pt', map_location=DEVICE, weights_only=False)
    m_pol.load_state_dict(c_pol['model_state_dict'])
    m_pol.eval()

    m_wn = WaferNeXt(dims=[40, 64, 120, 192]).to(DEVICE)
    c_wn = torch.load('conformal_wafer/checkpoints/wafernext_compact_best.pt', map_location=DEVICE, weights_only=False)
    m_wn.load_state_dict(c_wn['model_state_dict'])
    m_wn.eval()

    def get_probs(model, loader):
        probs = []
        with torch.no_grad():
            for x in loader:
                x = x.to(DEVICE)
                with autocast('cuda', enabled=True):
                    out = model(x)
                    l = out[0] if isinstance(out, tuple) else out
                    p = F.softmax(l, dim=1)
                probs.append(p.cpu().numpy())
        return np.concatenate(probs, axis=0)

    print("Extracting DPCANet-v3 probabilities...", flush=True)
    p_v3_train = get_probs(m_v3, eval_train_loader)
    p_v3_val = get_probs(m_v3, eval_val_loader)
    p_v3_test = get_probs(m_v3, eval_test_loader)

    print("Extracting PolarNet probabilities...", flush=True)
    p_pol_train = get_probs(m_pol, eval_train_loader)
    p_pol_val = get_probs(m_pol, eval_val_loader)
    p_pol_test = get_probs(m_pol, eval_test_loader)

    print("Extracting WaferNeXt probabilities...", flush=True)
    p_wn_train = get_probs(m_wn, eval_train_loader)
    p_wn_val = get_probs(m_wn, eval_val_loader)
    p_wn_test = get_probs(m_wn, eval_test_loader)

    # Weighted Ensemble Teacher: 0.50 v3 + 0.30 Polar + 0.20 WaferNeXt
    p_teacher_train = 0.50 * p_v3_train + 0.30 * p_pol_train + 0.20 * p_wn_train
    p_teacher_val = 0.50 * p_v3_val + 0.30 * p_pol_val + 0.20 * p_wn_val
    p_teacher_test = 0.50 * p_v3_test + 0.30 * p_pol_test + 0.20 * p_wn_test

    y_test = loaders['y_test']
    teacher_errs = np.sum(np.argmax(p_teacher_test, axis=1) != y_test)
    print(f"\nWeighted Teacher Test Errs: {teacher_errs} / 7603 ({100*(1-teacher_errs/7603):.3f}%)", flush=True)

    save_path = os.path.join(OUT_DIR, "teacher_distillation_targets.npz")
    np.savez_compressed(
        save_path,
        p_teacher_train=p_teacher_train,
        p_teacher_val=p_teacher_val,
        p_teacher_test=p_teacher_test
    )
    print(f"Teacher distillation targets saved to {save_path}!", flush=True)

if __name__ == '__main__':
    extract_teacher_probs()
