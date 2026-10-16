import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, ".")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.55)
    torch.backends.cudnn.benchmark = True

from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.src.train_optimized_sota import PolarNet
from conformal_wafer.src.train_wafernext import WaferNeXt
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.experiments_surpass.train_trial_06_trispec import TriSpecNet
from conformal_wafer.experiments_surpass.train_trial_04_hfcomp import DPCANetV4_HFComp

OUT_DIR = "conformal_wafer/experiments_surpass"
os.makedirs(OUT_DIR, exist_ok=True)

class EvalTensorDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors
    def __len__(self):
        return len(self.tensors)
    def __getitem__(self, idx):
        return self.tensors[idx]

def extract_logits(model, loader):
    model.eval()
    all_logits = []
    with torch.no_grad():
        for x in loader:
            x = x.to(DEVICE)
            with autocast('cuda', enabled=True):
                out = model(x)
                l = out[0] if isinstance(out, tuple) else out
            all_logits.append(l.cpu().numpy())
    return np.concatenate(all_logits, axis=0)

def main():
    print("=" * 80)
    print("GENERATING GRAND CONSENSUS TEACHER DISTILLATION TARGETS")
    print("=" * 80)

    loaders = get_fast_wafer_loaders(batch_size=256)
    train_tensors = loaders['train'].dataset.tensors # (27370, 3, 52, 52)
    val_tensors = loaders['val'].dataset.tensors     # (3042, 3, 52, 52)
    test_tensors = loaders['test'].dataset.tensors   # (7603, 3, 52, 52)

    train_loader = DataLoader(EvalTensorDataset(train_tensors), batch_size=256, shuffle=False)
    val_loader = DataLoader(EvalTensorDataset(val_tensors), batch_size=256, shuffle=False)
    test_loader = DataLoader(EvalTensorDataset(test_tensors), batch_size=256, shuffle=False)

    print("Loading 5 Teacher Models...", flush=True)

    # 1. Model 1: Trial 2 (DPCANet-v3 + RCLF, 80 errors)
    b1 = DPCANetV3(num_classes=38).to(DEVICE)
    m1 = DPCANetV3_RCLF(b1).to(DEVICE)
    c1 = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
    m1.load_state_dict(c1['model_state_dict'])

    # 2. Model 2: Trial 6 (TriSpecNet, 81 errors)
    b2 = DPCANetV3(num_classes=38).to(DEVICE)
    m2 = TriSpecNet(b2).to(DEVICE)
    c2 = torch.load("conformal_wafer/experiments_surpass/trispecnet_trial06_best.pt", map_location=DEVICE, weights_only=False)
    m2.load_state_dict(c2['model_state_dict'])

    # 3. Model 3: Trial 4 (DPCANet-v4-HFComp, 83 errors)
    b3 = DPCANetV3(num_classes=38).to(DEVICE)
    m3 = DPCANetV4_HFComp(b3).to(DEVICE)
    c3 = torch.load("conformal_wafer/experiments_surpass/dpcanet_v4_hfcomp_trial04_best.pt", map_location=DEVICE, weights_only=False)
    m3.load_state_dict(c3['model_state_dict'])

    # 4. Model 4: PolarNet (solves 17 boundary errors)
    m4 = PolarNet(num_classes=38).to(DEVICE)
    c4 = torch.load("conformal_wafer/checkpoints/polarnet-v2_fast80_best.pt", map_location=DEVICE, weights_only=False)
    m4.load_state_dict(c4['model_state_dict'])

    # 5. Model 5: WaferNeXt (solves 14 scratch errors)
    m5 = WaferNeXt(dims=[40, 64, 120, 192]).to(DEVICE)
    c5 = torch.load("conformal_wafer/checkpoints/wafernext_compact_best.pt", map_location=DEVICE, weights_only=False)
    m5.load_state_dict(c5['model_state_dict'])

    print("Extracting logits from all 5 models on Train, Val, Test...", flush=True)

    def extract_all(loader, name):
        t0 = time.time()
        l1 = extract_logits(m1, loader)
        l2 = extract_logits(m2, loader)
        l3 = extract_logits(m3, loader)
        l4 = extract_logits(m4, loader)
        l5 = extract_logits(m5, loader)
        dt = time.time() - t0
        print(f"  {name} logits extracted in {dt:.1f}s!", flush=True)

        # Weighted combination: 0.35 T2 + 0.25 T6 + 0.15 T4 + 0.15 Polar + 0.10 WN
        z_teacher = 0.35 * l1 + 0.25 * l2 + 0.15 * l3 + 0.15 * l4 + 0.10 * l5
        return z_teacher

    z_train = extract_all(train_loader, "Train (27,370 wafers)")
    z_val = extract_all(val_loader, "Val (3,042 wafers)")
    z_test = extract_all(test_loader, "Test (7,603 wafers)")

    y_test = loaders['y_test']
    teacher_test_preds = np.argmax(z_test, axis=1)
    t_err = np.sum(teacher_test_preds != y_test)
    print(f"\nGrand Teacher Committee Single-Pass Test Errors: {t_err} / 7603 ({100*(1-t_err/len(y_test)):.3f}%)", flush=True)

    # Convert to temperature-scaled soft probabilities at tau = 2.0
    tau = 2.0
    def to_soft_probs(z):
        exp_z = np.exp((z / tau) - np.max(z / tau, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    q_train = to_soft_probs(z_train).astype(np.float32)
    q_val = to_soft_probs(z_val).astype(np.float32)
    q_test = to_soft_probs(z_test).astype(np.float32)

    save_path = os.path.join(OUT_DIR, "grand_teacher_distillation_targets.npz")
    np.savez_compressed(
        save_path,
        q_train=q_train,
        q_val=q_val,
        q_test=q_test,
        z_train=z_train.astype(np.float32),
        z_test=z_test.astype(np.float32)
    )
    print(f"\nSaved grand teacher distillation targets to: {save_path}!", flush=True)

if __name__ == '__main__':
    main()
