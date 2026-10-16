import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, ".")

# GPU Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

DATA_PATH = "conformal_wafer/data/Wafer_Map_Datasets.npz"
OUT_DIR = "conformal_wafer/experiments_surpass"
os.makedirs(OUT_DIR, exist_ok=True)

from conformal_wafer.src.dataset import CLASS_DEFINITIONS

class FastWaferDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])

        ch0 = (img == 0).astype(np.float32)
        ch1 = (img == 1).astype(np.float32)
        ch2 = (img >= 2).astype(np.float32)
        tensor_img = np.stack([ch0, ch1, ch2], axis=0)

        return {
            'image': torch.from_numpy(tensor_img),
            'label': torch.tensor(label, dtype=torch.long)
        }

def get_loaders(batch_size=128):
    raw = np.load(DATA_PATH, allow_pickle=True)
    x = raw['arr_0']
    y_vecs = raw['arr_1']

    vec_map = {tuple(vec): idx for idx, (cid, name, vec, depth) in enumerate(CLASS_DEFINITIONS)}
    y = np.zeros(len(y_vecs), dtype=np.int64)
    for i in range(len(y_vecs)):
        y[i] = vec_map[tuple(y_vecs[i].tolist())]

    n_total = len(x)
    train_full_idx, test_idx = train_test_split(np.arange(n_total), test_size=0.20, stratify=y, random_state=42)
    train_idx, val_idx = train_test_split(train_full_idx, test_size=0.10, stratify=y[train_full_idx], random_state=42)

    val_ds = FastWaferDataset(x[val_idx], y[val_idx])
    test_ds = FastWaferDataset(x[test_idx], y[test_idx])

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return val_loader, test_loader, y[val_idx], y[test_idx]

def extract_probs(model, loader):
    model.eval()
    single_probs, d4_probs = [], []

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(DEVICE)

            with autocast('cuda', enabled=True):
                out_single = model(images)
                logits_single = out_single[0] if isinstance(out_single, tuple) else out_single
            p_single = F.softmax(logits_single, dim=1)
            single_probs.append(p_single.cpu().numpy())

            # D4 Dihedral 8-View TTA
            p_d4_sum = 0.0
            for k in range(4):
                xk = torch.rot90(images, k, [2, 3])
                for flip in [False, True]:
                    x_view = torch.flip(xk, [3]) if flip else xk
                    with autocast('cuda', enabled=True):
                        out_v = model(x_view)
                        l_v = out_v[0] if isinstance(out_v, tuple) else out_v
                    p_d4_sum = p_d4_sum + F.softmax(l_v, dim=1)
            p_d4 = p_d4_sum / 8.0
            d4_probs.append(p_d4.cpu().numpy())

    return np.concatenate(single_probs, axis=0), np.concatenate(d4_probs, axis=0)

def main():
    val_loader, test_loader, y_val, y_test = get_loaders(batch_size=128)
    print(f"Loaded dataset: Val={len(y_val)}, Test={len(y_test)}")

    # Models dictionary
    models_dict = {}

    # 1. DPCANet-v2
    from conformal_wafer.src.run_sota_campaign import DPCANet
    m_v2 = DPCANet(in_channels=3, num_classes=38).to(DEVICE)
    c_v2 = torch.load('conformal_wafer/checkpoints/dpcanet-v2_80split_best.pt', map_location=DEVICE, weights_only=False)
    m_v2.load_state_dict(c_v2['model_state_dict'])
    models_dict['DPCANet_v2'] = m_v2

    # 2. DPCANet-v3
    from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
    m_v3 = DPCANetV3(num_classes=38).to(DEVICE)
    c_v3 = torch.load('conformal_wafer/experiments_sota/dpcanet_v3_best.pt', map_location=DEVICE, weights_only=False)
    m_v3.load_state_dict(c_v3['model_state_dict'])
    models_dict['DPCANet_v3'] = m_v3

    # 3. HFEdgeDPCANet
    from conformal_wafer.src.train_hf_edgedpcanet import HFEdgeDPCANet
    m_hf = HFEdgeDPCANet(num_classes=38).to(DEVICE)
    c_hf = torch.load('conformal_wafer/checkpoints/hf_edgedpcanet_sota_best.pt', map_location=DEVICE, weights_only=False)
    m_hf.load_state_dict(c_hf['model_state_dict'])
    models_dict['HFEdgeDPCANet'] = m_hf

    # 4. RDPWaferNet
    from conformal_wafer.src.train_rdp_wafernet import RDPWaferNet
    m_rdp = RDPWaferNet(num_classes=38).to(DEVICE)
    c_rdp = torch.load('conformal_wafer/checkpoints/rdp_wafernet_best.pt', map_location=DEVICE, weights_only=False)
    m_rdp.load_state_dict(c_rdp['model_state_dict'])
    models_dict['RDPWaferNet'] = m_rdp

    # 5. MSResNet
    from conformal_wafer.src.train_optimized_sota import MSResNet
    m_ms = MSResNet(in_channels=3, num_classes=38).to(DEVICE)
    c_ms = torch.load('conformal_wafer/checkpoints/ms-resnet_fast80_best.pt', map_location=DEVICE, weights_only=False)
    m_ms.load_state_dict(c_ms['model_state_dict'])
    models_dict['MSResNet'] = m_ms

    # 6. PolarNet
    from conformal_wafer.src.train_optimized_sota import PolarNet
    m_pol = PolarNet(num_classes=38).to(DEVICE)
    c_pol = torch.load('conformal_wafer/checkpoints/polarnet-v2_fast80_best.pt', map_location=DEVICE, weights_only=False)
    m_pol.load_state_dict(c_pol['model_state_dict'])
    models_dict['PolarNet'] = m_pol

    saved_data = {'y_val': y_val, 'y_test': y_test}

    print("\n" + "=" * 80)
    print("EXTRACTING PREDICTIONS & D4-TTA FOR ALL 6 MODELS")
    print("=" * 80)

    for name, model in models_dict.items():
        t0 = time.time()
        print(f"Processing {name}...")
        v_s, v_d4 = extract_probs(model, val_loader)
        t_s, t_d4 = extract_probs(model, test_loader)
        dt = time.time() - t0

        acc_ts = np.mean(np.argmax(t_s, axis=1) == y_test) * 100
        err_ts = np.sum(np.argmax(t_s, axis=1) != y_test)
        acc_td4 = np.mean(np.argmax(t_d4, axis=1) == y_test) * 100
        err_td4 = np.sum(np.argmax(t_d4, axis=1) != y_test)

        print(f"  {name:15s} ({dt:.1f}s): Single Acc = {acc_ts:.3f}% ({err_ts} errs) | D4-TTA Acc = {acc_td4:.3f}% ({err_td4} errs)")

        saved_data[f"{name}_val_single"] = v_s
        saved_data[f"{name}_val_d4"] = v_d4
        saved_data[f"{name}_test_single"] = t_s
        saved_data[f"{name}_test_d4"] = t_d4

    save_path = os.path.join(OUT_DIR, "multi_model_tta_probs.npz")
    np.savez_compressed(save_path, **saved_data)
    print(f"\nAll model predictions saved to: {save_path}")

    # =========================================================================
    # ENSEMBLE & ORACLE ACCURACY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("ENSEMBLE & ORACLE BENCHMARK AGAINST SOTA (RDP-Net 66 ERRORS / 99.13%)")
    print("=" * 80)

    # 1. Oracle Bound across all 6 models with D4-TTA
    d4_test_list = [saved_data[f"{name}_test_d4"] for name in models_dict.keys()]
    preds_d4 = np.stack([np.argmax(p, axis=1) for p in d4_test_list], axis=1)
    oracle_correct = np.any(preds_d4 == y_test[:, None], axis=1)
    oracle_errs = int(len(y_test) - np.sum(oracle_correct))
    print(f"★ 6-MODEL D4-TTA ORACLE UPPER BOUND: {np.mean(oracle_correct)*100:.3f}% ({oracle_errs} errors / {len(y_test)})")
    if oracle_errs < 66:
        print(f"  --> CONFIRMED: The model pool contains enough diversity to surpass 99.13% (Theoretical limit: {oracle_errs} errors vs 66 SOTA)!")
    else:
        print(f"  --> Note: Oracle achieves {oracle_errs} errors.")

    # 2. Equal-Weighted D4-TTA Ensemble (All 6 Models)
    p_ens_all = np.mean(d4_test_list, axis=0)
    acc_all = np.mean(np.argmax(p_ens_all, axis=1) == y_test) * 100
    err_all = int(np.sum(np.argmax(p_ens_all, axis=1) != y_test))
    print(f"\n1. Full 6-Model D4-TTA Equal Ensemble: {acc_all:.3f}% ({err_all} errors)")

    # 3. Top-4 Core Model D4-TTA Ensemble (DPCANet-v2 + v3 + HFEdge + RDPWaferNet)
    top4_keys = ['DPCANet_v2', 'DPCANet_v3', 'HFEdgeDPCANet', 'RDPWaferNet']
    p_top4 = np.mean([saved_data[f"{k}_test_d4"] for k in top4_keys], axis=0)
    acc_top4 = np.mean(np.argmax(p_top4, axis=1) == y_test) * 100
    err_top4 = int(np.sum(np.argmax(p_top4, axis=1) != y_test))
    print(f"2. Core 4-Model D4-TTA Ensemble (v2+v3+HF+RDP): {acc_top4:.3f}% ({err_top4} errors)")

    # 4. Top-3 DPCANet-v2 + v3 + RDPWaferNet
    p_top3 = np.mean([saved_data['DPCANet_v2_test_d4'], saved_data['DPCANet_v3_test_d4'], saved_data['RDPWaferNet_test_d4']], axis=0)
    acc_top3 = np.mean(np.argmax(p_top3, axis=1) == y_test) * 100
    err_top3 = int(np.sum(np.argmax(p_top3, axis=1) != y_test))
    print(f"3. Core 3-Model D4-TTA Ensemble (v2+v3+RDP): {acc_top3:.3f}% ({err_top3} errors)")

    # 5. Nelder-Mead Optimization of Weights on Validation Set
    from scipy.optimize import minimize
    v_d4_list = [saved_data[f"{name}_val_d4"] for name in models_dict.keys()]

    def nll_loss(weights):
        w = np.exp(weights) / np.sum(np.exp(weights))
        p_comb = sum(w[i] * v_d4_list[i] for i in range(len(w)))
        eps = 1e-7
        return -np.mean(np.log(p_comb[np.arange(len(y_val)), y_val] + eps))

    res = minimize(nll_loss, np.zeros(len(models_dict)), method='Nelder-Mead')
    opt_w = np.exp(res.x) / np.sum(np.exp(res.x))
    print("\n4. Optimal Validation Weights (Nelder-Mead):")
    for name, w_val in zip(models_dict.keys(), opt_w):
        print(f"   - {name:15s}: {w_val:.4f}")

    p_opt_test = sum(opt_w[i] * d4_test_list[i] for i in range(len(opt_w)))
    acc_opt = np.mean(np.argmax(p_opt_test, axis=1) == y_test) * 100
    err_opt = int(np.sum(np.argmax(p_opt_test, axis=1) != y_test))
    print(f"--> Optimal Val-Weighted D4 Ensemble: Test Acc = {acc_opt:.3f}% ({err_opt} errors / {len(y_test)})")

if __name__ == '__main__':
    main()
