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
print(f"Using device: {DEVICE}")

DATA_PATH = "conformal_wafer/data/Wafer_Map_Datasets.npz"
OUT_DIR = "conformal_wafer/experiments_surpass"
os.makedirs(OUT_DIR, exist_ok=True)

from conformal_wafer.src.dataset import CLASS_DEFINITIONS
from conformal_wafer.src.run_sota_campaign import DPCANet
from conformal_wafer.src.models import DAPNet, create_resnet18
from conformal_wafer.src.train_wafernext import WaferNeXt

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

    models_dict = {}

    # 1. DPCANet-ASL
    m_asl = DPCANet(in_channels=3, num_classes=38).to(DEVICE)
    c_asl = torch.load('conformal_wafer/checkpoints/dpcanet-v2_asl_finetuned.pt', map_location=DEVICE, weights_only=False)
    m_asl.load_state_dict(c_asl['model_state_dict'])
    models_dict['DPCANet_ASL'] = m_asl

    # 2. DAPNet-LS
    m_dap = DAPNet(in_channels=3, num_classes=38).to(DEVICE)
    c_dap = torch.load('conformal_wafer/checkpoints/dapnet_ls01.pt', map_location=DEVICE, weights_only=False)
    m_dap.load_state_dict(c_dap['model_state_dict'])
    models_dict['DAPNet_LS'] = m_dap

    # 3. WaferNeXt
    m_wn = WaferNeXt(dims=[40, 64, 120, 192]).to(DEVICE)
    c_wn = torch.load('conformal_wafer/checkpoints/wafernext_compact_best.pt', map_location=DEVICE, weights_only=False)
    m_wn.load_state_dict(c_wn['model_state_dict'])
    models_dict['WaferNeXt'] = m_wn

    # 4. ResNet18
    m_res = create_resnet18(in_channels=3, num_classes=38).to(DEVICE)
    c_res = torch.load('conformal_wafer/checkpoints/resnet18_best.pt', map_location=DEVICE, weights_only=False)
    m_res.load_state_dict(c_res['model_state_dict'])
    models_dict['ResNet18'] = m_res

    saved_data = {'y_val': y_val, 'y_test': y_test}

    print("\n" + "=" * 80)
    print("EXTRACTING PREDICTIONS & D4-TTA FOR 4 ADDITIONAL MODELS")
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

    save_path = os.path.join(OUT_DIR, "extra_models_tta_probs.npz")
    np.savez_compressed(save_path, **saved_data)
    print(f"\nExtra model predictions saved to: {save_path}")

if __name__ == '__main__':
    main()
