import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

DATA_PATH = "conformal_wafer/data/Wafer_Map_Datasets.npz"
VECS = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.float32)

class WaferTensorDataset(Dataset):
    def __init__(self, tensors, labels, prim_labels, augment=False):
        self.tensors = tensors  # (N, 3, 52, 52) float32
        self.labels = labels    # (N,) int64
        self.prim_labels = prim_labels # (N, 8) float32
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.tensors[idx]
        y = self.labels[idx]
        y_prim = self.prim_labels[idx]

        if self.augment:
            # Fast in-memory dihedral augmentation
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, [2])
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, [1])
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                x = torch.rot90(x, k, [1, 2])

        return x, y, y_prim

_CACHED_LOADERS = None

def get_fast_wafer_loaders(batch_size=128):
    global _CACHED_LOADERS
    if _CACHED_LOADERS is not None:
        return _CACHED_LOADERS

    print("Loading Wafer Map Dataset into RAM...", flush=True)
    raw = np.load(DATA_PATH, allow_pickle=True)
    images = raw['arr_0']
    y_vecs = raw['arr_1']

    vec_map = {tuple(vec): idx for idx, (cid, name, vec, depth) in enumerate(CLASS_DEFINITIONS)}
    y = np.zeros(len(y_vecs), dtype=np.int64)
    for i in range(len(y_vecs)):
        y[i] = vec_map[tuple(y_vecs[i].tolist())]

    # Pre-convert to 3-channel binary tensors in float32
    print("Pre-converting images to 3-channel tensors...", flush=True)
    ch0 = (images == 0).astype(np.float32)
    ch1 = (images == 1).astype(np.float32)
    ch2 = (images >= 2).astype(np.float32)
    tensors = torch.from_numpy(np.stack([ch0, ch1, ch2], axis=1)) # (N, 3, 52, 52)
    labels = torch.from_numpy(y)
    prim_labels = torch.from_numpy(y_vecs.astype(np.float32))

    n_total = len(tensors)
    train_full_idx, test_idx = train_test_split(np.arange(n_total), test_size=0.20, stratify=y, random_state=42)
    train_idx, val_idx = train_test_split(train_full_idx, test_size=0.10, stratify=y[train_full_idx], random_state=42)

    train_ds = WaferTensorDataset(tensors[train_idx], labels[train_idx], prim_labels[train_idx], augment=True)
    val_ds = WaferTensorDataset(tensors[val_idx], labels[val_idx], prim_labels[val_idx], augment=False)
    test_ds = WaferTensorDataset(tensors[test_idx], labels[test_idx], prim_labels[test_idx], augment=False)

    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True),
        'y_val': y[val_idx],
        'y_test': y[test_idx],
        'vecs': VECS
    }
    _CACHED_LOADERS = loaders
    print(f"DataLoaders ready: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}", flush=True)
    return loaders
