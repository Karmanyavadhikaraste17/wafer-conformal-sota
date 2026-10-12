import os
import hashlib
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from .config import cfg

# Exact 38 defect definitions from MixedWM38 specification
CLASS_DEFINITIONS = [
    ('C1', 'Normal', [0, 0, 0, 0, 0, 0, 0, 0], 0),
    ('C2', 'Center', [1, 0, 0, 0, 0, 0, 0, 0], 1),
    ('C3', 'Donut', [0, 1, 0, 0, 0, 0, 0, 0], 1),
    ('C4', 'Edge_Loc', [0, 0, 1, 0, 0, 0, 0, 0], 1),
    ('C5', 'Edge_Ring', [0, 0, 0, 1, 0, 0, 0, 0], 1),
    ('C6', 'Loc', [0, 0, 0, 0, 1, 0, 0, 0], 1),
    ('C7', 'Near_Full', [0, 0, 0, 0, 0, 1, 0, 0], 1),
    ('C8', 'Scratch', [0, 0, 0, 0, 0, 0, 1, 0], 1),
    ('C9', 'Random', [0, 0, 0, 0, 0, 0, 0, 1], 1),
    ('C10', 'C+EL', [1, 0, 1, 0, 0, 0, 0, 0], 2),
    ('C11', 'C+ER', [1, 0, 0, 1, 0, 0, 0, 0], 2),
    ('C12', 'C+L', [1, 0, 0, 0, 1, 0, 0, 0], 2),
    ('C13', 'C+S', [1, 0, 0, 0, 0, 0, 1, 0], 2),
    ('C14', 'D+EL', [0, 1, 1, 0, 0, 0, 0, 0], 2),
    ('C15', 'D+ER', [0, 1, 0, 1, 0, 0, 0, 0], 2),
    ('C16', 'D+L', [0, 1, 0, 0, 1, 0, 0, 0], 2),
    ('C17', 'D+S', [0, 1, 0, 0, 0, 0, 1, 0], 2),
    ('C18', 'EL+L', [0, 0, 1, 0, 1, 0, 0, 0], 2),
    ('C19', 'EL+S', [0, 0, 1, 0, 0, 0, 1, 0], 2),
    ('C20', 'ER+L', [0, 0, 0, 1, 1, 0, 0, 0], 2),
    ('C21', 'ER+S', [0, 0, 0, 1, 0, 0, 1, 0], 2),
    ('C22', 'L+S', [0, 0, 0, 0, 1, 0, 1, 0], 2),
    ('C23', 'C+EL+L', [1, 0, 1, 0, 1, 0, 0, 0], 3),
    ('C24', 'C+EL+S', [1, 0, 1, 0, 0, 0, 1, 0], 3),
    ('C25', 'C+ER+L', [1, 0, 0, 1, 1, 0, 0, 0], 3),
    ('C26', 'C+ER+S', [1, 0, 0, 1, 0, 0, 1, 0], 3),
    ('C27', 'C+L+S', [1, 0, 0, 0, 1, 0, 1, 0], 3),
    ('C28', 'D+EL+L', [0, 1, 1, 0, 1, 0, 0, 0], 3),
    ('C29', 'D+EL+S', [0, 1, 1, 0, 0, 0, 1, 0], 3),
    ('C30', 'D+ER+L', [0, 1, 0, 1, 1, 0, 0, 0], 3),
    ('C31', 'D+ER+S', [0, 1, 0, 1, 0, 0, 1, 0], 3),
    ('C32', 'D+L+S', [0, 1, 0, 0, 1, 0, 1, 0], 3),
    ('C33', 'EL+L+S', [0, 0, 1, 0, 1, 0, 1, 0], 3),
    ('C34', 'ER+L+S', [0, 0, 0, 1, 1, 0, 1, 0], 3),
    ('C35', 'C+L+EL+S', [1, 0, 1, 0, 1, 0, 1, 0], 4),
    ('C36', 'C+L+ER+S', [1, 0, 0, 1, 1, 0, 1, 0], 4),
    ('C37', 'D+L+EL+S', [0, 1, 1, 0, 1, 0, 1, 0], 4),
    ('C38', 'D+L+ER+S', [0, 1, 0, 1, 1, 0, 1, 0], 4),
]

def generate_class_map():
    """Build and save class mapping dataframe."""
    records = []
    for idx, (cid, name, vec, depth) in enumerate(CLASS_DEFINITIONS):
        records.append({
            'class_idx': idx,
            'class_id': cid,
            'class_name': name,
            'vector': str(vec),
            'mix_depth': depth
        })
    df = pd.DataFrame(records)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    df.to_csv(cfg.CLASS_MAP_PATH, index=False)
    return df

class WaferDataset(Dataset):
    """
    Wafer Map Dataset with 3-channel categorical die encoding:
    Channel 0: Blank spot mask (x == 0)
    Channel 1: Normal die mask (x == 1)
    Channel 2: Broken die mask (x >= 2)
    """
    def __init__(self, images, labels, depths, vectors=None, is_train=False):
        self.images = images
        self.labels = labels
        self.depths = depths
        self.vectors = vectors
        self.is_train = is_train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # (52, 52)
        label = int(self.labels[idx])
        depth = int(self.depths[idx])

        # Convert to 3-channel categorical representation
        ch0 = (img == 0).astype(np.float32)
        ch1 = (img == 1).astype(np.float32)
        ch2 = (img >= 2).astype(np.float32)
        tensor_img = np.stack([ch0, ch1, ch2], axis=0)  # (3, 52, 52)

        if self.is_train:
            # Wafer geometry invariant augmentations (flips & 90-degree rotations)
            if random.random() > 0.5:
                tensor_img = np.flip(tensor_img, axis=2).copy()  # Horizontal flip
            if random.random() > 0.5:
                tensor_img = np.flip(tensor_img, axis=1).copy()  # Vertical flip
            k_rot = random.randint(0, 3)
            if k_rot > 0:
                tensor_img = np.rot90(tensor_img, k=k_rot, axes=(1, 2)).copy()

        tensor_img = torch.from_numpy(tensor_img)
        item = {
            'image': tensor_img,
            'label': torch.tensor(label, dtype=torch.long),
            'depth': torch.tensor(depth, dtype=torch.long),
        }
        if self.vectors is not None:
            item['vector'] = torch.from_numpy(self.vectors[idx]).float()
        return item

def load_raw_data(data_path=cfg.DATA_PATH):
    """Loads npz data and decodes labels to class indices (0..37) and mix depths."""
    raw = np.load(data_path, allow_pickle=True)
    x = raw['arr_0']
    y_vecs = raw['arr_1']

    # Vector to class index lookup
    vec_to_class = {}
    vec_to_depth = {}
    for idx, (cid, name, vec, depth) in enumerate(CLASS_DEFINITIONS):
        vec_to_class[tuple(vec)] = idx
        vec_to_depth[tuple(vec)] = depth

    y_indices = np.zeros(len(y_vecs), dtype=np.int64)
    y_depths = np.zeros(len(y_vecs), dtype=np.int64)

    for i in range(len(y_vecs)):
        vec_tuple = tuple(y_vecs[i].tolist())
        y_indices[i] = vec_to_class[vec_tuple]
        y_depths[i] = vec_to_depth[vec_tuple]

    return x, y_indices, y_depths, y_vecs

def get_stratified_splits(seed=cfg.SEED):
    """
    Performs leakage-safe 60/20/20 stratified split:
    Train (60%), Calibration (20%), Test (20%).
    Then carves 10% from Train for Validation (54% train, 6% val).
    Calibration and Test are NEVER touched during training.
    """
    x, y, depths, vecs = load_raw_data()
    n_total = len(x)

    # First split: 60% Train Full vs 40% (Cal + Test)
    train_full_idx, cal_test_idx = train_test_split(
        np.arange(n_total),
        test_size=(cfg.CAL_RATIO + cfg.TEST_RATIO),
        stratify=y,
        random_state=seed
    )

    # Second split: Split 40% equally into 20% Cal and 20% Test
    cal_relative_ratio = cfg.CAL_RATIO / (cfg.CAL_RATIO + cfg.TEST_RATIO)
    cal_idx, test_idx = train_test_split(
        cal_test_idx,
        train_size=cal_relative_ratio,
        stratify=y[cal_test_idx],
        random_state=seed
    )

    # Third split: Carve 10% validation out of Train Full
    train_idx, val_idx = train_test_split(
        train_full_idx,
        test_size=cfg.VAL_FROM_TRAIN_RATIO,
        stratify=y[train_full_idx],
        random_state=seed
    )

    split_indices = {
        'train': train_idx,
        'val': val_idx,
        'cal': cal_idx,
        'test': test_idx,
        'train_full': train_full_idx
    }

    # Verify and compute checksums
    checksums = {}
    for k, idxs in split_indices.items():
        h = hashlib.sha256(idxs.tobytes()).hexdigest()[:12]
        checksums[k] = (len(idxs), h)

    data_dict = {
        'x': x,
        'y': y,
        'depths': depths,
        'vecs': vecs,
        'splits': split_indices,
        'checksums': checksums
    }
    return data_dict

def get_dataloaders(batch_size=cfg.BATCH_SIZE, seed=cfg.SEED):
    """Returns PyTorch DataLoaders for all splits."""
    data = get_stratified_splits(seed=seed)
    x, y, depths, vecs = data['x'], data['y'], data['depths'], data['vecs']
    splits = data['splits']

    train_ds = WaferDataset(
        x[splits['train']], y[splits['train']], depths[splits['train']], vecs[splits['train']], is_train=True
    )
    val_ds = WaferDataset(
        x[splits['val']], y[splits['val']], depths[splits['val']], vecs[splits['val']], is_train=False
    )
    cal_ds = WaferDataset(
        x[splits['cal']], y[splits['cal']], depths[splits['cal']], vecs[splits['cal']], is_train=False
    )
    test_ds = WaferDataset(
        x[splits['test']], y[splits['test']], depths[splits['test']], vecs[splits['test']], is_train=False
    )

    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True),
        'cal': DataLoader(cal_ds, batch_size=batch_size, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True),
        'data_dict': data
    }
    return loaders
