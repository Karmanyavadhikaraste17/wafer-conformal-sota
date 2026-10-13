import math
import itertools
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from conformal_wafer.src.config import cfg
from conformal_wafer.src.dataset import WaferDataset, load_raw_data, get_stratified_splits, CLASS_DEFINITIONS

# -------------------------------------------------------------------------
# 1. Exact Grammar Definition (Verified on MixedWM38)
# -------------------------------------------------------------------------
# Bits: 0:C, 1:D, 2:EL, 3:ER, 4:L, 5:NF, 6:S, 7:RAND
V_LIST = [z for z in itertools.product([0, 1], repeat=8)
          if not (z[0] and z[1]) and not (z[2] and z[3])
          and not ((z[5] or z[7]) and sum(z) != 1)]
V_TENSOR = torch.tensor(V_LIST, dtype=torch.float32).to(cfg.DEVICE) # (38, 8)
KNOWN_TENSOR = V_TENSOR[V_TENSOR.sum(1) < 4]                        # (34, 8) known configs

# -------------------------------------------------------------------------
# 2. Polar Geometry & Equivariant Layers
# -------------------------------------------------------------------------
class PolarResample(nn.Module):
    """52x52 Cartesian wafer map -> (S, P) Equal-Area Polar representation."""
    def __init__(self, S=32, P=128):
        super().__init__()
        # Equal-area radial coordinate: s = (r/R)^2 in [0, 1]
        r = torch.sqrt((torch.arange(S, dtype=torch.float32) + 0.5) / S)
        phi = 2.0 * math.pi * torch.arange(P, dtype=torch.float32) / P
        grid_x = r[:, None] * torch.cos(phi[None, :])
        grid_y = r[:, None] * torch.sin(phi[None, :])
        grid = torch.stack([grid_x, grid_y], dim=-1) # (S, P, 2)
        self.register_buffer("grid", grid[None])      # (1, S, P, 2)

    def forward(self, x):
        return F.grid_sample(x, self.grid.expand(x.size(0), -1, -1, -1), align_corners=False)

class PolarConv(nn.Module):
    """Circular padding along azimuthal angle P, replicate padding along radial coordinate S."""
    def __init__(self, in_c, out_c, k=3, stride=1):
        super().__init__()
        self.p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=stride, bias=False)
        self.norm = nn.GroupNorm(min(8, out_c), out_c)
        self.act = nn.GELU()

    def forward(self, x):
        # Circular wrap in angle (dim 3), replicate in radius (dim 2)
        p = self.p
        x = torch.cat([x[..., -p:], x, x[..., :p]], dim=-1)
        x = F.pad(x, (0, 0, p, p), mode="replicate")
        return self.act(self.norm(self.conv(x)))

# -------------------------------------------------------------------------
# 3. Compositional Polar Network (CP-Net)
# -------------------------------------------------------------------------
class CompositionalPolarNet(nn.Module):
    def __init__(self, S=32, P=128):
        super().__init__()
        self.resample = PolarResample(S=S, P=P)
        
        # Polar Feature Extractor
        self.stem = PolarConv(3, 32, k=3)
        self.block1 = nn.Sequential(PolarConv(32, 64, k=3), PolarConv(64, 64, k=3, stride=2))
        self.block2 = nn.Sequential(PolarConv(64, 128, k=3), PolarConv(128, 128, k=3, stride=2))
        self.block3 = nn.Sequential(PolarConv(128, 256, k=3), PolarConv(256, 256, k=3))
        
        # Dual-Head:
        # Head 1: 8 Primitive Logits
        self.head_primitives = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 8)
        )
        
        # Head 2: 34-Class Known Configuration Direct Classifier
        self.head_configs = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 34)
        )

    def forward(self, x):
        x_polar = self.resample(x)
        feat = self.stem(x_polar)
        feat = self.block1(feat)
        feat = self.block2(feat)
        feat = self.block3(feat)
        prim_logits = self.head_primitives(feat)
        cfg_logits = self.head_configs(feat)
        return prim_logits, cfg_logits

# -------------------------------------------------------------------------
# 4. Library-Novelty Score Calculation
# -------------------------------------------------------------------------
def log_factorized_posterior(prim_logits, cfgs):
    """Computes log q(z | x) for all configurations in cfgs."""
    # prim_logits: (B, 8), cfgs: (N, 8)
    log_p = F.logsigmoid(prim_logits)     # log sigma(l)
    log_1mp = F.logsigmoid(-prim_logits)  # log (1 - sigma(l))
    # Matrix multiplication to sum across 8 bits
    return log_p @ cfgs.T + log_1mp @ (1.0 - cfgs).T

def compute_library_novelty_score(prim_logits, known_cfgs=KNOWN_TENSOR):
    """S_lib(x) = -log sum_{z in V_known} q(z | x)"""
    log_mass = log_factorized_posterior(prim_logits, known_cfgs)
    return -torch.logsumexp(log_mass, dim=1)

def main():
    print("=" * 80)
    print("TRAINING & BENCHMARKING COMPOSITIONAL POLAR NETWORK (CP-NET) WITH S_lib(x)")
    print("=" * 80)

    # 1. Load data
    x_raw, y_raw, depths_raw, vecs_raw = load_raw_data()
    data_dict = get_stratified_splits(seed=cfg.SEED)
    splits = data_dict['splits']
    train_idx = splits['train']
    val_idx = splits['val']
    cal_idx = splits['cal']
    test_idx = splits['test']

    # Filter to 34 known classes (C1-C34) for training/val/cal
    tr_34_idx = train_idx[y_raw[train_idx] < 34]
    val_34_idx = val_idx[y_raw[val_idx] < 34]
    cal_34_idx = cal_idx[y_raw[cal_idx] < 34]
    test_id_idx = test_idx[y_raw[test_idx] < 34]
    test_ood_idx = test_idx[y_raw[test_idx] >= 34] # Exactly C35-C38 (Depth 4)

    print(f"Dataset Partitions:")
    print(f"  34-Class Train (C1-C34):   {len(tr_34_idx)} samples")
    print(f"  34-Class Val (C1-C34):     {len(val_34_idx)} samples")
    print(f"  34-Class Cal (C1-C34):     {len(cal_34_idx)} samples")
    print(f"  Test In-Distribution:      {len(test_id_idx)} samples")
    print(f"  Test Unseen OOD (C35-C38): {len(test_ood_idx)} samples (Zero exposure)")

    # Create datasets
    train_ds = WaferDataset(x_raw[tr_34_idx], y_raw[tr_34_idx], depths_raw[tr_34_idx], vecs_raw[tr_34_idx], is_train=True)
    val_ds = WaferDataset(x_raw[val_34_idx], y_raw[val_34_idx], depths_raw[val_34_idx], vecs_raw[val_34_idx], is_train=False)
    cal_ds = WaferDataset(x_raw[cal_34_idx], y_raw[cal_34_idx], depths_raw[cal_34_idx], vecs_raw[cal_34_idx], is_train=False)
    test_id_ds = WaferDataset(x_raw[test_id_idx], y_raw[test_id_idx], depths_raw[test_id_idx], vecs_raw[test_id_idx], is_train=False)
    test_ood_ds = WaferDataset(x_raw[test_ood_idx], y_raw[test_ood_idx], depths_raw[test_ood_idx], vecs_raw[test_ood_idx], is_train=False)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    cal_loader = DataLoader(cal_ds, batch_size=256, shuffle=False)
    test_id_loader = DataLoader(test_id_ds, batch_size=256, shuffle=False)
    test_ood_loader = DataLoader(test_ood_ds, batch_size=256, shuffle=False)

    # Instantiate model
    model = CompositionalPolarNet().to(cfg.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    epochs = 25
    print(f"\nTraining CP-Net ({epochs} epochs)...")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            imgs = batch['image'].to(cfg.DEVICE, non_blocking=True)
            labels = batch['label'].to(cfg.DEVICE, non_blocking=True)
            vecs = batch['vector'].to(cfg.DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=True):
                prim_logits, cfg_logits = model(imgs)
                l_bce = bce_loss_fn(prim_logits, vecs)
                l_ce = ce_loss_fn(cfg_logits, labels)
                loss = l_bce + l_ce
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        if ep % 5 == 0 or ep == epochs:
            model.eval()
            val_correct = 0
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch['image'].to(cfg.DEVICE)
                    labels = batch['label'].to(cfg.DEVICE)
                    _, cfg_logits = model(imgs)
                    preds = torch.argmax(cfg_logits, dim=1)
                    val_correct += (preds == labels).sum().item()
            print(f"Epoch {ep:02d}/{epochs:02d} | Val 34-Class Accuracy: {val_correct/len(val_ds)*100:.2f}%")

    print(f"Training completed in {time.time()-t0:.1f}s.")

    # -------------------------------------------------------------------------
    # 5. Extract Library Novelty Scores & Conformal Calibration
    # -------------------------------------------------------------------------
    model.eval()

    def get_scores(loader):
        all_slib = []
        with torch.no_grad():
            for batch in loader:
                imgs = batch['image'].to(cfg.DEVICE)
                prim_logits, _ = model(imgs)
                s = compute_library_novelty_score(prim_logits)
                all_slib.append(s.cpu().numpy())
        return np.concatenate(all_slib, axis=0)

    cal_slib = get_scores(cal_loader)
    id_slib = get_scores(test_id_loader)
    ood_slib = get_scores(test_ood_loader)

    # Calibrate Conformal Outlier Threshold at alpha = 0.05 on CALIBRATION split
    # P(S_lib(X) > q_hat) <= alpha
    alpha = 0.05
    n_cal = len(cal_slib)
    q_idx = int(np.ceil((n_cal + 1) * (1.0 - alpha))) - 1
    q_hat = np.sort(cal_slib)[min(max(q_idx, 0), n_cal - 1)]

    # Realized Test ID False Rejection Rate
    id_false_rejection = float(np.mean(id_slib > q_hat))
    # Unseen Defect OOD Rejection Rate on C35-C38
    ood_rejection = float(np.mean(ood_slib > q_hat))

    # AUROC & AUPR
    y_binary = np.concatenate([np.zeros(len(id_slib)), np.ones(len(ood_slib))])
    all_scores = np.concatenate([id_slib, ood_slib])
    auroc = float(roc_auc_score(y_binary, all_scores))
    p, r, _ = precision_recall_curve(y_binary, all_scores)
    aupr = float(auc(r, p))

    print("\n" + "=" * 80)
    print("BREAKTHROUGH RESULTS: COMPOSITIONAL POLAR S_lib(x) ON TRUE UNSEEN C35-C38:")
    print("=" * 80)
    print(f"  Calibration Quantile q_hat (alpha=0.05):   {q_hat:.4f}")
    print(f"  Realized In-Dist False Rejection Rate:     {id_false_rejection*100:.2f}% ({int(np.sum(id_slib > q_hat))}/{len(id_slib)})")
    print(f"  Unseen Quad-Mixture (C35-C38) Rejection:   {ood_rejection*100:.2f}% ({int(np.sum(ood_slib > q_hat))}/{len(ood_slib)})")
    print(f"  AUROC (ID vs Unseen OOD):                  {auroc:.4f}")
    print(f"  AUPR (ID vs Unseen OOD, base pi=0.1052):   {aupr:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
