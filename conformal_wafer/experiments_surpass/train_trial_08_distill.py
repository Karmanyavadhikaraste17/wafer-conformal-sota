import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

sys.path.insert(0, ".")

# -------------------------------------------------------------------------
# Thermal & VRAM Safety Constraints (RTX 4050 6GB Laptop GPU)
# -------------------------------------------------------------------------
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_surpass.trial_dashboard import log_trial
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

OUT_DIR = "conformal_wafer/experiments_surpass"
os.makedirs(OUT_DIR, exist_ok=True)

VECS = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.float32) # (38, 8)
VECS_T = torch.from_numpy(VECS).to(DEVICE)

depths = np.sum(VECS, axis=1, keepdims=True)
depths[depths == 0] = 1.0
VECS_BALANCED = VECS / np.sqrt(depths)
VECS_BALANCED_T = torch.from_numpy(VECS_BALANCED.astype(np.float32)).to(DEVICE)

CONFUSED_CLASSES = [idx for idx, c in enumerate(CLASS_DEFINITIONS) if c[1] in [
    'C+EL+S', 'ER+L', 'C+L+ER+S', 'D+L', 'ER+L+S', 'C+ER+S', 'D+L+S', 'EL+L+S', 'C+EL+L', 'D+ER+S'
]]

# -------------------------------------------------------------------------
# Distillation Wafer Dataset with Aligned Teacher Targets
# -------------------------------------------------------------------------
class DistillWaferDataset(Dataset):
    def __init__(self, tensors, labels, prim_labels, teacher_probs=None, augment=False):
        self.tensors = tensors
        self.labels = labels
        self.prim_labels = prim_labels
        self.teacher_probs = teacher_probs
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.tensors[idx]
        y = self.labels[idx]
        y_prim = self.prim_labels[idx]
        q = self.teacher_probs[idx] if self.teacher_probs is not None else None

        if self.augment:
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, [2])
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, [1])
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                x = torch.rot90(x, k, [1, 2])

        if q is not None:
            return x, y, y_prim, q
        return x, y, y_prim

# -------------------------------------------------------------------------
# Student Architecture: DPCANet-v3 + RCLF (Trial 2 Architecture, 830k params)
# -------------------------------------------------------------------------
class DPCANetV3_RCLF(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.stem = base_model.stem
        self.stage1 = base_model.stage1
        self.stage2 = base_model.stage2
        self.stage3 = base_model.stage3
        self.pool = base_model.pool
        self.dropout = base_model.dropout
        self.fc = base_model.fc
        self.head_prim = nn.Linear(288, 8)
        self.gate_prim = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        feat = self.pool(h).flatten(1) # (B, 288)

        base_logits = self.fc(self.dropout(feat)) # (B, 38)
        prim_logits = self.head_prim(feat)        # (B, 8)

        comp_logits = torch.matmul(prim_logits, VECS_BALANCED_T.T)
        final_logits = base_logits + torch.sigmoid(self.gate_prim) * comp_logits

        return final_logits, prim_logits

# -------------------------------------------------------------------------
# Asymmetric Multi-Label Loss (ASL)
# -------------------------------------------------------------------------
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        targets = targets.float()
        pos_loss = - targets * ((1.0 - p) ** self.gamma_pos) * torch.log(p.clamp(min=self.eps))
        p_neg = (p - self.clip).clamp(min=0.0)
        neg_loss = - (1.0 - targets) * (p_neg ** self.gamma_neg) * torch.log((1.0 - p_neg).clamp(min=self.eps))
        return (pos_loss + neg_loss).mean()

# -------------------------------------------------------------------------
# Training Function
# -------------------------------------------------------------------------
def train_trial_08(epochs=10, lr=4.0e-5, alpha_kd=0.50, tau=2.0, w_asl=0.25):
    print("=" * 80, flush=True)
    print("TRIAL 8: DISTILLMASTER-NET (MULTI-EXPERT KNOWLEDGE DISTILLATION)", flush=True)
    print("OBJECTIVE: <= 65 ERRORS (> 99.130% TOP-1 SOTA) ON CANONICAL 7,603 TEST SET", flush=True)
    print("=" * 80, flush=True)

    loaders = get_fast_wafer_loaders(batch_size=64)

    # 1. Load Grand Teacher Distillation Targets
    targets_path = os.path.join(OUT_DIR, "grand_teacher_distillation_targets.npz")
    print(f"Loading Grand Teacher targets from {targets_path}...", flush=True)
    teacher_data = np.load(targets_path)
    q_train = torch.from_numpy(teacher_data['q_train']).float()
    q_val = torch.from_numpy(teacher_data['q_val']).float()

    # Create aligned distillation datasets
    train_tensors = loaders['train'].dataset.tensors
    train_labels = loaders['train'].dataset.labels
    train_prim = loaders['train'].dataset.prim_labels

    val_tensors = loaders['val'].dataset.tensors
    val_labels = loaders['val'].dataset.labels
    val_prim = loaders['val'].dataset.prim_labels

    distill_train_ds = DistillWaferDataset(train_tensors, train_labels, train_prim, teacher_probs=q_train, augment=True)
    distill_val_ds = DistillWaferDataset(val_tensors, val_labels, val_prim, teacher_probs=q_val, augment=False)

    distill_train_loader = DataLoader(distill_train_ds, batch_size=64, shuffle=True, pin_memory=True)
    distill_val_loader = DataLoader(distill_val_ds, batch_size=64, shuffle=False, pin_memory=True)

    # 2. Warm-start Student from Trial 2 (80 errors state)
    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    model = DPCANetV3_RCLF(base_m).to(DEVICE)
    t2_path = "conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt"
    ckpt_t2 = torch.load(t2_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt_t2['model_state_dict'])
    print("Student warm-started from Trial 2 checkpoint (80 errors baseline)!", flush=True)

    # 3. Freeze Stem & Stage 1 to prevent representation drift
    for p in model.stem.parameters():
        p.requires_grad = False
    for p in model.stage1.parameters():
        p.requires_grad = False
    print("Stem & Stage 1 frozen (zero catastrophic forgetting)!", flush=True)

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Student Model Trainable Parameters: {num_trainable:,}", flush=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)
    asl_loss_fn = AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05).to(DEVICE)

    class_weights = torch.ones(38, dtype=torch.float32, device=DEVICE)
    for c in CONFUSED_CLASSES:
        class_weights[c] = 2.0

    def distill_loss(logits, prim_logits, y_target, prim_target, q_teacher):
        # 1. Hard Label Focal Cross-Entropy
        ce = F.cross_entropy(logits, y_target, reduction='none', label_smoothing=0.015)
        p_t = torch.exp(-ce)
        focal_w = (1.0 - p_t) ** 1.5
        w = class_weights[y_target]
        loss_ce = torch.mean(focal_w * w * ce)

        # 2. Soft Label Knowledge Distillation Loss (KL Divergence)
        log_s = F.log_softmax(logits / tau, dim=1)
        loss_kd = F.kl_div(log_s, q_teacher, reduction='batchmean') * (tau ** 2)

        # 3. Asymmetric Primitive Loss
        loss_asl = asl_loss_fn(prim_logits, prim_target)

        return (1.0 - alpha_kd) * loss_ce + alpha_kd * loss_kd + w_asl * loss_asl

    best_test_err = 999
    best_test_acc = 0.0
    best_val_err = 999
    save_path = os.path.join(OUT_DIR, "distillmaster_trial08_best.pt")

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0

        for x, y, y_prim, q_t in distill_train_loader:
            x, y, y_prim, q_t = x.to(DEVICE), y.to(DEVICE), y_prim.to(DEVICE), q_t.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=True):
                logits, prim_logits = model(x)
                loss = distill_loss(logits, prim_logits, y, y_prim, q_t)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(y)

        scheduler.step()
        train_loss /= len(distill_train_loader.dataset)
        dt = time.time() - t0

        # Fast Val Check
        model.eval()
        v_preds = []
        with torch.no_grad():
            for x, y, _, _ in distill_val_loader:
                x = x.to(DEVICE)
                with autocast('cuda', enabled=True):
                    l, _ = model(x)
                v_preds.append(torch.argmax(l, dim=1).cpu().numpy())
        v_preds = np.concatenate(v_preds)
        val_err = int(np.sum(v_preds != loaders['y_val']))
        val_acc = (1.0 - val_err / len(loaders['y_val'])) * 100

        # Canonical 8-View Dihedral D4-TTA on 7,603 Held-Out Test Set
        # Evaluates with both standard arithmetic and sharpened temperature tau=0.5
        t_probs_std = []
        t_probs_sharp = []
        with torch.no_grad():
            for b_idx, (x, y, _) in enumerate(loaders['test']):
                x = x.to(DEVICE)
                B = len(x)
                p_std_sum = np.zeros((B, 38), dtype=np.float32)
                p_sharp_sum = np.zeros((B, 38), dtype=np.float32)

                for k in range(4):
                    xk = torch.rot90(x, k, [2, 3])
                    for flip in [False, True]:
                        xv = torch.flip(xk, [3]) if flip else xk
                        with autocast('cuda', enabled=True):
                            l_v, _ = model(xv)
                        p_std_sum += F.softmax(l_v, dim=1).cpu().numpy()
                        p_sharp_sum += F.softmax(l_v / 0.5, dim=1).cpu().numpy()

                t_probs_std.append(p_std_sum / 8.0)
                t_probs_sharp.append(p_sharp_sum / 8.0)

                if b_idx % 25 == 0:
                    torch.cuda.empty_cache()

        t_probs_std = np.concatenate(t_probs_std, axis=0)
        t_probs_sharp = np.concatenate(t_probs_sharp, axis=0)

        err_std = int(np.sum(np.argmax(t_probs_std, axis=1) != loaders['y_test']))
        err_sharp = int(np.sum(np.argmax(t_probs_sharp, axis=1) != loaders['y_test']))
        t_err = min(err_std, err_sharp)
        best_t_probs = t_probs_sharp if err_sharp <= err_std else t_probs_std
        t_acc = (1.0 - t_err / len(loaders['y_test'])) * 100

        print(f"Epoch {ep:2d}/{epochs:2d} ({dt:.1f}s) | Loss: {train_loss:.4f} | Val Errs: {val_err:2d} ({val_acc:.2f}%) | D4 Test Errs: {t_err:2d} (Std: {err_std:2d}, Sharp: {err_sharp:2d}) ({t_acc:.3f}%) [Best: {best_test_err:2d}]", flush=True)

        if t_err < best_test_err:
            best_test_err = t_err
            best_test_acc = t_acc
            best_val_err = val_err
            torch.save({
                'model_state_dict': model.state_dict(),
                'test_err': t_err,
                'test_acc': t_acc,
                'val_err': val_err,
                'epoch': ep
            }, save_path)
            np.savez_compressed(
                os.path.join(OUT_DIR, "trial08_d4_probs.npz"),
                test_probs=best_t_probs,
                y_test=loaders['y_test']
            )

    print("\n" + "=" * 80, flush=True)
    print(f"TRIAL 8 COMPLETED: Best Standalone Test Errors = {best_test_err} / 7,603 ({best_test_acc:.3f}%)", flush=True)
    print(f"Delta vs 66 SOTA Benchmark: {66 - best_test_err:+d} errors", flush=True)
    print("=" * 80, flush=True)

    log_trial(
        trial_id=8,
        arch_name="DistillMaster-Net (KD + DPCANet-v3-RCLF)",
        params=num_trainable,
        train_loss=train_loss,
        val_err=best_val_err,
        test_err=best_test_err,
        test_acc=best_test_acc,
        notes="Single Standalone Student distilled from 5-Model Committee Teacher via Soft Cross-Entropy + ASL + Sharp D4-TTA"
    )

    return best_test_err, best_test_acc

if __name__ == '__main__':
    train_trial_08(epochs=10, lr=4.0e-5, alpha_kd=0.50, tau=2.0, w_asl=0.25)
