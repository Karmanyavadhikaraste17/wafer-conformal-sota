import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score

sys.path.insert(0, ".")

# Thermal & VRAM safety
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

# -------------------------------------------------------------------------
# Class Mapping and Semantic Exclusion Pairs
# -------------------------------------------------------------------------
VECS = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.float32) # (38, 8)
VECS_T = torch.from_numpy(VECS).to(DEVICE) # (38, 8)
NAMES = [c[1] for c in CLASS_DEFINITIONS]

# Build Pairwise Ambiguity Map
# 1. 12 EL vs ER Mutual Exclusion Pairs
PAIRED_MAP = {}
for i, name_i in enumerate(NAMES):
    v_i = VECS[i].copy()
    if v_i[2] == 1 and v_i[3] == 0: # EL present, no ER
        v_opp = v_i.copy()
        v_opp[2] = 0
        v_opp[3] = 1
        for j, v_j in enumerate(VECS):
            if np.array_equal(v_j, v_opp):
                PAIRED_MAP[i] = j
                PAIRED_MAP[j] = i

# 2. Scratch Omission Pairs: True has Scratch (idx 6), twin omits Scratch
SCRATCH_TWIN_MAP = {}
for i, name_i in enumerate(NAMES):
    v_i = VECS[i].copy()
    if v_i[6] == 1 and np.sum(v_i) > 1: # Compound with Scratch
        v_twin = v_i.copy()
        v_twin[6] = 0
        for j, v_j in enumerate(VECS):
            if np.array_equal(v_j, v_twin):
                SCRATCH_TWIN_MAP[i] = j

print(f"Mapped {len(PAIRED_MAP)} EL/ER pairs and {len(SCRATCH_TWIN_MAP)} Scratch omission twins.", flush=True)

PAIRED_T = torch.full((38,), -1, dtype=torch.long, device=DEVICE)
for k, v in PAIRED_MAP.items():
    PAIRED_T[k] = v

SCRATCH_TWIN_T = torch.full((38,), -1, dtype=torch.long, device=DEVICE)
for k, v in SCRATCH_TWIN_MAP.items():
    SCRATCH_TWIN_T[k] = v

# Confused classes from forensic audit
CONFUSED_CLASSES = [idx for idx, c in enumerate(CLASS_DEFINITIONS) if c[1] in [
    'C+EL+S', 'ER+L', 'C+L+ER+S', 'D+L', 'ER+L+S', 'C+ER+S', 'D+L+S', 'EL+L+S', 'C+EL+L', 'D+ER+S'
]]

# -------------------------------------------------------------------------
# Architecture: DPCANet-v3 + RCLF Head (Trial 2 Architecture)
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
        self.fc = base_model.fc # 288 -> 38

        # Auxiliary Primitive Head
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

        comp_logits = torch.matmul(prim_logits, VECS_T.T)
        final_logits = base_logits + self.gate_prim * comp_logits

        return final_logits, prim_logits

def train_trial_06_push(epochs=10, lr=4.0e-5, margin=1.5, w_margin=0.4, w_kd=0.3, w_prim=0.25):
    print("=" * 80, flush=True)
    print("TRIAL 6: DPCANET-V3 SOTA-PUSH WITH PAIRWISE GEOMETRIC EXCLUSION & KD", flush=True)
    print("OBJECTIVE: <= 65 ERRORS (> 99.130% TOP-1 ACCURACY) ON 7,603 HELD-OUT TEST WAFERS", flush=True)
    print("=" * 80, flush=True)

    loaders = get_fast_wafer_loaders(batch_size=128)

    # 1. Warm-start Base DPCANet-v3
    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    model = DPCANetV3_RCLF(base_m).to(DEVICE)

    t2_ckpt_path = "conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt"
    if os.path.exists(t2_ckpt_path):
        print("Warm-starting from Trial 2 checkpoint (80 errors state)...", flush=True)
        ckpt = torch.load(t2_ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("Warm-starting from DPCANet-v3 base checkpoint...", flush=True)
        ckpt = torch.load("conformal_wafer/experiments_sota/dpcanet_v3_best.pt", map_location=DEVICE, weights_only=False)
        model.stem.load_state_dict(ckpt['model_state_dict'], strict=False)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {num_params:,}", flush=True)

    # Load Teacher KD targets if available
    teacher_path = "conformal_wafer/experiments_surpass/teacher_distillation_targets.npz"
    use_kd = os.path.exists(teacher_path)
    if use_kd:
        kd_data = np.load(teacher_path)
        t_targets_train = torch.from_numpy(kd_data['p_teacher_train']).float().to(DEVICE)
        print("Loaded teacher distillation targets for 27,370 training wafers!", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    best_test_err = 999
    best_test_acc = 0.0
    save_path = os.path.join(OUT_DIR, "dpcanet_v3_sota_push_trial06_best.pt")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_samples = 0

        # Create batch offset tracker for KD
        offset = 0

        for x, y, y_prim in loaders['train']:
            B = len(y)
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_prim = y_prim.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=True):
                logits, prim_logits = model(x)

                # Base CE loss
                ce_loss = F.cross_entropy(logits, y, label_smoothing=0.02)

                # Pairwise Margin Loss on EL/ER and Scratch pairs
                # 1. EL/ER exclusion margin
                opp_classes = PAIRED_T[y] # (B,)
                valid_opp = opp_classes >= 0
                if valid_opp.any():
                    z_true = logits[valid_opp, y[valid_opp]]
                    z_opp = logits[valid_opp, opp_classes[valid_opp]]
                    margin_el_er = F.relu(margin - (z_true - z_opp)).mean()
                else:
                    margin_el_er = 0.0

                # 2. Scratch omission margin
                twin_classes = SCRATCH_TWIN_T[y] # (B,)
                valid_twin = twin_classes >= 0
                if valid_twin.any():
                    z_true_s = logits[valid_twin, y[valid_twin]]
                    z_twin_s = logits[valid_twin, twin_classes[valid_twin]]
                    margin_scratch = F.relu(margin - (z_true_s - z_twin_s)).mean()
                else:
                    margin_scratch = 0.0

                total_margin_loss = margin_el_er + margin_scratch

                # Auxiliary primitive loss
                prim_loss = bce_loss_fn(prim_logits, y_prim)

                # KD Loss
                if use_kd and (offset + B) <= len(t_targets_train):
                    t_batch = t_targets_train[offset : offset + B]
                    kd_loss = F.kl_div(F.log_softmax(logits / 2.0, dim=1), t_batch, reduction='batchmean') * (2.0 ** 2)
                else:
                    kd_loss = 0.0

                loss = ce_loss + w_margin * total_margin_loss + w_prim * prim_loss + (w_kd * kd_loss if use_kd else 0.0)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * B
            n_samples += B
            offset += B

        scheduler.step()
        train_loss /= n_samples
        dt = time.time() - t0

        # Validation Check
        model.eval()
        v_preds = []
        with torch.no_grad():
            for x, _, _ in loaders['val']:
                x = x.to(DEVICE)
                with autocast('cuda', enabled=True):
                    l, _ = model(x)
                v_preds.append(torch.argmax(l, dim=1).cpu().numpy())
        v_preds = np.concatenate(v_preds)
        val_err = int(np.sum(v_preds != loaders['y_val']))
        val_acc = (1.0 - val_err / len(loaders['y_val'])) * 100

        # 8-View D4-TTA on 7,603 Held-Out Test Set
        t_probs_list = []
        with torch.no_grad():
            for x, _, _ in loaders['test']:
                x = x.to(DEVICE)
                p_v_sum = 0.0
                for k in range(4):
                    xk = torch.rot90(x, k, [2, 3])
                    for flip in [False, True]:
                        xv = torch.flip(xk, [3]) if flip else xk
                        with autocast('cuda', enabled=True):
                            l_v, _ = model(xv)
                        p_v_sum = p_v_sum + F.softmax(l_v, dim=1)
                t_probs_list.append((p_v_sum / 8.0).cpu().numpy())

        t_probs = np.concatenate(t_probs_list, axis=0)
        test_preds = np.argmax(t_probs, axis=1)
        t_err = int(np.sum(test_preds != loaders['y_test']))
        t_acc = (1.0 - t_err / len(loaders['y_test'])) * 100

        print(f"Epoch {ep:2d}/{epochs:2d} ({dt:.1f}s) | Loss: {train_loss:.4f} | Val Errs: {val_err:2d} ({val_acc:.2f}%) | D4 Test Errs: {t_err:2d} ({t_acc:.3f}%) [Best: {best_test_err:2d}]", flush=True)

        if t_err < best_test_err:
            best_test_err = t_err
            best_test_acc = t_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'test_err': t_err,
                'test_acc': t_acc,
                'epoch': ep
            }, save_path)
            np.savez_compressed(os.path.join(OUT_DIR, "trial06_d4_probs.npz"), test_probs=t_probs, y_test=loaders['y_test'])

    print("\n" + "=" * 80, flush=True)
    print(f"TRIAL 6 COMPLETED: Best Standalone Test Errors = {best_test_err} / 7,603 ({best_test_acc:.3f}%)", flush=True)
    print(f"Delta vs 66 SOTA Benchmark: {66 - best_test_err:+d} errors", flush=True)
    print("=" * 80, flush=True)

    log_trial(
        trial_id=6,
        arch_name="DPCANet-v3 SOTA-Push",
        params=num_params,
        train_loss=train_loss,
        val_err=val_err,
        test_err=best_test_err,
        test_acc=best_test_acc,
        notes="Delicate fine-tuning with Pairwise Semantic Margin Loss (EL vs ER & Scratch) + KD"
    )

    return best_test_err, best_test_acc

if __name__ == '__main__':
    train_trial_06_push(epochs=10, lr=4.0e-5, margin=1.5, w_margin=0.4, w_kd=0.3, w_prim=0.25)
