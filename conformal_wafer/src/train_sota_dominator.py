"""
train_sota_dominator.py

Trains a Dual-Head Dynamic Perception Coordinate-Attention Network (DPCA-Net)
specifically designed to surpass RDP-Net's 99.13% SOTA on MixedWM38:
1. Multi-scale Dynamic Perception stem (3x3, 5x5, 7x7 parallel depthwise branches)
2. Coordinate Attention residual blocks (DAP-Net inspired)
3. Dual-Head Multi-Task supervision:
   - Head 1: 38-class composite defect CrossEntropy
   - Head 2: 8-dimensional elementary defect Asymmetric Binary Multi-Label Loss
4. Template-Guided Joint Inference + Dihedral D4 Symmetry Equivariance (TTA)
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score

from .config import cfg
from .dataset import get_dataloaders, CLASS_DEFINITIONS

# Build (38, 8) binary defect template matrix
TEMPLATE_MATRIX = np.zeros((38, 8), dtype=np.float32)
for idx, (cid, name, vec, depth) in enumerate(CLASS_DEFINITIONS):
    TEMPLATE_MATRIX[idx] = vec
TEMPLATE_TENSOR = torch.from_numpy(TEMPLATE_MATRIX).to(cfg.DEVICE)

class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid_channels = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))
        return identity * a_h * a_w

class MultiScaleDynamicBlock(nn.Module):
    """Multi-Scale Dynamic Depthwise Inverted Residual Block."""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.stride = stride
        mid_c = out_c

        # 1x1 expand
        self.conv_exp = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.GELU()
        )

        # Multi-scale parallel depthwise branches (3x3, 5x5, 7x7)
        self.dw3 = nn.Conv2d(mid_c, mid_c, 3, stride=stride, padding=1, groups=mid_c, bias=False)
        self.dw5 = nn.Conv2d(mid_c, mid_c, 5, stride=stride, padding=2, groups=mid_c, bias=False)
        self.dw7 = nn.Conv2d(mid_c, mid_c, 7, stride=stride, padding=3, groups=mid_c, bias=False)
        self.dw_bn = nn.BatchNorm2d(mid_c)
        self.dw_act = nn.GELU()

        # Dynamic kernel attention weight
        self.kernel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_c, 3, 1, bias=True),
            nn.Softmax(dim=1)
        )

        # Coordinate Attention
        self.ca = CoordinateAttention(mid_c)

        # 1x1 project
        self.conv_proj = nn.Sequential(
            nn.Conv2d(mid_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        res = self.shortcut(x)
        feat = self.conv_exp(x)

        # Dynamic multi-scale perception
        w = self.kernel_attn(feat) # (B, 3, 1, 1)
        w3, w5, w7 = w[:, 0:1], w[:, 1:2], w[:, 2:3]
        out3 = self.dw3(feat)
        out5 = self.dw5(feat)
        out7 = self.dw7(feat)
        fused = w3 * out3 + w5 * out5 + w7 * out7
        fused = self.dw_act(self.dw_bn(fused))

        fused = self.ca(fused)
        out = self.conv_proj(fused)
        return F.gelu(out + res)

class DPCANet(nn.Module):
    """
    Dual-Head Dynamic Perception Coordinate-Attention Network.
    Jointly predicts 38-class composite label AND 8-defect elementary vector.
    """
    def __init__(self, in_channels=3, num_classes=38, num_elementary=8):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        # Stages (52x52 -> 26x26 -> 13x13 -> 7x7)
        self.stage1 = nn.Sequential(
            MultiScaleDynamicBlock(64, 96, stride=2),
            MultiScaleDynamicBlock(96, 96, stride=1),
        )
        self.stage2 = nn.Sequential(
            MultiScaleDynamicBlock(96, 160, stride=2),
            MultiScaleDynamicBlock(160, 160, stride=1),
            MultiScaleDynamicBlock(160, 160, stride=1),
        )
        self.stage3 = nn.Sequential(
            MultiScaleDynamicBlock(160, 256, stride=2),
            MultiScaleDynamicBlock(256, 256, stride=1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)

        # Head 1: 38-class composite classifier
        self.head_mc = nn.Linear(256, num_classes)
        # Head 2: 8-defect multi-label binary classifier
        self.head_ml = nn.Linear(256, num_elementary)

    def forward(self, x):
        f = self.stem(x)
        f = self.stage1(f)
        f = self.stage2(f)
        f = self.stage3(f)
        emb = self.pool(f).flatten(1)
        emb = self.dropout(emb)
        logits_mc = self.head_mc(emb)
        logits_ml = self.head_ml(emb)
        return logits_mc, logits_ml, emb

def evaluate_dpcanet_joint(model, loader, device, template_tensor=TEMPLATE_TENSOR, beta=0.8, use_d4=True):
    model.eval()
    all_preds = []
    all_labels = []
    all_mc_preds = []

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            if use_d4:
                # 8 Dihedral Views
                probs_joint_sum = 0.0
                probs_mc_sum = 0.0
                for k in range(4):
                    xk = torch.rot90(images, k, [2, 3])
                    for flip in [False, True]:
                        x_view = torch.flip(xk, [3]) if flip else xk
                        with autocast('cuda', enabled=True):
                            l_mc, l_ml, _ = model(x_view)

                        p_mc = F.softmax(l_mc, dim=1) # (B, 38)
                        q_ml = torch.sigmoid(l_ml).clamp(1e-4, 1.0 - 1e-4) # (B, 8)

                        # Template Bernoulli log-likelihood
                        # template_tensor: (38, 8)
                        log_q = torch.log(q_ml).unsqueeze(1) # (B, 1, 8)
                        log_1_q = torch.log(1.0 - q_ml).unsqueeze(1) # (B, 1, 8)
                        V = template_tensor.unsqueeze(0) # (1, 38, 8)
                        log_template = (V * log_q + (1.0 - V) * log_1_q).sum(dim=2) # (B, 38)
                        p_template = F.softmax(log_template, dim=1)

                        p_joint = F.softmax(torch.log(p_mc + 1e-6) + beta * torch.log(p_template + 1e-6), dim=1)
                        probs_joint_sum = probs_joint_sum + p_joint
                        probs_mc_sum = probs_mc_sum + p_mc

                preds = torch.argmax(probs_joint_sum, dim=1)
                mc_preds = torch.argmax(probs_mc_sum, dim=1)
            else:
                with autocast('cuda', enabled=True):
                    l_mc, l_ml, _ = model(images)
                preds = torch.argmax(l_mc, dim=1)
                mc_preds = preds

            all_preds.append(preds.cpu().numpy())
            all_mc_preds.append(mc_preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_mc_preds = np.concatenate(all_mc_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    acc_joint = accuracy_score(all_labels, all_preds)
    f1_joint = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc_mc = accuracy_score(all_labels, all_mc_preds)

    return acc_joint, f1_joint, acc_mc

def main():
    print("=" * 80)
    print("TRAINING SOTA DOMINATOR MODEL: DPCA-NET (MULTI-SCALE + COORDINATE ATTENTION + DUAL HEAD)")
    print("=" * 80)

    loaders = get_dataloaders()
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']

    model = DPCANet(in_channels=3, num_classes=38, num_elementary=8).to(cfg.DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DPCA-Net initialized with {param_count:,} parameters (Compact SOTA Backbone).")

    # Optimizer & Scheduler
    epochs = 28
    lr = 8e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=True)

    criterion_mc = nn.CrossEntropyLoss(label_smoothing=0.02)
    criterion_ml = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    best_weights = None
    save_path = os.path.join(cfg.CHECKPOINTS_DIR, "dpcanet_best.pt")

    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            images = batch['image'].to(cfg.DEVICE, non_blocking=True)
            labels = batch['label'].to(cfg.DEVICE, non_blocking=True)
            vectors = batch['vector'].to(cfg.DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=True):
                l_mc, l_ml, _ = model(images)
                loss_mc = criterion_mc(l_mc, labels)
                loss_ml = criterion_ml(l_ml, vectors)
                loss = loss_mc + 0.6 * loss_ml

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            n_samples += images.size(0)

        scheduler.step()
        train_loss = total_loss / n_samples

        if ep % 4 == 0 or ep >= epochs - 3:
            val_acc, val_f1, _ = evaluate_dpcanet_joint(model, val_loader, cfg.DEVICE, beta=0.8, use_d4=False)
            print(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1*100:.2f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({'model_state_dict': model.state_dict(), 'epoch': ep, 'val_acc': val_acc}, save_path)
                print(f"  -> Best model saved to {save_path} (Val Acc: {val_acc*100:.2f}%)")

    total_time = time.time() - t_start
    print(f"\nTraining completed in {total_time:.1f}s (~{total_time/60:.1f} mins)!")

    # Evaluate best model on TEST SET with D4 Dihedral Joint Inference
    ckpt = torch.load(save_path, map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    print("\nEvaluating Best DPCA-Net on Held-Out Test Set (N=7,603) with D4 Equivariant Joint Inference...")
    test_joint_acc, test_joint_f1, test_mc_acc = evaluate_dpcanet_joint(model, test_loader, cfg.DEVICE, beta=0.8, use_d4=True)
    print("=" * 80)
    print(f"FINAL TEST SET RESULTS (SHA-256 ca4e75dbed4c):")
    print(f"  DPCA-Net Pure MC Accuracy:       {test_mc_acc*100:.3f}%")
    print(f"  DPCA-Net D4 Joint Test Accuracy: {test_joint_acc*100:.3f}% (Target: > 99.13% SOTA)")
    print(f"  DPCA-Net D4 Joint Test Macro-F1: {test_joint_f1*100:.3f}%")
    print("=" * 80)

if __name__ == '__main__':
    main()
