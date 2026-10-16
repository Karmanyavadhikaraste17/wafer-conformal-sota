import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

sys.path.insert(0, ".")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.55)

from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.src.train_optimized_sota import PolarNet
from conformal_wafer.src.dataset import CLASS_DEFINITIONS
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders

VECS = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.float32)
VECS_T = torch.from_numpy(VECS).to(DEVICE)

depths = np.sum(VECS, axis=1, keepdims=True)
depths[depths == 0] = 1.0
VECS_BALANCED = VECS / np.sqrt(depths)
VECS_BALANCED_T = torch.from_numpy(VECS_BALANCED.astype(np.float32)).to(DEVICE)

# -------------------------------------------------------------------------
# Define Base DPCANetV3_RCLF (matching Trial 2 structure)
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
        feat = self.pool(h).flatten(1)
        base_logits = self.fc(self.dropout(feat))
        prim_logits = self.head_prim(feat)
        comp_logits = torch.matmul(prim_logits, VECS_T.T)
        final_logits = base_logits + self.gate_prim * comp_logits
        return final_logits, prim_logits

# -------------------------------------------------------------------------
# Dual-Coordinate Cross-Attention Network (DC-CANet)
# -------------------------------------------------------------------------
class DCCANet(nn.Module):
    def __init__(self, cart_model, polar_model):
        super().__init__()
        # Branch 1: Cartesian
        self.cart_stem = cart_model.stem
        self.cart_stage1 = cart_model.stage1
        self.cart_stage2 = cart_model.stage2
        self.cart_stage3 = cart_model.stage3
        self.cart_pool = cart_model.pool
        self.cart_dropout = cart_model.dropout
        self.cart_fc = cart_model.fc
        self.cart_head_prim = cart_model.head_prim
        self.cart_gate_prim = cart_model.gate_prim

        # Branch 2: Polar
        self.polar_resample = polar_model.resample
        self.polar_stem = polar_model.stem
        self.polar_stage1 = polar_model.stage1
        self.polar_stage2 = polar_model.stage2
        self.polar_stage3 = polar_model.stage3
        self.polar_ca = polar_model.ca
        self.polar_pool = polar_model.pool
        self.polar_dropout = polar_model.dropout
        self.polar_fc = polar_model.fc

        # Cross-Coordinate Fusion Gate
        self.cross_gate = nn.Sequential(
            nn.Linear(288 + 200, 128),
            nn.GELU(),
            nn.Linear(128, 38)
        )
        # Initialize gate to be virtually zero at start: bias = -5.0 -> sigmoid(-5) = 0.0067
        nn.init.zeros_(self.cross_gate[2].weight)
        nn.init.constant_(self.cross_gate[2].bias, -5.0)

        # Fused primitive head
        self.fused_prim_head = nn.Sequential(
            nn.Linear(288 + 200, 64),
            nn.GELU(),
            nn.Linear(64, 8)
        )
        nn.init.zeros_(self.fused_prim_head[2].weight)
        nn.init.zeros_(self.fused_prim_head[2].bias)

    def forward(self, x):
        # 1. Cartesian Stream
        h = self.cart_stage3(self.cart_stage2(self.cart_stage1(self.cart_stem(x))))
        f_cart = self.cart_pool(h).flatten(1) # (B, 288)

        base_cart = self.cart_fc(self.cart_dropout(f_cart))
        prim_cart = self.cart_head_prim(f_cart)
        comp_cart = torch.matmul(prim_cart, VECS_T.T)
        z_cart = base_cart + self.cart_gate_prim * comp_cart

        # 2. Polar Stream
        px = self.polar_resample(x)
        p = self.polar_ca(self.polar_stage3(self.polar_stage2(self.polar_stage1(self.polar_stem(px)))))
        f_polar = self.polar_pool(p).flatten(1) # (B, 200)
        z_polar = self.polar_fc(self.polar_dropout(f_polar))

        # 3. Fusion
        f_fused = torch.cat([f_cart, f_polar], dim=1) # (B, 488)
        gate = torch.sigmoid(self.cross_gate(f_fused)) # (B, 38)

        prim_fused = prim_cart + 0.20 * self.fused_prim_head(f_fused)
        comp_fused = torch.matmul(prim_fused, VECS_BALANCED_T.T)

        final_logits = (1.0 - gate) * z_cart + gate * z_polar + 0.10 * gate * comp_fused

        return final_logits, prim_fused

def verify_init():
    print("Building DCCANet and loading weights...", flush=True)
    cart_base = DPCANetV3(num_classes=38).to(DEVICE)
    cart_model = DPCANetV3_RCLF(cart_base).to(DEVICE)

    t2_path = "conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt"
    ckpt_t2 = torch.load(t2_path, map_location=DEVICE, weights_only=False)
    cart_model.load_state_dict(ckpt_t2['model_state_dict'])

    polar_model = PolarNet(num_classes=38).to(DEVICE)
    pol_path = "conformal_wafer/checkpoints/polarnet-v2_fast80_best.pt"
    ckpt_pol = torch.load(pol_path, map_location=DEVICE, weights_only=False)
    polar_model.load_state_dict(ckpt_pol['model_state_dict'])

    model = DCCANet(cart_model, polar_model).to(DEVICE)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DCCANet Total Trainable Parameters: {num_params:,}", flush=True)

    loaders = get_fast_wafer_loaders(batch_size=128)
    print("Evaluating DCCANet at Epoch 0 (initialization) on Test Set (D4-TTA)...", flush=True)

    t_probs_list = []
    with torch.no_grad():
        for x, y, _ in loaders['test']:
            x = x.to(DEVICE)
            p_cpu = np.zeros((len(x), 38), dtype=np.float32)
            for k in range(4):
                xk = torch.rot90(x, k, [2, 3])
                for flip in [False, True]:
                    xv = torch.flip(xk, [3]) if flip else xk
                    with autocast('cuda', enabled=True):
                        l, _ = model(xv)
                    p_cpu += F.softmax(l, dim=1).cpu().numpy()
            t_probs_list.append(p_cpu / 8.0)

    t_probs = np.concatenate(t_probs_list, axis=0)
    preds = np.argmax(t_probs, axis=1)
    errs = int(np.sum(preds != loaders['y_test']))
    acc = (1.0 - errs / len(loaders['y_test'])) * 100
    print(f"\n================================================================================")
    print(f"DCCANet Step 0 Test Errors: {errs} / 7603 ({acc:.3f}%)")
    print(f"================================================================================")

if __name__ == '__main__':
    verify_init()
