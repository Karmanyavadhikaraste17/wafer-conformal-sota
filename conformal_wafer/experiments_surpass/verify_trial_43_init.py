"""
verify_trial_43_init.py
Verify that Trial 43 initializes at exactly the 80-error champion baseline with 0 regression.
"""

import sys
sys.path.insert(0, ".")
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from conformal_wafer.src.dataset import CLASS_DEFINITIONS
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VECS = np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.float32)
depths = np.sum(VECS, axis=1, keepdims=True)
depths[depths == 0] = 1.0
VECS_BALANCED = VECS / np.sqrt(depths)
VECS_BALANCED_T = torch.from_numpy(VECS_BALANCED.astype(np.float32)).to(DEVICE)

class FixedSteerableLineBank(nn.Module):
    def __init__(self, kernel_size=9, num_angles=8):
        super().__init__()
        weights = np.zeros((num_angles, 1, kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2
        for i, angle_deg in enumerate(np.linspace(0, 180, num_angles, endpoint=False)):
            theta = np.deg2rad(angle_deg)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            for r in range(-center, center + 1):
                y = int(round(center + r * sin_t))
                x = int(round(center + r * cos_t))
                if 0 <= y < kernel_size and 0 <= x < kernel_size:
                    weights[i, 0, y, x] = 1.0
            s = np.sum(weights[i, 0])
            if s > 0: weights[i, 0] /= s
        self.register_buffer("line_kernels", torch.tensor(weights, dtype=torch.float32))
        self.adapter = nn.Sequential(
            nn.Conv2d(num_angles, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.pool = nn.AdaptiveMaxPool2d((13, 13))
        self.proj = nn.Conv2d(32, 64, 1, bias=False)

    def forward(self, x):
        # x: (B, 3, 52, 52), channel 2 is defect channel
        p_def = x[:, 2:3] # (B, 1, 52, 52)
        resp = F.conv2d(p_def, self.line_kernels, padding=4) # (B, 8, 52, 52)
        feat = self.adapter(resp) # (B, 32, 52, 52)
        feat_13 = self.pool(feat) # (B, 32, 13, 13)
        return self.proj(feat_13).flatten(2).permute(0, 2, 1) # (B, 169, 64)

class PolarHarmonicDescriptor(nn.Module):
    def __init__(self, S=32, P=64):
        super().__init__()
        r = torch.sqrt((torch.arange(S, dtype=torch.float32) + 0.5) / S)
        phi = 2.0 * math.pi * torch.arange(P, dtype=torch.float32) / P
        grid_x = r[:, None] * torch.cos(phi[None, :])
        grid_y = r[:, None] * torch.sin(phi[None, :])
        grid = torch.stack([grid_x, grid_y], dim=-1)
        self.register_buffer("grid", grid[None])
        self.fc = nn.Sequential(
            nn.Linear(S * 2, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )

    def forward(self, x):
        B = x.shape[0]
        p = F.grid_sample(x, self.grid.expand(B, -1, -1, -1), align_corners=False)
        p_def = p[:, 2] # (B, S, P)
        r_mean = torch.mean(p_def, dim=-1)
        r_var = torch.var(p_def, dim=-1, unbiased=False)
        h = torch.cat([r_mean, r_var], dim=-1) # (B, 64)
        return self.fc(h).unsqueeze(1) # (B, 1, 64)

class MCCAHybridNet(nn.Module):
    def __init__(self, cart_model, embed_dim=64):
        super().__init__()
        self.cart = cart_model
        self.line_bank = FixedSteerableLineBank()
        self.polar_module = PolarHarmonicDescriptor()
        self.to_kv_cart = nn.Conv2d(160, embed_dim, 1, bias=False)
        self.query_tokens = nn.Parameter(torch.randn(8, embed_dim) * 0.02)
        self.classifier = nn.Linear(embed_dim, 1)
        self.gate_attn = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 1. Cartesian Backbone
        h_stem = self.cart.stem(x)
        h_s1 = self.cart.stage1(h_stem)
        h_s2 = self.cart.stage2(h_s1) # (B, 160, 13, 13)
        h_s3 = self.cart.stage3(h_s2)
        feat_cart = self.cart.pool(h_s3).flatten(1)
        base_logits = self.cart.fc(self.cart.dropout(feat_cart))
        prim_base = self.cart.head_prim(feat_cart)

        # 2. Multi-Scale Key/Values
        kv_cart = self.to_kv_cart(h_s2).flatten(2).permute(0, 2, 1) # (B, 169, 64)
        kv_line = self.line_bank(x) # (B, 169, 64)
        kv_polar = self.polar_module(x) # (B, 1, 64)
        KV = torch.cat([kv_cart, kv_line, kv_polar], dim=1) # (B, 339, 64)

        # 3. Primitive Queries Cross-Attention
        B = x.shape[0]
        Q = self.query_tokens.unsqueeze(0).repeat(B, 1, 1) # (B, 8, 64)
        attn = F.softmax(torch.bmm(Q, KV.transpose(1, 2)) / 8.0, dim=-1) # (B, 8, 339)
        attended = torch.bmm(attn, KV) # (B, 8, 64)
        prim_delta = self.classifier(attended).squeeze(-1) # (B, 8)

        # 4. Residual Compositional Fusion
        prim_total = prim_base + torch.tanh(self.gate_attn) * prim_delta
        comp_logits = torch.matmul(prim_total, VECS_BALANCED_T.to(x.device).T)
        final_logits = base_logits + torch.sigmoid(self.cart.gate_prim) * comp_logits
        return final_logits, prim_total

if __name__ == '__main__':
    loaders = get_fast_wafer_loaders(batch_size=128)
    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
    c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
    cart_model.load_state_dict(c_cart['model_state_dict'])

    model = MCCAHybridNet(cart_model).to(DEVICE)
    model.eval()

    # Verify initial accuracy matches Trial 02 exactly
    all_preds = []
    with torch.no_grad():
        for x, _, _ in loaders['test']:
            x = x.to(DEVICE)
            with torch.amp.autocast('cuda'):
                logits, _ = model(x)
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    preds = np.concatenate(all_preds)
    errs = np.sum(preds != loaders['y_test'])
    print(f"MCCAHybridNet Step 0 Standalone Test Errors: {errs} / 7603 ({100*(1-errs/7603):.3f}%)")
    assert errs == 80, f"Expected 80 errors, got {errs}"
    print("VERIFICATION COMPLETE: Zero-initialization guarantee perfectly validated!")
