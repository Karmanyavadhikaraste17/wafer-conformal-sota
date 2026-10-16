import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

# -------------------------------------------------------------------------
# Polar Stream Modules
# -------------------------------------------------------------------------
class PolarResample(nn.Module):
    def __init__(self, S=32, P=64):
        super().__init__()
        r = torch.sqrt((torch.arange(S, dtype=torch.float32) + 0.5) / S)
        phi = 2.0 * math.pi * torch.arange(P, dtype=torch.float32) / P
        grid_x = r[:, None] * torch.cos(phi[None, :])
        grid_y = r[:, None] * torch.sin(phi[None, :])
        grid = torch.stack([grid_x, grid_y], dim=-1)
        self.register_buffer("grid", grid[None])

    def forward(self, x):
        return F.grid_sample(x, self.grid.expand(x.size(0), -1, -1, -1), align_corners=False)

class PolarConv(nn.Module):
    def __init__(self, in_c, out_c, k=3, stride=1):
        super().__init__()
        self.p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()

    def forward(self, x):
        p = self.p
        x = torch.cat([x[..., -p:], x, x[..., :p]], dim=-1)
        x = F.pad(x, (0, 0, p, p), mode="replicate")
        return self.act(self.bn(self.conv(x)))

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

# -------------------------------------------------------------------------
# Directional Scratch Stream
# -------------------------------------------------------------------------
class DirectionalScratchStream(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        # 4 directional kernels: H (1x7), V (7x1), D1 (diagonal), D2 (anti-diagonal)
        self.h_conv = nn.Conv2d(in_channels, 16, kernel_size=(1, 7), padding=(0, 3), bias=False)
        self.v_conv = nn.Conv2d(in_channels, 16, kernel_size=(7, 1), padding=(3, 0), bias=False)
        self.d_conv = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        h = self.h_conv(x)
        v = self.v_conv(x)
        d = self.d_conv(x)
        f = self.act(self.bn(torch.cat([h, v, d], dim=1)))
        return self.pool(f).flatten(1) # (B, 64)

# -------------------------------------------------------------------------
# HoloWafer-Net Architecture
# -------------------------------------------------------------------------
class HoloWaferNet(nn.Module):
    def __init__(self, base_cart, base_polar, num_classes=38):
        super().__init__()
        # 1. Cartesian Stream (warm-started from DPCANet-v3)
        self.cart_stem = base_cart.stem
        self.cart_s1 = base_cart.stage1
        self.cart_s2 = base_cart.stage2
        self.cart_s3 = base_cart.stage3 # Output: (B, 288, 7, 7)
        self.cart_avg = nn.AdaptiveAvgPool2d(1)
        self.cart_max = nn.AdaptiveMaxPool2d(1) # Dual pooling preserves micro peaks!

        # 2. Polar Stream (warm-started from PolarNet-v2)
        self.polar_resample = base_polar.resample
        self.polar_stem = base_polar.stem
        self.polar_s1 = base_polar.stage1
        self.polar_s2 = base_polar.stage2
        self.polar_s3 = base_polar.stage3 # Output: (B, 200, 8, 16)
        self.polar_ca = base_polar.ca
        self.polar_avg = nn.AdaptiveAvgPool2d(1)
        self.polar_max = nn.AdaptiveMaxPool2d(1)

        # 3. Scratch Stream
        self.scratch_stream = DirectionalScratchStream(3, 64)

        # 4. Cross-Domain Gating
        # Cart dims: 288*2 = 576. Polar dims: 200*2 = 400. Scratch dims: 64.
        self.polar_to_cart_gate = nn.Sequential(
            nn.Linear(400, 128),
            nn.GELU(),
            nn.Linear(128, 576),
            nn.Sigmoid()
        )
        self.cart_to_polar_gate = nn.Sequential(
            nn.Linear(576, 128),
            nn.GELU(),
            nn.Linear(128, 400),
            nn.Sigmoid()
        )

        # 5. Unified Representation Projection
        # 576 + 400 + 64 = 1040 dims
        self.fusion_fc = nn.Sequential(
            nn.Linear(1040, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        # 6. Heads
        self.compound_head = nn.Linear(256, num_classes)
        self.primitive_head = nn.Linear(256, 8)

        # 7. Learnable composition scalar
        self.gate_prim = nn.Parameter(torch.tensor([0.30], dtype=torch.float32))

    def forward(self, x, vec_grammar=None):
        # Cartesian forward
        c0 = self.cart_stem(x)
        c1 = self.cart_s1(c0)
        c2 = self.cart_s2(c1)
        c3 = self.cart_s3(c2)
        f_cart = torch.cat([self.cart_avg(c3).flatten(1), self.cart_max(c3).flatten(1)], dim=1) # (B, 576)

        # Polar forward
        p_in = self.polar_resample(x)
        p0 = self.polar_stem(p_in)
        p1 = self.polar_s1(p0)
        p2 = self.polar_s2(p1)
        p3 = self.polar_s3(p2)
        p_ca = self.polar_ca(p3)
        f_polar = torch.cat([self.polar_avg(p_ca).flatten(1), self.polar_max(p_ca).flatten(1)], dim=1) # (B, 400)

        # Scratch forward
        f_scratch = self.scratch_stream(x) # (B, 64)

        # Bi-directional gating
        g_p2c = self.polar_to_cart_gate(f_polar)
        g_c2p = self.cart_to_polar_gate(f_cart)

        f_cart_gated = f_cart * (1.0 + g_p2c)
        f_polar_gated = f_polar * (1.0 + g_c2p)

        # Unified Fusion
        f_fused = torch.cat([f_cart_gated, f_polar_gated, f_scratch], dim=1) # (B, 1040)
        emb = self.fusion_fc(f_fused) # (B, 256)

        # Dual Head predictions
        comp_logits = self.compound_head(emb) # (B, 38)
        prim_logits = self.primitive_head(emb) # (B, 8)

        if vec_grammar is not None:
            implied_comp = torch.matmul(prim_logits, vec_grammar.T)
            final_logits = comp_logits + torch.sigmoid(self.gate_prim) * implied_comp
        else:
            final_logits = comp_logits

        return final_logits, prim_logits

if __name__ == '__main__':
    from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
    from conformal_wafer.src.train_optimized_sota import PolarNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Testing HoloWaferNet on device:", device)

    base_c = DPCANetV3(num_classes=38).to(device)
    base_p = PolarNet(num_classes=38).to(device)

    # Load weights
    c_c = torch.load("conformal_wafer/experiments_sota/dpcanet_v3_best.pt", map_location=device, weights_only=False)
    c_p = torch.load("conformal_wafer/checkpoints/polarnet-v2_fast80_best.pt", map_location=device, weights_only=False)
    base_c.load_state_dict(c_c['model_state_dict'])
    base_p.load_state_dict(c_p['model_state_dict'])
    print("Pretrained base models loaded.")

    model = HoloWaferNet(base_c, base_p, num_classes=38).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"HoloWaferNet Trainable Parameters: {n_params:,}")

    dummy = torch.randn(16, 3, 52, 52).to(device)
    out_c, out_p = model(dummy)
    print(f"Forward output shapes: Compound {out_c.shape}, Primitive {out_p.shape}")
    loss = out_c.sum() + out_p.sum()
    loss.backward()
    print("Backward pass successful!")
