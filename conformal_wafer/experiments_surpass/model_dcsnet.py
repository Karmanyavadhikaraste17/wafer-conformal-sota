"""
model_dcsnet.py

Trial 41 Architecture: DualCoord-SteerableNet (DCS-Net)
Engineered to resolve both fundamental physical bottlenecks of MixedWM38:
1. Directional Micro-Scratches (1-pixel width): Steerable 8-orientation line filter bank
   at full 52x52 resolution with orientation-max highway.
2. Cartesian Circular Distortion (Edge-Ring vs Edge-Local): Continuous Polar Stream with
   circular padding along theta and angular variance harmonic descriptor.
"""

import sys
sys.path.insert(0, ".")
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SteerableLineFilterBank(nn.Module):
    """
    Bank of 8 oriented line filters at angles:
    0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5 degrees.
    Applies 1D line kernels (length 9) rotated across 8 directions at full 52x52 resolution.
    Computes an Orientation-Invariant Peak Saliency Map: S(x) = max_theta Conv_theta(x).
    """
    def __init__(self, in_channels=3, kernel_size=9, num_angles=8):
        super().__init__()
        self.num_angles = num_angles
        self.kernel_size = kernel_size
        
        # Construct fixed oriented line kernels
        weights = np.zeros((num_angles, 1, kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2
        for i, angle_deg in enumerate(np.linspace(0, 180, num_angles, endpoint=False)):
            theta = np.deg2rad(angle_deg)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            for r in range(-center, center + 1):
                # Sample along the line
                y = int(round(center + r * sin_t))
                x = int(round(center + r * cos_t))
                if 0 <= y < kernel_size and 0 <= x < kernel_size:
                    weights[i, 0, y, x] = 1.0
            # Normalize kernel energy
            s = np.sum(weights[i, 0])
            if s > 0:
                weights[i, 0] /= s
                
        # Register fixed filter bank
        self.register_buffer("line_kernels", torch.tensor(weights, dtype=torch.float32))
        
        # Learnable 1x1 conv to project input channels to 1 defect density channel
        self.defect_proj = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # Post-filter refinement
        self.out_conv = nn.Sequential(
            nn.Conv2d(num_angles, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

    def forward(self, x):
        # x: (B, 3, 52, 52)
        # Defect density map: channel 2 is defect pixels
        defect_density = self.defect_proj(x) # (B, 1, 52, 52)
        
        # Convolve with all 8 oriented line kernels simultaneously
        # line_kernels: (8, 1, 9, 9)
        padding = self.kernel_size // 2
        responses = F.conv2d(defect_density, self.line_kernels, padding=padding) # (B, 8, 52, 52)
        
        # Max response across orientations: detects scratch in ANY angle
        max_response, _ = torch.max(responses, dim=1, keepdim=True) # (B, 1, 52, 52)
        
        # Multi-channel line representation
        feat_line = self.out_conv(responses) # (B, 32, 52, 52)
        return feat_line, max_response

class PolarHarmonicModule(nn.Module):
    """
    Continuous Polar Stream with circular padding and Angular Variance Descriptor:
    Converts wafer to (r, theta) coordinates (32, 64).
    Computes:
    - Angular DC component F_0(r): radial density
    - Angular variance Var_theta(p(r, theta)): discriminates Edge-Ring (Var=0) from Edge-Local (Var>>0).
    """
    def __init__(self, S=32, P=64):
        super().__init__()
        self.S = S
        self.P = P
        
        # Equal-area polar grid
        r = torch.sqrt((torch.arange(S, dtype=torch.float32) + 0.5) / S)
        phi = 2.0 * math.pi * torch.arange(P, dtype=torch.float32) / P
        grid_x = r[:, None] * torch.cos(phi[None, :])
        grid_y = r[:, None] * torch.sin(phi[None, :])
        grid = torch.stack([grid_x, grid_y], dim=-1)
        self.register_buffer("grid", grid[None])
        
        # Circular polar conv: circular padding along theta (dim 3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.GELU()
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_harmonic = nn.Linear(S * 2, 64) # S radial means + S radial variances

    def forward(self, x):
        B = x.shape[0]
        # Resample to (B, 3, 32, 64)
        p = F.grid_sample(x, self.grid.expand(B, -1, -1, -1), align_corners=False) # (B, 3, S, P)
        
        # 1. Physical Angular Harmonic Features on raw defect channel (ch 2)
        p_def = p[:, 2, :, :] # (B, S, P)
        r_mean = torch.mean(p_def, dim=-1) # (B, S) - Radial DC profile
        r_var = torch.var(p_def, dim=-1, unbiased=False) # (B, S) - Angular variance per radius!
        # Edge-Ring has high r_mean near edge (S-1), near-zero r_var
        # Edge-Local has high r_mean AND high r_var!
        harmonic_vec = torch.cat([r_mean, r_var], dim=-1) # (B, 2*S = 64)
        h_desc = F.gelu(self.fc_harmonic(harmonic_vec)) # (B, 64)
        
        # 2. Polar Convolutions with circular padding on theta
        # Pad r with replicate (top/bottom), theta with circular (left/right)
        p_pad = F.pad(p, (1, 1, 0, 0), mode='circular')
        p_pad = F.pad(p_pad, (0, 0, 1, 1), mode='replicate')
        f1 = self.act1(self.bn1(self.conv1(p_pad)))
        
        f1_pad = F.pad(f1, (1, 1, 0, 0), mode='circular')
        f1_pad = F.pad(f1_pad, (0, 0, 1, 1), mode='replicate')
        f2 = self.act2(self.bn2(self.conv2(f1_pad)))
        
        f_polar = self.pool(f2).flatten(1) # (B, 64)
        return torch.cat([f_polar, h_desc], dim=-1), p # (B, 128)

class DualCoordSteerableNet(nn.Module):
    """
    Trial 41: DualCoord-SteerableNet (DCS-Net)
    Unified single-checkpoint model with:
    - Steerable Line Filter Bank Highway for hairline scratches
    - Polar Harmonic Module for Edge-Ring / Edge-Local disambiguation
    - Multi-scale Cartesian Backbone (DPCANet-v3 Stage 1, 2, 3)
    - Residual Compositional Logit Fusion (RCLF) Head
    """
    def __init__(self, base_cart_model, num_classes=38):
        super().__init__()
        # 1. Steerable Line Filter Bank
        self.line_bank = SteerableLineFilterBank(in_channels=3, kernel_size=9, num_angles=8)
        
        # 2. Polar Harmonic Stream
        self.polar_stream = PolarHarmonicModule(S=32, P=64)
        
        # 3. Cartesian Backbone (borrow stem and stages from DPCANet-v3)
        self.stem = base_cart_model.stem
        self.stage1 = base_cart_model.stage1
        self.stage2 = base_cart_model.stage2
        self.stage3 = base_cart_model.stage3
        self.pool = base_cart_model.pool
        
        # Input stem fuse: merges input image (3ch) + line features (32ch) -> 64ch (matches DPCANetV3 stage1 input)
        self.fuse_stem = nn.Sequential(
            nn.Conv2d(3 + 32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        # Final classification head:
        # Cartesian pooled feature: 288
        # Polar + Harmonic feature: 128
        # Total feature: 288 + 128 = 416
        in_feat = 288 + 128
        self.head = nn.Sequential(
            nn.Linear(in_feat, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(192, num_classes)
        )
        
        # Auxiliary primitive defect head (8 primitives: C, D, EL, ER, L, NF, R, S)
        self.prim_head = nn.Sequential(
            nn.Linear(in_feat, 64),
            nn.GELU(),
            nn.Linear(64, 8)
        )

    def forward(self, x):
        # 1. Steerable Line Filter Bank at 52x52
        feat_line, max_scratch_response = self.line_bank(x) # (B, 32, 52, 52)
        
        # 2. Polar Stream with Angular Harmonics
        feat_polar, _ = self.polar_stream(x) # (B, 128)
        
        # 3. Enhanced Cartesian Stream with fused line features
        x_fused = torch.cat([x, feat_line], dim=1) # (B, 35, 52, 52)
        h = self.fuse_stem(x_fused) # (B, 40, 52, 52)
        h = self.stage1(h) # (B, 80, 26, 26)
        h = self.stage2(h) # (B, 160, 13, 13)
        h = self.stage3(h) # (B, 288, 7, 7)
        feat_cart = self.pool(h).flatten(1) # (B, 288)
        
        # 4. Joint Feature Concatenation
        feat_all = torch.cat([feat_cart, feat_polar], dim=-1) # (B, 416)
        
        logits = self.head(feat_all) # (B, 38)
        prim_logits = self.prim_head(feat_all) # (B, 8)
        return logits, prim_logits

if __name__ == '__main__':
    from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
    base_m = DPCANetV3(num_classes=38)
    net = DualCoordSteerableNet(base_m)
    x = torch.randn(4, 3, 52, 52)
    l, p = net(x)
    total_p = sum(param.numel() for param in net.parameters())
    print("DualCoordSteerableNet instantiated successfully!")
    print(f"Total Parameters: {total_p:,}")
    print(f"Output Logits: {l.shape}, Primitives: {p.shape}")
