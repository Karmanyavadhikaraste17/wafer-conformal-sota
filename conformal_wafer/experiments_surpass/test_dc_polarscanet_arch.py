"""
test_dc_polarscanet_arch.py

Prototype and verify Trial 32 architecture: DC-PolarSCANet
Fusing Cartesian DPCANet-v3 + RCLF and PolarNet-v2 into a single unified model.
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

from conformal_wafer.src.dataset import CLASS_DEFINITIONS
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.src.train_optimized_sota import PolarNet, PolarResample, PolarConv, CoordinateAttention

VECS = torch.from_numpy(np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.float32)).to(DEVICE)
depths = torch.sum(VECS, dim=1, keepdim=True)
depths[depths == 0] = 1.0
VECS_BALANCED = VECS / torch.sqrt(depths)

import numpy as np

class DCPolarCrossAttention(nn.Module):
    """8 primitive queries attending jointly across Cartesian (13x13=169) and Polar (8x16=128) tokens"""
    def __init__(self, d_cart=160, d_polar=140, d_model=128, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.query_embed = nn.Parameter(torch.randn(8, d_model) * 0.02)
        
        self.proj_cart = nn.Linear(d_cart, d_model)
        self.proj_polar = nn.Linear(d_polar, d_model)
        
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, feat_cart_map, feat_polar_map):
        # feat_cart_map: (B, 160, 13, 13) -> (B, 169, 128)
        B = feat_cart_map.shape[0]
        f_c = self.proj_cart(feat_cart_map.flatten(2).permute(0, 2, 1))
        # feat_polar_map: (B, 140, 8, 16) -> (B, 128, 128)
        f_p = self.proj_polar(feat_polar_map.flatten(2).permute(0, 2, 1))
        
        kv = torch.cat([f_c, f_p], dim=1).contiguous() # (B, 297, 128)
        
        q = self.query_embed.unsqueeze(0).repeat(B, 1, 1).contiguous() # (B, 8, 128)
        attn_out, _ = self.mha(query=q, key=kv, value=kv)
        attn_out = self.norm(attn_out + q)
        prim_logits = self.head(attn_out).squeeze(-1) # (B, 8)
        return prim_logits

class DCPolarSCANet(nn.Module):
    """
    Trial 32: Dual-Coordinate Polar Spatial Cross-Attention Network
    Unified single-model architecture marrying Cartesian and Polar perception.
    """
    def __init__(self, cart_model, polar_model):
        super().__init__()
        self.cart = cart_model
        self.polar = polar_model
        
        # Cross-attention across Stage 2 Cartesian (160) and Stage 2 Polar (140)
        self.cross_attn = DCPolarCrossAttention(d_cart=160, d_polar=140, d_model=128, num_heads=4)
        
        # Learnable gating parameters
        self.gate_polar = nn.Parameter(torch.tensor([-1.5])) # initial sigmoid ~ 0.18
        self.gate_attn = nn.Parameter(torch.tensor([-1.5]))  # initial sigmoid ~ 0.18

    def forward(self, x):
        # 1. Cartesian forward pass
        h_stem = self.cart.stem(x)
        h_s1 = self.cart.stage1(h_stem)
        h_s2 = self.cart.stage2(h_s1) # (B, 160, 13, 13)
        h_s3 = self.cart.stage3(h_s2)
        f_cart = self.cart.pool(h_s3).flatten(1) # (B, 288)
        logits_cart = self.cart.fc(self.cart.dropout(f_cart)) # (B, 38)
        prim_base = self.cart.head_prim(f_cart) # (B, 8)
        
        # 2. Polar forward pass
        p = self.polar.resample(x)
        fp_stem = self.polar.stem(p)
        fp_s1 = self.polar.stage1(fp_stem)
        fp_s2 = self.polar.stage2(fp_s1) # (B, 140, 8, 16)
        fp_s3 = self.polar.stage3(fp_s2)
        fp_ca = self.polar.ca(fp_s3)
        f_polar = self.polar.pool(fp_ca).flatten(1) # (B, 200)
        logits_polar = self.polar.fc(self.polar.dropout(f_polar)) # (B, 38)
        
        # 3. Dual-Coordinate Spatial Cross-Attention
        delta_prim = self.cross_attn(h_s2, fp_s2) # (B, 8)
        prim_enhanced = prim_base + torch.sigmoid(self.gate_attn) * delta_prim
        
        # 4. Compositional projection
        comp_logits = torch.matmul(prim_enhanced, VECS_BALANCED.T) # (B, 38)
        
        # 5. Unified logit fusion
        final_logits = (
            logits_cart 
            + torch.sigmoid(self.gate_polar) * logits_polar 
            + torch.sigmoid(self.cart.gate_prim) * comp_logits
        )
        return final_logits, prim_enhanced

def test_arch():
    print("Testing DCPolarSCANet Architecture...")
    # Initialize base models
    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
    c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
    cart_model.load_state_dict(c_cart['model_state_dict'])
    print("Loaded Trial 2 Cartesian model successfully!")
    
    polar_model = PolarNet(num_classes=38, S=32, P=64).to(DEVICE)
    c_pol = torch.load("conformal_wafer/checkpoints/polarnet-v2_fast80_best.pt", map_location=DEVICE, weights_only=False)
    polar_model.load_state_dict(c_pol['model_state_dict'])
    print("Loaded PolarNet checkpoint successfully!")
    
    model = DCPolarSCANet(cart_model, polar_model).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    # Test forward pass with batch size 16
    x = torch.randn(16, 3, 52, 52, device=DEVICE)
    with torch.amp.autocast('cuda'):
        logits, prims = model(x)
    print(f"Forward output shapes: logits={logits.shape}, prims={prims.shape}")
    print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print("Architecture verified 100% successfully!")

if __name__ == '__main__':
    test_arch()
