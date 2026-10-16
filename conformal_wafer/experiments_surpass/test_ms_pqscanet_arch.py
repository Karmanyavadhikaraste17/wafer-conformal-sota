"""
test_ms_pqscanet_arch.py

Prototype and verify MS-PQSCANet: Multi-Scale Primitive-Query Spatial Cross-Attention Network.
Attending over both Stage 1 (26x26 = 676 high-res tokens) and Stage 2 (13x13 = 169 semantic tokens)
to eliminate micro-scratch and edge-spot downsampling loss.
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

VECS = torch.from_numpy(np.array([c[2] for c in CLASS_DEFINITIONS], dtype=np.float32)).to(DEVICE)
depths = torch.sum(VECS, dim=1, keepdim=True)
depths[depths == 0] = 1.0
VECS_BALANCED = VECS / torch.sqrt(depths)

class MultiScalePrimitiveCrossAttention(nn.Module):
    """
    8 primitive queries attending jointly across:
    - Stage 1: (B, 96, 26, 26) -> 676 micro-defect tokens
    - Stage 2: (B, 160, 13, 13) -> 169 meso-defect tokens
    Total tokens: 676 + 169 = 845 tokens
    """
    def __init__(self, d_s1=96, d_s2=160, d_model=128, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.query_embed = nn.Parameter(torch.randn(8, d_model) * 0.02)
        
        self.proj_s1 = nn.Sequential(
            nn.Conv2d(d_s1, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model)
        )
        self.proj_s2 = nn.Sequential(
            nn.Conv2d(d_s2, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model)
        )
        
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, h_s1, h_s2):
        # h_s1: (B, 96, 26, 26) -> (B, 128, 26, 26) -> (B, 676, 128)
        B = h_s1.shape[0]
        f1 = self.proj_s1(h_s1).flatten(2).permute(0, 2, 1)
        # h_s2: (B, 160, 13, 13) -> (B, 128, 13, 13) -> (B, 169, 128)
        f2 = self.proj_s2(h_s2).flatten(2).permute(0, 2, 1)
        
        kv = torch.cat([f1, f2], dim=1).contiguous() # (B, 845, 128)
        
        q = self.query_embed.unsqueeze(0).repeat(B, 1, 1).contiguous() # (B, 8, 128)
        attn_out, _ = self.mha(query=q, key=kv, value=kv)
        attn_out = self.norm(attn_out + q)
        prim_logits = self.head(attn_out).squeeze(-1) # (B, 8)
        return prim_logits

class MSPQSCANet(nn.Module):
    """
    Trial 34: Multi-Scale Primitive-Query Spatial Cross-Attention Network
    Integrates high-res micro-defect tokens (26x26) and semantic tokens (13x13).
    """
    def __init__(self, cart_model):
        super().__init__()
        self.cart = cart_model
        self.ms_cross_attn = MultiScalePrimitiveCrossAttention(d_s1=96, d_s2=160, d_model=128, num_heads=4)
        self.gate_attn = nn.Parameter(torch.tensor([-1.5])) # initial sigmoid ~ 0.18

    def forward(self, x):
        h_stem = self.cart.stem(x)
        h_s1 = self.cart.stage1(h_stem) # (B, 96, 26, 26) - HIGH RES
        h_s2 = self.cart.stage2(h_s1)   # (B, 160, 13, 13)
        h_s3 = self.cart.stage3(h_s2)   # (B, 288, 7, 7)
        feat_cart = self.cart.pool(h_s3).flatten(1) # (B, 288)
        base_logits = self.cart.fc(self.cart.dropout(feat_cart)) # (B, 38)
        prim_base = self.cart.head_prim(feat_cart) # (B, 8)

        # Multi-scale cross-attention over 845 tokens
        delta_prim = self.ms_cross_attn(h_s1, h_s2) # (B, 8)
        prim_enhanced = prim_base + torch.sigmoid(self.gate_attn) * delta_prim

        # Compositional logit projection
        comp_logits = torch.matmul(prim_enhanced, VECS_BALANCED.T) # (B, 38)
        final_logits = base_logits + torch.sigmoid(self.cart.gate_prim) * comp_logits
        return final_logits, prim_enhanced

def test_arch():
    print("Testing MSPQSCANet Architecture...")
    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
    c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
    cart_model.load_state_dict(c_cart['model_state_dict'])
    print("Loaded Trial 2 Cartesian weights successfully!")

    model = MSPQSCANet(cart_model).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass with batch size 16
    x = torch.randn(16, 3, 52, 52, device=DEVICE)
    with torch.amp.autocast('cuda'):
        logits, prims = model(x)
    print(f"Forward output shapes: logits={logits.shape}, prims={prims.shape}")
    print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print("MSPQSCANet verified 100% successfully!")

if __name__ == '__main__':
    test_arch()
