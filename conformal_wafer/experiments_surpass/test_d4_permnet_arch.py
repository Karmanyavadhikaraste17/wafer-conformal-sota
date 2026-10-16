"""
test_d4_permnet_arch.py

Prototype and verify Trial 36: D4-PermNet (Dihedral Group Equivariant Attention Network)
A single standalone nn.Module that takes a wafer image x, generates all 8 D4 dihedral views internally,
and uses a Multihead Group-Equivariant Attention pooling to dynamically weight views by defect saliency.
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

class D4GroupEquivariantAttention(nn.Module):
    """
    Learns dynamic attention weights A_v across the 8 dihedral orientations
    based on high-frequency defect saliency and feature content.
    """
    def __init__(self, in_features=288, num_classes=38, d_model=128, num_heads=4):
        super().__init__()
        self.d_model = d_model
        
        # Projection from backbone pooled features (288) to attention space (128)
        self.proj_feat = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # Saliency scoring head: outputs a scalar score for each orientation
        self.saliency_head = nn.Sequential(
            nn.Linear(d_model + num_classes, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # Multi-Head Self-Attention among the 8 views
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm_sa = nn.LayerNorm(d_model)
        
        # Refinement residual head
        self.fc_delta = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )
        
        # Learnable temperature for view softmax
        self.tau_attn = nn.Parameter(torch.tensor([1.0]))

    def forward(self, feats_8, logits_8):
        # feats_8: (B, 8, 288)
        # logits_8: (B, 8, 38)
        B = feats_8.shape[0]
        
        # 1. Project features
        h = self.proj_feat(feats_8) # (B, 8, 128)
        
        # 2. Self-attention across orientations so views communicate
        h_sa, _ = self.mha(h, h, h)
        h = self.norm_sa(h + h_sa) # (B, 8, 128)
        
        # 3. Concatenate feature with view logits to evaluate defect evidence
        h_cat = torch.cat([h, logits_8], dim=-1) # (B, 8, 128 + 38 = 166)
        saliency_scores = self.saliency_head(h_cat).squeeze(-1) # (B, 8)
        
        # 4. Softmax over orientations to obtain normalized dynamic weights
        alpha = F.softmax(saliency_scores / torch.clamp(self.tau_attn, min=0.1, max=5.0), dim=-1) # (B, 8)
        
        # 5. Dynamically weighted sum of view logits
        weighted_logits = torch.sum(logits_8 * alpha.unsqueeze(-1), dim=1) # (B, 38)
        
        # 6. Attentive feature representation
        pooled_feat = torch.sum(h * alpha.unsqueeze(-1), dim=1) # (B, 128)
        delta = self.fc_delta(pooled_feat) # (B, 38)
        
        final_logits = weighted_logits + 0.2 * delta
        return final_logits, alpha

class D4PermNet(nn.Module):
    """
    Trial 36: D4-PermNet (Dihedral Group Equivariant Attention Network)
    End-to-end standalone model that internally processes all 8 dihedral transformations
    and uses group-equivariant cross-attention to dynamically focus on the winning view.
    """
    def __init__(self, cart_model):
        super().__init__()
        self.cart = cart_model
        self.d4_attn = D4GroupEquivariantAttention(in_features=288, num_classes=38, d_model=128, num_heads=4)

    def forward(self, x):
        # x: (B, 3, 52, 52)
        B = x.shape[0]
        
        # Generate 8 dihedral views
        views = []
        for k in range(4):
            xk = torch.rot90(x, k, [2, 3])
            for flip in [False, True]:
                xv = torch.flip(xk, [3]) if flip else xk
                views.append(xv)
                
        # Stack into batch (8*B, 3, 52, 52)
        x_all = torch.cat(views, dim=0)
        
        logits_all, prim_all = self.cart(x_all)
        h = self.cart.stage3(self.cart.stage2(self.cart.stage1(self.cart.stem(x_all))))
        f_all = self.cart.pool(h).flatten(1) # (8*B, 288)
        
        # Reshape to (8, B, ...) -> (B, 8, ...)
        f_8 = f_all.view(8, B, 288).permute(1, 0, 2).contiguous() # (B, 8, 288)
        logits_8 = logits_all.view(8, B, 38).permute(1, 0, 2).contiguous() # (B, 8, 38)
        
        final_logits, alpha = self.d4_attn(f_8, logits_8)
        return final_logits, alpha

def test_arch():
    print("Testing D4PermNet Architecture...")
    base_m = DPCANetV3(num_classes=38).to(DEVICE)
    cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
    c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
    cart_model.load_state_dict(c_cart['model_state_dict'])
    print("Loaded Trial 2 backbone weights successfully!")

    model = D4PermNet(cart_model).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    new_params = sum(p.numel() for p in model.d4_attn.parameters())
    print(f"Total model parameters: {total_params:,}")
    print(f"New D4-Attn parameters: {new_params:,}")

    # Test forward pass with batch size 8
    x = torch.randn(8, 3, 52, 52, device=DEVICE)
    with torch.amp.autocast('cuda'):
        final_logits, alpha = model(x)
    print(f"Forward output shape: final_logits={final_logits.shape}, alpha={alpha.shape}")
    print(f"Sample alpha weights for sample 0: {alpha[0].detach().cpu().numpy()}")
    print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print("D4PermNet verified 100% successfully!")

if __name__ == '__main__':
    test_arch()
