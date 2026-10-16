import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PrimitiveQueryCrossAttention(nn.Module):
    def __init__(self, in_channels=160, embed_dim=128, num_queries=8):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        
        self.query_tokens = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        self.to_k = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.to_v = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.prim_classifier = nn.Linear(embed_dim, 1)
        
    def forward(self, feat_map):
        B, C, H, W = feat_map.shape
        K = self.to_k(feat_map).flatten(2) # (B, D, H*W)
        V = self.to_v(feat_map).flatten(2) # (B, D, H*W)
        
        Q = self.query_tokens.unsqueeze(0).expand(B, -1, -1) # (B, 8, D)
        scores = torch.bmm(Q, K) / math.sqrt(self.embed_dim) # (B, 8, H*W)
        attn = F.softmax(scores, dim=-1) # (B, 8, H*W)
        
        attended = torch.bmm(attn, V.transpose(1, 2)) # (B, 8, D)
        prim_logits = self.prim_classifier(attended).squeeze(-1) # (B, 8)
        return prim_logits, attn

if __name__ == '__main__':
    pqca = PrimitiveQueryCrossAttention()
    dummy_feat = torch.randn(4, 160, 13, 13)
    logits, attn = pqca(dummy_feat)
    print("Primitive logits shape:", logits.shape)
    print("Attention weights shape:", attn.shape)
    assert logits.shape == (4, 8)
    assert attn.shape == (4, 8, 169)
    print("PrimitiveQueryCrossAttention verified successfully!")
