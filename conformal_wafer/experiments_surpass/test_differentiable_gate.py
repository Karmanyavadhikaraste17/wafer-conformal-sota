import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

data_t2 = np.load("conformal_wafer/experiments_surpass/trial02_d4_probs.npz")
data_m = np.load("conformal_wafer/experiments_surpass/multi_model_tta_probs.npz")
data_e = np.load("conformal_wafer/experiments_surpass/extra_models_tta_probs.npz")

y_val = torch.from_numpy(data_t2['y_val']).long()
y_test = torch.from_numpy(data_t2['y_test']).long()

p_t2_v = torch.from_numpy(data_t2['val_probs']).float()
p_t2_t = torch.from_numpy(data_t2['test_probs']).float()

p_pol_v = torch.from_numpy(data_m['PolarNet_val_d4']).float()
p_pol_t = torch.from_numpy(data_m['PolarNet_test_d4']).float()

p_wn_v = torch.from_numpy(data_e['WaferNeXt_val_d4']).float()
p_wn_t = torch.from_numpy(data_e['WaferNeXt_test_d4']).float()

print("=" * 80)
print("TESTING DIFFERENTIABLE LOGIT-SPACE GATING FOR SINGLE STANDALONE MODEL")
print("=" * 80)

# Check base errors
err_t2_v = int(torch.sum(torch.argmax(p_t2_v, dim=1) != y_val))
err_t2_t = int(torch.sum(torch.argmax(p_t2_t, dim=1) != y_test))
print(f"Trial 2 Alone -> Val Errs: {err_t2_v} / {len(y_val)} | Test Errs: {err_t2_t} / {len(y_test)} ({100*(1-err_t2_t/len(y_test)):.3f}%)")

# Let's see what happens if we train a small Gating MLP on Validation Set:
# Input to Gate: [p_t2, p_pol, margin_t2] -> (B, 38 + 38 + 1) = 77
def get_features(p1, p2, p3):
    top2_1, _ = torch.topk(p1, k=2, dim=1)
    m1 = (top2_1[:, 0] - top2_1[:, 1]).unsqueeze(1)
    
    top2_2, _ = torch.topk(p2, k=2, dim=1)
    m2 = (top2_2[:, 0] - top2_2[:, 1]).unsqueeze(1)
    
    ent1 = -torch.sum(p1 * torch.log(torch.clamp(p1, min=1e-7)), dim=1, keepdim=True)
    return torch.cat([p1, p2, p3, m1, m2, ent1], dim=1)

X_feat_v = get_features(p_t2_v, p_pol_v, p_wn_v)
X_feat_t = get_features(p_t2_t, p_pol_t, p_wn_t)

print(f"Feature dimension for router: {X_feat_v.shape[1]}")

# Trainable Single Router Module
class StandaloneGatedFusion(nn.Module):
    def __init__(self, in_dim=117):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)
        )
        # Initialize router to heavily favor Trial 2 at start: [4.0, -2.0, -2.0]
        with torch.no_grad():
            self.router[-1].bias.copy_(torch.tensor([3.0, -1.0, -1.0]))
            self.router[-1].weight.zero_()

    def forward(self, p1, p2, p3, feat):
        w = F.softmax(self.router(feat), dim=1) # (B, 3)
        p_fused = w[:, 0:1] * p1 + w[:, 1:2] * p2 + w[:, 2:3] * p3
        return p_fused, w

# Train router on Validation Split with Cross Entropy
torch.manual_seed(42)
router = StandaloneGatedFusion(in_dim=X_feat_v.shape[1])
optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3, weight_decay=1e-4)

best_v_err = err_t2_v
best_t_err = err_t2_t
best_weights = None

for epoch in range(1, 201):
    router.train()
    optimizer.zero_grad()
    p_fused, w = router(p_t2_v, p_pol_v, p_wn_v, X_feat_v)
    loss = F.nll_loss(torch.log(torch.clamp(p_fused, min=1e-7)), y_val)
    loss.backward()
    optimizer.step()
    
    router.eval()
    with torch.no_grad():
        p_fused_v, w_v = router(p_t2_v, p_pol_v, p_wn_v, X_feat_v)
        v_err = int(torch.sum(torch.argmax(p_fused_v, dim=1) != y_val))
        
        p_fused_t, w_t = router(p_t2_t, p_pol_t, p_wn_t, X_feat_t)
        t_err = int(torch.sum(torch.argmax(p_fused_t, dim=1) != y_test))
        
        if v_err < best_v_err:
            best_v_err = v_err
            best_t_err = t_err
            best_weights = router.state_dict()
            print(f"Epoch {epoch:3d} | Val Loss: {loss.item():.4f} | Val Errs: {v_err:2d} (was {err_t2_v}) | Held-Out Test Errs: {t_err:2d} ({(1 - t_err/len(y_test))*100:.3f}%) [Delta vs 80: {80 - t_err:+d}]")

print("=" * 80)
print(f"Final Validation Selected Router Result on Test: {best_t_err} errors ({(1 - best_t_err/len(y_test))*100:.3f}%)")
print("=" * 80)
