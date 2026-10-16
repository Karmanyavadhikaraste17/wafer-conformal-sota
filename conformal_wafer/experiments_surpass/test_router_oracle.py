"""
test_router_oracle.py
Extract 8-view logits for Train, Val, and Test sets, and find an optimal orientation router.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, ".")

from conformal_wafer.src.dataset import CLASS_DEFINITIONS
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
loaders = get_fast_wafer_loaders(batch_size=128)

base_m = DPCANetV3(num_classes=38).to(DEVICE)
cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
cart_model.load_state_dict(c_cart['model_state_dict'])
cart_model.eval()

def extract_8_views(loader):
    all_logits = []
    all_prim = []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(DEVICE)
            B = x.shape[0]
            views = []
            for k in range(4):
                xk = torch.rot90(x, k, [2, 3])
                for flip in [False, True]:
                    xv = torch.flip(xk, [3]) if flip else xk
                    views.append(xv)
            x_all = torch.cat(views, dim=0)
            with torch.amp.autocast('cuda'):
                logits_all = cart_model(x_all)
                if isinstance(logits_all, tuple):
                    logits_all = logits_all[0]
            # (8, B, 38) -> (B, 8, 38)
            l_8 = logits_all.view(8, B, 38).permute(1, 0, 2).cpu()
            all_logits.append(l_8)
    return torch.cat(all_logits, dim=0)

print("Extracting 8-view logits for Val (3042 wafers)...")
val_logits_8 = extract_8_views(loaders['val'])
y_val = loaders['y_val']

print("Extracting 8-view logits for Test (7603 wafers)...")
test_logits_8 = extract_8_views(loaders['test'])
y_test = loaders['y_test']

# Baseline uniform tau=0.50 check
p_val = F.softmax(val_logits_8 / 0.50, dim=-1).mean(dim=1)
val_errs = (p_val.argmax(dim=-1).numpy() != y_val).sum()

p_test = F.softmax(test_logits_8 / 0.50, dim=-1).mean(dim=1)
test_errs = (p_test.argmax(dim=-1).numpy() != y_test).sum()

print(f"Baseline Uniform (tau=0.50) -> Val Errs: {val_errs}/3042 | Test Errs: {test_errs}/7603")

# Val & Test Oracle
val_oracle = sum(1 for i in range(len(y_val)) if y_val[i] in val_logits_8[i].argmax(dim=-1).numpy())
test_oracle = sum(1 for i in range(len(y_test)) if y_test[i] in test_logits_8[i].argmax(dim=-1).numpy())
print(f"Oracle -> Val Errs: {len(y_val) - val_oracle}/3042 | Test Errs: {len(y_test) - test_oracle}/7603")

# Can a router learn view weights w_i = MLP(logits_i)?
# A view has features: top-1 prob, top-1 - top-2 margin, entropy, compound logit sum
class ViewScorer(nn.Module):
    def __init__(self):
        super().__init__()
        # Takes 38 logits for a view -> produces scalar view quality score
        self.net = nn.Sequential(
            nn.Linear(38, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
    def forward(self, l_8):
        # l_8: (B, 8, 38)
        B = l_8.shape[0]
        scores = self.net(l_8).squeeze(-1) # (B, 8)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1) # (B, 8, 1)
        p_8 = F.softmax(l_8 / 0.50, dim=-1) # (B, 8, 38)
        p_out = torch.sum(weights * p_8, dim=1) # (B, 38)
        return torch.log(p_out + 1e-9)

scorer = ViewScorer()
opt = torch.optim.AdamW(scorer.parameters(), lr=1e-3, weight_decay=1e-3)
best_val_err = val_errs
best_test_at_best_val = test_errs

for epoch in range(1, 101):
    scorer.train()
    opt.zero_grad()
    logits = scorer(val_logits_8)
    loss = F.cross_entropy(logits, torch.tensor(y_val, dtype=torch.long))
    loss.backward()
    opt.step()
    
    scorer.eval()
    with torch.no_grad():
        v_preds = scorer(val_logits_8).argmax(dim=-1).numpy()
        v_err = (v_preds != y_val).sum()
        t_preds = scorer(test_logits_8).argmax(dim=-1).numpy()
        t_err = (t_preds != y_test).sum()
        
    if v_err < best_val_err:
        best_val_err = v_err
        best_test_at_best_val = t_err
        print(f"Epoch {epoch:3d} | New Best Val Errs: {v_err}/3042 | Corresponding Test Errs: {t_err}/7603")

print(f"Final Router Result: Best Val Errs = {best_val_err} -> Test Errs = {best_test_at_best_val}")
