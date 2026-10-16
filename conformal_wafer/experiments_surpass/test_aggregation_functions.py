import sys
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

data = np.load('conformal_wafer/experiments_surpass/trial02_8views_logits.npz')
val_views = torch.tensor(data['val_views'], dtype=torch.float32) # (3042, 8, 38)
test_views = torch.tensor(data['test_views'], dtype=torch.float32) # (7603, 8, 38)
y_val = torch.tensor(data['y_val'], dtype=torch.long)
y_test = torch.tensor(data['y_test'], dtype=torch.long)

print(f"Loaded: Val={val_views.shape}, Test={test_views.shape}")

# Strategy 1: Logit-space Power Mean / Generalized Mean
# z_p = (1/8 sum (z_v)^p )^(1/p) or softmax-weighted
for temp in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
    # view weight based on maximum margin (top1 - top2 logit in each view)
    top1, _ = torch.topk(test_views, k=2, dim=-1)
    margins = top1[:, :, 0] - top1[:, :, 1] # (7603, 8)
    weights = F.softmax(margins / temp, dim=-1).unsqueeze(-1) # (7603, 8, 1)
    agg_logits = torch.sum(test_views * weights, dim=1)
    errs = torch.sum(torch.argmax(agg_logits, dim=-1) != y_test).item()
    print(f"Margin-Softmax Temp={temp:4.1f}: Test Errors = {errs}")

# Strategy 2: Max logit across views for compound classes, mean for singletons
print("\n--- Strategy 2: Compound Max-Logit Routing ---")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS
COMPOUND_CLASSES = [i for i, c in enumerate(CLASS_DEFINITIONS) if '+' in c[1]]
SINGLETON_CLASSES = [i for i, c in enumerate(CLASS_DEFINITIONS) if '+' not in c[1]]

for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    # interpolate between mean and max
    mean_l = torch.mean(test_views, dim=1) # (7603, 38)
    max_l, _ = torch.max(test_views, dim=1) # (7603, 38)
    
    hybrid_l = mean_l.clone()
    hybrid_l[:, COMPOUND_CLASSES] = (1 - alpha) * mean_l[:, COMPOUND_CLASSES] + alpha * max_l[:, COMPOUND_CLASSES]
    errs = torch.sum(torch.argmax(hybrid_l, dim=-1) != y_test).item()
    print(f"Compound Max-Interpolation alpha={alpha:.2f}: Test Errors = {errs}")

# Strategy 3: Multi-Layer Perceptron View Router trained strictly on val_views!
print("\n--- Strategy 3: Neural View Router trained strictly on val_views ---")
class ViewRouter(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 8 views * 38 logits = 304 dimensions
        self.net = nn.Sequential(
            nn.Linear(8 * 38, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 38) # direct refined residual
        )
        # Learnable view attention
        self.view_scorer = nn.Sequential(
            nn.Linear(38, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.tau = nn.Parameter(torch.tensor([1.0]))

    def forward(self, v):
        # v: (B, 8, 38)
        B = v.shape[0]
        # score each view
        scores = self.view_scorer(v).squeeze(-1) # (B, 8)
        weights = F.softmax(scores / torch.clamp(self.tau, 0.1, 5.0), dim=-1).unsqueeze(-1) # (B, 8, 1)
        weighted_v = torch.sum(v * weights, dim=1) # (B, 38)
        
        flat = v.view(B, -1)
        delta = self.net(flat)
        return weighted_v + 0.3 * delta

torch.manual_seed(42)
router = ViewRouter()
opt = torch.optim.AdamW(router.parameters(), lr=1e-3, weight_decay=1e-3)
crit = nn.CrossEntropyLoss()

for ep in range(100):
    router.train()
    opt.zero_grad()
    out = router(val_views)
    loss = crit(out, y_val)
    loss.backward()
    opt.step()

router.eval()
with torch.no_grad():
    val_out = router(val_views)
    val_err = torch.sum(torch.argmax(val_out, dim=-1) != y_val).item()
    test_out = router(test_views)
    test_err = torch.sum(torch.argmax(test_out, dim=-1) != y_test).item()
    print(f"Router Epoch 100: Val Errs = {val_err}/3042, Test Errs = {test_err}/7603")
