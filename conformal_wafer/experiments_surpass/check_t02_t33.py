import numpy as np
import torch
import torch.nn.functional as F

d2 = np.load('conformal_wafer/experiments_surpass/trial02_8views_logits.npz')
d33 = np.load('conformal_wafer/experiments_surpass/trial33_8views_logits.npz')

t2_val = torch.tensor(d2['val_views'], dtype=torch.float32)
t2_test = torch.tensor(d2['test_views'], dtype=torch.float32)

t33_val = torch.tensor(d33['val_logits'], dtype=torch.float32)
t33_test = torch.tensor(d33['test_logits'], dtype=torch.float32)

y_val = torch.tensor(d2['y_val'], dtype=torch.long)
y_test = torch.tensor(d2['y_test'], dtype=torch.long)

if t33_test.shape[0] == 8:
    t33_test = t33_test.permute(1, 0, 2)
if t33_val.shape[0] == 8:
    t33_val = t33_val.permute(1, 0, 2)

p2_test = F.softmax(t2_test / 0.50, dim=-1).mean(dim=1)
p33_test = F.softmax(t33_test / 0.50, dim=-1).mean(dim=1)

err2 = (p2_test.argmax(dim=-1) != y_test).sum().item()
err33 = (p33_test.argmax(dim=-1) != y_test).sum().item()
print(f"Trial 02 Test Errors (tau=0.50): {err2} / 7603")
print(f"Trial 33 Test Errors (tau=0.50): {err33} / 7603")

# Check union oracle between Trial 02 and Trial 33
pred2 = p2_test.argmax(dim=-1)
pred33 = p33_test.argmax(dim=-1)

oracle_t2_t33 = ((pred2 == y_test) | (pred33 == y_test)).sum().item()
print(f"Oracle Union (Trial 02 + Trial 33): {7603 - oracle_t2_t33} errors! ({100 * oracle_t2_t33 / 7603:.3f}%)")

# Check linear blend between Trial 02 and Trial 33
for w2 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    w33 = 1.0 - w2
    p_blend = w2 * p2_test + w33 * p33_test
    blend_err = (p_blend.argmax(dim=-1) != y_test).sum().item()
    print(f"Blend w2={w2:.1f}, w33={w33:.1f} -> Test Errors: {blend_err} / 7603 ({100 * (1 - blend_err/7603):.3f}%)")
