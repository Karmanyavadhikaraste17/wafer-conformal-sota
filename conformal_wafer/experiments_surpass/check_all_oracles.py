import os
import glob
import numpy as np

files = sorted(glob.glob("conformal_wafer/experiments_surpass/trial*_d4_probs.npz"))
print(f"Found {len(files)} trial probability files:")

preds_dict = {}
y_test = None

for f in files:
    name = os.path.basename(f).replace("_d4_probs.npz", "")
    data = np.load(f)
    keys = list(data.keys())
    # find probs
    p = None
    for k in ['probs', 'arr_0', 'test_probs']:
        if k in keys:
            p = data[k]
            break
    if p is None:
        p = data[keys[0]]
    if 'y_test' in keys:
        y_test = data['y_test']
    
    if p.ndim == 3: # (8, 7603, 38)
        p = p.mean(axis=0)
    pred = np.argmax(p, axis=-1)
    preds_dict[name] = pred
    print(f"  {name:15s} -> shape: {p.shape}, pred shape: {pred.shape}")

if y_test is None:
    # load from trial02
    d2 = np.load('conformal_wafer/experiments_surpass/trial02_8views_logits.npz')
    y_test = d2['y_test']

print("\n--- STANDALONE ACCURACY PER MODEL ---")
for name, pred in preds_dict.items():
    errs = np.sum(pred != y_test)
    print(f"{name:15s}: {errs:3d} errors ({100*(1 - errs/len(y_test)):.3f}%)")

# Grand Oracle
all_preds = np.stack(list(preds_dict.values()), axis=0) # (M, 7603)
any_correct = np.any(all_preds == y_test[None, :], axis=0)
grand_oracle_errs = np.sum(~any_correct)
print(f"\nGRAND ORACLE across all {len(preds_dict)} models: {grand_oracle_errs} errors ({100*(1 - grand_oracle_errs/len(y_test)):.3f}%)!")

# Majority voting
from scipy.stats import mode
maj_preds = mode(all_preds, axis=0, keepdims=False)[0]
maj_errs = np.sum(maj_preds != y_test)
print(f"Majority Voting across all {len(preds_dict)} models: {maj_errs} errors ({100*(1 - maj_errs/len(y_test)):.3f}%)")
