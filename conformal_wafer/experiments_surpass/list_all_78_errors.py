"""
list_all_78_errors.py
List and categorize all 78 errors of Cartesian DPCANet-v3 (Trial 2).
"""

import sys
import numpy as np

sys.path.insert(0, ".")
from conformal_wafer.src.dataset import CLASS_DEFINITIONS

data = np.load("conformal_wafer/experiments_surpass/amalgamation_probs.npz")
p_cart_test = data['p_cart_test']
y_test = data['y_test']

preds = np.argmax(p_cart_test, axis=-1)
err_indices = np.where(preds != y_test)[0]

print(f"Total Errors: {len(err_indices)} / 7603")
print("=" * 80)
print(f"{'Idx':<5} | {'True Class':<15} | {'Pred Class':<15} | {'True Prob':<10} | {'Pred Prob':<10} | {'Category'}")
print("=" * 80)

error_types = {}
for idx in err_indices:
    t_id = y_test[idx]
    p_id = preds[idx]
    t_name = CLASS_DEFINITIONS[t_id][1]
    p_name = CLASS_DEFINITIONS[p_id][1]
    
    t_prob = p_cart_test[idx, t_id]
    p_prob = p_cart_test[idx, p_id]
    
    # Categorize error
    t_parts = set(t_name.split('+'))
    p_parts = set(p_name.split('+'))
    
    if ('ER' in t_parts and 'EL' in p_parts) or ('EL' in t_parts and 'ER' in p_parts) or \
       (t_name == 'Edge_Ring' and p_name == 'Edge_Loc') or (t_name == 'Edge_Loc' and p_name == 'Edge_Ring'):
        cat = "Boundary (ER vs EL)"
    elif 'S' in t_parts and 'S' not in p_parts:
        cat = "Scratch Omission"
    elif 'S' not in t_parts and 'S' in p_parts:
        cat = "Scratch False Pos"
    elif len(t_parts) > len(p_parts):
        cat = "Primitive Omission"
    elif len(t_parts) < len(p_parts):
        cat = "Primitive False Pos"
    else:
        cat = "Class Confusion"
        
    error_types[cat] = error_types.get(cat, 0) + 1
    print(f"{idx:<5d} | {t_name:<15s} | {p_name:<15s} | {t_prob:<10.3f} | {p_prob:<10.3f} | {cat}")

print("=" * 80)
print("Error Category Breakdown:")
for cat, count in sorted(error_types.items(), key=lambda x: -x[1]):
    print(f"  {cat:<25s}: {count:2d} errors ({100*count/len(err_indices):.1f}%)")
