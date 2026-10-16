import numpy as np

p1 = np.load("conformal_wafer/experiments_surpass/multi_model_tta_probs.npz")
print("Keys in multi_model_tta_probs.npz:", p1.files)
for k in p1.files:
    if 'Polar' in k:
        print(k, p1[k].shape)
