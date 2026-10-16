import numpy as np

d = np.load("conformal_wafer/experiments_surpass/grand_teacher_distillation_targets.npz")
print("Files:", d.files)
for k in d.files:
    print(k, d[k].shape, d[k].dtype)
