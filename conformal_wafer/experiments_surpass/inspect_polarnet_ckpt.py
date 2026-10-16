import torch
import sys
sys.path.insert(0, ".")

ckpt_path = "conformal_wafer/checkpoints/polarnet-v2_fast80_best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print("Keys in polarnet ckpt:", ckpt.keys())
if 'model_state_dict' in ckpt:
    sd = ckpt['model_state_dict']
else:
    sd = ckpt

print(f"Total weight tensors: {len(sd)}")
for k in list(sd.keys())[:10]:
    print(" ", k, sd[k].shape)
