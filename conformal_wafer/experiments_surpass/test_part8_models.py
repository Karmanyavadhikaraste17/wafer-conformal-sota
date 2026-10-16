import torch
import sys
sys.path.insert(0, ".")
from conformal_wafer.experiments_surpass.autonomous_overnight_surpass_loop_part8 import (
    GaborLineHighwayNet, PQSCANet, GrandMasterNet, DPCANetV3, DPCANetV3_RCLF
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

base_m = DPCANetV3(num_classes=38).to(DEVICE)
cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
cart_model.load_state_dict(c_cart['model_state_dict'])

dummy_x = torch.randn(2, 3, 52, 52, device=DEVICE)

print("1. Testing GaborLineHighwayNet...")
m1 = GaborLineHighwayNet(cart_model).to(DEVICE)
out1, prim1 = m1(dummy_x)
print("   m1 output:", out1.shape, prim1.shape)
assert out1.shape == (2, 38) and prim1.shape == (2, 8)

print("2. Testing PQSCANet...")
m2 = PQSCANet(cart_model).to(DEVICE)
out2, prim2 = m2(dummy_x)
print("   m2 output:", out2.shape, prim2.shape)
assert out2.shape == (2, 38) and prim2.shape == (2, 8)

print("3. Testing GrandMasterNet...")
m3 = GrandMasterNet(cart_model).to(DEVICE)
out3, prim3 = m3(dummy_x)
print("   m3 output:", out3.shape, prim3.shape)
assert out3.shape == (2, 38) and prim3.shape == (2, 8)

print("ALL PART 8 MODELS VERIFIED PERFECTLY!")
