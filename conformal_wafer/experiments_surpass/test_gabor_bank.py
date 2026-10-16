import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GaborFilterBank(nn.Module):
    def __init__(self, num_orientations=8, kernel_size=7, wavelength=2.5, sigma=1.5, gamma=0.4):
        super().__init__()
        self.num_orientations = num_orientations
        self.kernel_size = kernel_size
        
        # Precompute 2D spatial grid
        radius = kernel_size // 2
        y, x = torch.meshgrid(
            torch.linspace(-radius, radius, kernel_size),
            torch.linspace(-radius, radius, kernel_size),
            indexing='ij'
        )
        
        kernels = []
        for i in range(num_orientations):
            theta = i * math.pi / num_orientations
            x_prime = x * math.cos(theta) + y * math.sin(theta)
            y_prime = -x * math.sin(theta) + y * math.cos(theta)
            
            gaussian = torch.exp(-(x_prime**2 + (gamma * y_prime)**2) / (2 * sigma**2))
            sinusoid = torch.cos(2 * math.pi * x_prime / wavelength)
            gabor = gaussian * sinusoid
            
            # Zero-mean DC component removal & L1 normalization
            gabor = gabor - gabor.mean()
            gabor = gabor / (torch.norm(gabor, p=1) + 1e-8)
            kernels.append(gabor)
            
        weight = torch.stack(kernels).unsqueeze(1) # (8, 1, 7, 7)
        self.register_buffer('weight', weight.float())
        
    def forward(self, x):
        # x is (B, 1, H, W) defect density
        return F.conv2d(x, self.weight, padding=self.kernel_size // 2)

if __name__ == '__main__':
    gfb = GaborFilterBank()
    dummy = torch.randn(4, 1, 52, 52)
    out = gfb(dummy)
    print("Gabor output shape:", out.shape)
    assert out.shape == (4, 8, 52, 52)
    print("GaborFilterBank initialized and verified successfully!")
