import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import timm

# -------------------------------------------------------------------------
# 1. Coordinate Attention Mechanism (Hou et al. CVPR 2021 / DAP-Net 2025)
# -------------------------------------------------------------------------
class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for efficient feature recalibration along
    orthogonal spatial directions (height and width).
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid_channels = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Direction-aware 1D pooling
        x_h = self.pool_h(x)  # (N, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (N, C, W, 1) -> (N, C, W, 1)

        # Concatenate along spatial dimension
        y = torch.cat([x_h, x_w], dim=2)  # (N, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))

        # Split back into height and width tensors
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)  # Back to (N, C, 1, W)

        # Generate directional attention weights
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))

        # Recalibrate features
        out = identity * a_h * a_w
        return out

# -------------------------------------------------------------------------
# 2. Depthwise Separable Convolution Block with Coordinate Attention
# -------------------------------------------------------------------------
class DAPBlock(nn.Module):
    """
    DAP-Net Basic Building Block:
    Depthwise Separable Conv + Coordinate Attention + Residual Connection
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride

        # Depthwise Conv
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.dw_act = nn.ReLU(inplace=True)

        # Pointwise Conv
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)

        # Coordinate Attention
        self.coord_att = CoordinateAttention(out_channels)

        # Residual shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.shortcut(x)

        out = self.dw_act(self.dw_bn(self.dw_conv(x)))
        out = self.pw_bn(self.pw_conv(out))
        out = self.coord_att(out)

        out = self.out_act(out + res)
        return out

# -------------------------------------------------------------------------
# 3. DAP-Net Architecture
# -------------------------------------------------------------------------
class DAPNet(nn.Module):
    """
    DAP-Net (IEEE Access 2025 reimplementation):
    Multi-Scale Stem -> Hierarchical Depthwise Separable Stages with
    Coordinate Attention -> Global Pooling -> Classifier Head.
    """
    def __init__(self, in_channels=3, num_classes=38):
        super().__init__()

        # Multi-scale feature stem for 52x52 wafer maps
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 1: 52x52, 64 channels
        self.stage1 = nn.Sequential(
            DAPBlock(64, 64, stride=1),
            DAPBlock(64, 64, stride=1)
        )

        # Stage 2: 26x26, 128 channels
        self.stage2 = nn.Sequential(
            DAPBlock(64, 128, stride=2),
            DAPBlock(128, 128, stride=1)
        )

        # Stage 3: 13x13, 256 channels
        self.stage3 = nn.Sequential(
            DAPBlock(128, 256, stride=2),
            DAPBlock(256, 256, stride=1)
        )

        # Stage 4: 7x7, 512 channels
        self.stage4 = nn.Sequential(
            DAPBlock(256, 512, stride=2),
            DAPBlock(512, 512, stride=1)
        )

        # Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        out = self.fc(x)
        return out

# -------------------------------------------------------------------------
# 4. ResNet-18 Baseline (Adapted for 52x52)
# -------------------------------------------------------------------------
def create_resnet18(in_channels=3, num_classes=38):
    model = tv_models.resnet18(weights=None)
    # Replace initial 7x7 stride 2 conv with 3x3 stride 1 conv for 52x52 inputs
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Replace initial maxpool with Identity to retain spatial fidelity
    model.maxpool = nn.Identity()
    # Replace final FC
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# -------------------------------------------------------------------------
# 5. ViT-Tiny Baseline (Adapted for 52x52)
# -------------------------------------------------------------------------
def create_vit_tiny(in_channels=3, num_classes=38, img_size=52):
    model = timm.create_model(
        'vit_tiny_patch16_224',
        pretrained=False,
        in_chans=in_channels,
        num_classes=num_classes,
        img_size=img_size,
        patch_size=4
    )
    return model

# -------------------------------------------------------------------------
# Model Factory
# -------------------------------------------------------------------------
def get_model(model_name: str, num_classes=38):
    name = model_name.lower().replace("-", "").replace("_", "")
    if "dap" in name:
        return DAPNet(in_channels=3, num_classes=num_classes)
    elif "resnet" in name:
        return create_resnet18(in_channels=3, num_classes=num_classes)
    elif "vit" in name:
        return create_vit_tiny(in_channels=3, num_classes=num_classes, img_size=52)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
