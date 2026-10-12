import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .config import cfg

class GradCAM:
    """
    Custom Grad-CAM implementation for wafer map visualization.
    Hooks into the final convolutional layer of the model.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward()

        # Channel-wise global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Normalize CAM to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        # Upsample to wafer map dimensions (52, 52)
        cam = F.interpolate(cam, size=(cfg.IMG_SIZE, cfg.IMG_SIZE), mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy(), target_class, torch.softmax(output, dim=1)[0, target_class].item()

    def remove_hooks(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

def generate_gradcam_figure(model, test_dataset, class_df, prediction_sets, target_indices, save_path=None):
    """
    Generates a 2x2 grid Grad-CAM figure showing:
    1 single-defect, 1 double-mix, 1 triple-mix, 1 ambiguous/four-mixed wafer map.
    """
    if save_path is None:
        save_path = os.path.join(cfg.FIGURES_DIR, "gradcam_examples.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Find target layer in DAPNet or ResNet
    if hasattr(model, 'stage4'):
        target_layer = model.stage4[-1].pw_conv
    elif hasattr(model, 'layer4'):
        target_layer = model.layer4[-1].conv2
    else:
        # Fallback to first available Conv2d from end
        conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
        target_layer = conv_layers[-1]

    cam_extractor = GradCAM(model, target_layer)
    device = next(model.parameters()).device

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for idx, sample_idx in enumerate(target_indices):
        item = test_dataset[sample_idx]
        img_tensor = item['image'].unsqueeze(0).to(device)
        true_label = item['label'].item()
        true_depth = item['depth'].item()
        pred_set = prediction_sets[sample_idx]

        cam_mask, pred_class, conf = cam_extractor.generate_cam(img_tensor)

        # Reconstruct display image from 3 channels:
        # 0: Blank, 1: Normal die, 2: Defective die
        img_np = item['image'].numpy()
        display_map = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.float32)
        display_map[img_np[0] == 1] = [0.15, 0.15, 0.15]  # Background / blank: dark grey
        display_map[img_np[1] == 1] = [0.20, 0.70, 0.90]  # Normal die: cyan
        display_map[img_np[2] == 1] = [0.95, 0.20, 0.20]  # Defective die: bright red

        true_name = class_df.loc[class_df['class_idx'] == true_label, 'class_name'].values[0]
        pred_name = class_df.loc[class_df['class_idx'] == pred_class, 'class_name'].values[0]
        set_names = [class_df.loc[class_df['class_idx'] == c, 'class_id'].values[0] for c in pred_set]

        ax = axes[idx]
        ax.imshow(display_map)
        ax.imshow(cam_mask, cmap='jet', alpha=0.45)
        ax.set_title(
            f"True: {true_name} (Depth {true_depth})\n"
            f"Pred: {pred_name} (Conf: {conf*100:.1f}%)\n"
            f"RAPS 90% Set: {set_names} (|S|={len(pred_set)})",
            fontsize=10, pad=8
        )
        ax.axis('off')

    cam_extractor.remove_hooks()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grad-CAM figure saved to: {save_path}")
    return save_path
