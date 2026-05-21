"""
Grad-CAM Implementation
========================
Gradient-weighted Class Activation Mapping for explainability.

Generates heatmaps showing which regions of an image the model
focuses on when making a classification decision.

Works with:
- CNN models (targets last conv layer)
- ViT models (uses attention rollout as alternative)

Interview talking points:
    - Grad-CAM uses gradients flowing into the last conv layer
    - Channel-wise importance = global average pool of gradients
    - Weighted combination of feature maps → heatmap
    - Validates model is looking at the RIGHT features (not background)
    - Critical for medical AI, safety-critical systems
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks.

    Generates class-discriminative heatmaps by:
    1. Forward pass → get feature maps from target layer
    2. Backward pass → get gradients w.r.t. target class
    3. Global average pool gradients → channel weights
    4. Weighted sum of feature maps → heatmap
    5. ReLU → only positive contributions

    Usage:
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam.generate(image, class_idx=5)
        overlay = gradcam.overlay_heatmap(image, heatmap)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model: The trained model
            target_layer: The conv layer to extract feature maps from.
                          For ResNet: model.backbone.layer4[-1]
                          For custom CNN: model.features[-1]
                          For EfficientNet: model.backbone.features[-1]
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a single image.

        Args:
            input_tensor: Input image (1, C, H, W) — preprocessed
            class_idx: Target class index. If None, uses predicted class.

        Returns:
            Heatmap (H, W) normalized to [0, 1]
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for the target class
        target = output[0, class_idx]
        target.backward()

        # Get gradients and activations
        gradients = self.gradients[0]   # (C, H', W')
        activations = self.activations[0]  # (C, H', W')

        # Channel-wise importance weights (global average pooling of gradients)
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination of feature maps
        heatmap = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        # ReLU — only positive contributions matter
        heatmap = F.relu(heatmap)

        # Normalize to [0, 1]
        heatmap = heatmap.cpu().numpy()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    def generate_batch(
        self,
        images: torch.Tensor,
        class_indices: Optional[List[int]] = None,
    ) -> List[np.ndarray]:
        """
        Generate Grad-CAM heatmaps for a batch of images.

        Args:
            images: Batch of images (B, C, H, W)
            class_indices: Target class for each image. None = use predicted.

        Returns:
            List of heatmaps
        """
        heatmaps = []
        for i in range(images.size(0)):
            img = images[i:i+1]
            cls_idx = class_indices[i] if class_indices else None
            heatmap = self.generate(img, cls_idx)
            heatmaps.append(heatmap)
        return heatmaps


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        image: Original image (H, W, 3) in [0, 1] float
        heatmap: Grad-CAM heatmap (H', W') in [0, 1]
        alpha: Transparency of heatmap overlay
        colormap: Matplotlib colormap name

    Returns:
        Overlaid image (H, W, 3) in [0, 1]
    """
    import cv2

    # Resize heatmap to match image
    h, w = image.shape[:2]
    heatmap_resized = np.array(
        plt.cm.get_cmap(colormap)(
            np.uint8(255 * np.resize(heatmap, (h, w)))
        )[:, :, :3]
    )

    # Simple resize using numpy if heatmap shape doesn't match
    if heatmap.shape[0] != h or heatmap.shape[1] != w:
        # Use PIL for resizing
        from PIL import Image
        heatmap_pil = Image.fromarray(np.uint8(255 * heatmap))
        heatmap_pil = heatmap_pil.resize((w, h), Image.BILINEAR)
        heatmap_resized_gray = np.array(heatmap_pil).astype(np.float32) / 255.0

        # Apply colormap
        cmap = plt.cm.get_cmap(colormap)
        heatmap_resized = cmap(heatmap_resized_gray)[:, :, :3]

    # Overlay
    overlay = alpha * heatmap_resized + (1 - alpha) * image
    overlay = np.clip(overlay, 0, 1)

    return overlay


def denormalize_image(
    tensor: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> np.ndarray:
    """
    Denormalize a tensor image back to [0, 1] range for display.

    Args:
        tensor: Normalized image tensor (C, H, W)
        mean: Normalization mean used
        std: Normalization std used

    Returns:
        Image as numpy array (H, W, C) in [0, 1]
    """
    img = tensor.clone().cpu()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def get_target_layer(model, config: dict):
    """
    Automatically determine the correct Grad-CAM target layer
    based on model type.

    Args:
        model: The model
        config: Experiment config

    Returns:
        The target layer module
    """
    model_type = config["model"]["type"]

    if model_type == "cnn_scratch":
        # Last conv block in our custom CNN
        return model.features[-1]

    elif model_type == "transfer_learning":
        backbone = config["model"]["backbone"]
        if backbone == "resnet50":
            return model.backbone.layer4[-1]
        elif backbone == "efficientnet_b0":
            return model.backbone.features[-1]

    elif model_type == "vit":
        # For ViT, target the last transformer block's LayerNorm
        if hasattr(model.model, "blocks"):
            return model.model.blocks[-1].norm1
        return model.model.norm

    raise ValueError(f"Cannot determine target layer for model type: {model_type}")


def create_gradcam_grid(
    images: List[np.ndarray],
    heatmaps: List[np.ndarray],
    predictions: List[str],
    true_labels: List[str],
    title: str = "Grad-CAM Visualization",
    save_path: Optional[str] = None,
    cols: int = 4,
) -> str:
    """
    Create a grid of Grad-CAM visualizations.

    Each cell shows: original image | heatmap overlay | prediction vs true label

    Args:
        images: List of original images (H, W, 3) in [0, 1]
        heatmaps: List of Grad-CAM heatmaps
        predictions: List of predicted class names
        true_labels: List of true class names
        title: Grid title
        save_path: Where to save
        cols: Number of columns in grid

    Returns:
        Path to saved figure
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 6, rows * 3))

    if rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        row = i // cols
        col = i % cols

        # Original image
        ax_orig = axes[row, col * 2]
        ax_orig.imshow(images[i])
        ax_orig.set_title(f"True: {true_labels[i]}", fontsize=9)
        ax_orig.axis("off")

        # Heatmap overlay
        ax_heat = axes[row, col * 2 + 1]
        overlay_img = overlay_heatmap(images[i], heatmaps[i])
        ax_heat.imshow(overlay_img)
        correct = "✓" if predictions[i] == true_labels[i] else "✗"
        color = "green" if correct == "✓" else "red"
        ax_heat.set_title(f"Pred: {predictions[i]} {correct}", fontsize=9, color=color)
        ax_heat.axis("off")

    # Hide unused axes
    for i in range(n, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col * 2].axis("off")
        axes[row, col * 2 + 1].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path is None:
        os.makedirs("./results/figures", exist_ok=True)
        save_path = "./results/figures/gradcam_grid.png"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"📊 Grad-CAM grid saved: {save_path}")
    plt.close(fig)

    return save_path
