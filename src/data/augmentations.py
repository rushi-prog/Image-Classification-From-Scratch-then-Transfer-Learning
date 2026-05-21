"""
Data Augmentation Pipeline
===========================
Standard torchvision transforms + Mixup + CutMix implementations.

Mixup: Blends two images & labels with a random ratio
CutMix: Pastes a random patch from one image onto another, mixes labels by area ratio

Both are applied at the BATCH level (after DataLoader), not per-image.
This is the correct implementation — many tutorials get this wrong.
"""

import numpy as np
import torch
import torchvision.transforms as T
from typing import Tuple, Optional


def get_train_transforms(image_size: int, aug_config: dict) -> T.Compose:
    """
    Build training transform pipeline from config.

    Args:
        image_size: Target image size (square)
        aug_config: Augmentation config dict

    Returns:
        Composed transform pipeline
    """
    transforms_list = []

    # Resize to target size
    transforms_list.append(T.Resize((image_size, image_size)))

    # Random horizontal flip
    if aug_config.get("random_horizontal_flip", True):
        transforms_list.append(T.RandomHorizontalFlip(p=0.5))

    # Random crop with padding
    if aug_config.get("random_crop", False):
        padding = aug_config.get("random_crop_padding", 4)
        transforms_list.append(
            T.RandomCrop(image_size, padding=padding, padding_mode="reflect")
        )

    # Color jitter
    cj = aug_config.get("color_jitter", None)
    if cj:
        transforms_list.append(
            T.ColorJitter(
                brightness=cj.get("brightness", 0),
                contrast=cj.get("contrast", 0),
                saturation=cj.get("saturation", 0),
                hue=cj.get("hue", 0),
            )
        )

    # Convert to tensor
    transforms_list.append(T.ToTensor())

    # Normalize (ImageNet stats for pretrained models, or dataset stats)
    norm = aug_config.get("normalize", None)
    if norm:
        transforms_list.append(T.Normalize(mean=norm["mean"], std=norm["std"]))
    else:
        # Default: basic [0,1] range normalization (for scratch CNN)
        transforms_list.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    # Random erasing (applied after ToTensor)
    if aug_config.get("random_erasing", False):
        transforms_list.append(
            T.RandomErasing(
                p=aug_config.get("random_erasing_prob", 0.25),
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
            )
        )

    return T.Compose(transforms_list)


def get_val_transforms(image_size: int, aug_config: dict) -> T.Compose:
    """
    Build validation/test transform pipeline.
    Only resize + normalize — no random augmentations.
    """
    transforms_list = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ]

    norm = aug_config.get("normalize", None)
    if norm:
        transforms_list.append(T.Normalize(mean=norm["mean"], std=norm["std"]))
    else:
        transforms_list.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    return T.Compose(transforms_list)


# ============================================================
# Mixup & CutMix — Batch-Level Augmentations
# ============================================================


class Mixup:
    """
    Mixup augmentation (Zhang et al., 2018).

    Blends two images and their labels:
        x_mixed = λ * x_i + (1 - λ) * x_j
        y_mixed = λ * y_i + (1 - λ) * y_j

    where λ ~ Beta(α, α)

    Applied at the BATCH level, not per-image.
    """

    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: Beta distribution parameter. Higher = more mixing.
                   α=0.2 is standard, α=0.4 for more aggressive mixing.
        """
        self.alpha = alpha

    def __call__(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply Mixup to a batch.

        Args:
            images: Batch of images (B, C, H, W)
            targets: Batch of labels (B,)

        Returns:
            mixed_images: Blended images
            targets_a: Original labels
            targets_b: Shuffled labels
            lam: Mixing coefficient
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = images.size(0)
        # Random permutation for pairing
        index = torch.randperm(batch_size, device=images.device)

        mixed_images = lam * images + (1 - lam) * images[index]
        targets_a = targets
        targets_b = targets[index]

        return mixed_images, targets_a, targets_b, lam


class CutMix:
    """
    CutMix augmentation (Yun et al., 2019).

    Cuts a random patch from one image and pastes it onto another.
    Labels are mixed proportionally to the area of the patch.

    Applied at the BATCH level.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Beta distribution parameter for the area ratio.
                   α=1.0 is standard (uniform distribution of area).
        """
        self.alpha = alpha

    def __call__(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix to a batch.

        Args:
            images: Batch of images (B, C, H, W)
            targets: Batch of labels (B,)

        Returns:
            mixed_images: Images with pasted patches
            targets_a: Original labels
            targets_b: Pasted patch labels
            lam: Effective mixing ratio (adjusted by actual bbox area)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)

        # Generate random bounding box
        _, _, H, W = images.shape
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(H, W, lam)

        # Paste the patch
        mixed_images = images.clone()
        mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]

        # Adjust lambda by actual area ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (H * W)

        targets_a = targets
        targets_b = targets[index]

        return mixed_images, targets_a, targets_b, lam

    @staticmethod
    def _rand_bbox(H: int, W: int, lam: float) -> Tuple[int, int, int, int]:
        """Generate a random bounding box with area ratio (1-lam)."""
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        # Random center
        cx = np.random.randint(H)
        cy = np.random.randint(W)

        # Clamp to image boundaries
        bbx1 = np.clip(cx - cut_h // 2, 0, H)
        bby1 = np.clip(cy - cut_w // 2, 0, W)
        bbx2 = np.clip(cx + cut_h // 2, 0, H)
        bby2 = np.clip(cy + cut_w // 2, 0, W)

        return bbx1, bby1, bbx2, bby2


class MixupCutMix:
    """
    Wrapper that randomly applies either Mixup or CutMix per batch.
    This is the standard approach used in DeiT, EfficientNetV2, etc.
    """

    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        mix_prob: float = 0.5,
        switch_prob: float = 0.5,
        num_classes: int = 101,
        enabled: bool = True,
    ):
        """
        Args:
            mixup_alpha: Mixup Beta parameter
            cutmix_alpha: CutMix Beta parameter
            mix_prob: Probability of applying any mixing per batch
            switch_prob: When mixing, probability of CutMix vs Mixup
            num_classes: Number of classes (for one-hot conversion)
            enabled: Master switch
        """
        self.mixup = Mixup(alpha=mixup_alpha)
        self.cutmix = CutMix(alpha=cutmix_alpha)
        self.mix_prob = mix_prob
        self.switch_prob = switch_prob
        self.num_classes = num_classes
        self.enabled = enabled

    def __call__(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Mixup or CutMix to a batch.

        Returns:
            images: (possibly mixed) images
            targets: Soft labels (B, num_classes) — one-hot blended

        Note: Returns SOFT LABELS. The loss function must support
              soft targets (use our mixup_criterion or soft CE).
        """
        if not self.enabled or np.random.rand() > self.mix_prob:
            # No mixing — return one-hot hard labels
            return images, self._one_hot(targets)

        # Choose Mixup or CutMix
        if np.random.rand() < self.switch_prob:
            mixed_images, targets_a, targets_b, lam = self.cutmix(images, targets)
        else:
            mixed_images, targets_a, targets_b, lam = self.mixup(images, targets)

        # Create soft labels
        targets_one_hot = (
            lam * self._one_hot(targets_a) + (1 - lam) * self._one_hot(targets_b)
        )

        return mixed_images, targets_one_hot

    def _one_hot(self, targets: torch.Tensor) -> torch.Tensor:
        """Convert integer labels to one-hot vectors."""
        return torch.nn.functional.one_hot(
            targets.long(), num_classes=self.num_classes
        ).float()


def mixup_criterion(
    criterion: torch.nn.Module,
    pred: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Compute the Mixup/CutMix loss as a weighted combination.

    Use this if you're working with hard labels + separate targets_a/b
    instead of soft labels.

    Args:
        criterion: Loss function (e.g., CrossEntropyLoss)
        pred: Model predictions (B, num_classes)
        targets_a: Original labels
        targets_b: Mixed labels
        lam: Mixing coefficient

    Returns:
        Weighted loss
    """
    return lam * criterion(pred, targets_a) + (1 - lam) * criterion(pred, targets_b)
