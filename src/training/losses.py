"""
Loss Functions
===============
CrossEntropy with label smoothing and soft target support
for Mixup/CutMix training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Label smoothing prevents the model from becoming overconfident
    by distributing some probability mass to non-target classes:
        y_smooth = (1 - ε) * y_onehot + ε / num_classes

    Args:
        smoothing: Label smoothing factor ε (typically 0.1)
        reduction: 'mean' or 'sum'
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy.

        Args:
            pred: Model logits (B, num_classes)
            target: Either hard labels (B,) or soft labels (B, num_classes)

        Returns:
            Scalar loss
        """
        if target.dim() == 1:
            # Hard labels → apply label smoothing
            return self._hard_label_loss(pred, target)
        else:
            # Soft labels (from Mixup/CutMix) → use soft CE directly
            return self._soft_label_loss(pred, target)

    def _hard_label_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Label-smoothed CE with hard (integer) targets."""
        num_classes = pred.size(1)
        log_probs = F.log_softmax(pred, dim=1)

        # Create smoothed target distribution
        with torch.no_grad():
            smooth_target = torch.zeros_like(log_probs)
            smooth_target.fill_(self.smoothing / (num_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_target * log_probs).sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def _soft_label_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """CE with soft (probability distribution) targets."""
        log_probs = F.log_softmax(pred, dim=1)
        loss = -(target * log_probs).sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross-entropy loss for soft targets (from Mixup/CutMix).
    No label smoothing — just handles probability distributions as targets.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(pred, dim=1)
        loss = -(target * log_probs).sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_function(config: dict) -> nn.Module:
    """
    Factory function to build the loss function from config.

    Automatically selects the right loss based on:
    - Whether label smoothing is enabled
    - Whether Mixup/CutMix is enabled (needs soft target support)
    """
    training_cfg = config.get("training", {})
    aug_cfg = config.get("augmentation", {})
    smoothing = training_cfg.get("label_smoothing", 0.0)

    mixup_enabled = aug_cfg.get("mixup", {}).get("enabled", False)
    cutmix_enabled = aug_cfg.get("cutmix", {}).get("enabled", False)

    if smoothing > 0 or mixup_enabled or cutmix_enabled:
        print(f"📉 Loss: LabelSmoothingCrossEntropy (smoothing={smoothing})")
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    else:
        print("📉 Loss: CrossEntropyLoss")
        return nn.CrossEntropyLoss()
