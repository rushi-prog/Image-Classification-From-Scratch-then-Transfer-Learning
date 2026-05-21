"""
Weights & Biases Integration
==============================
W&B logging wrapper with graceful fallback when W&B is not configured.

Usage:
    logger = WandBLogger(config)
    logger.log_metrics({"loss": 0.5, "accuracy": 0.92}, step=100)
    logger.log_images(images, predictions, labels, step=100)
    logger.finish()

Set config.logging.wandb.enabled = True and provide your entity
after creating a W&B account at https://wandb.ai
"""

import os
from typing import Dict, List, Optional, Any

import torch
import numpy as np

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBLogger:
    """
    Weights & Biases logger with graceful degradation.
    Falls back to console-only logging when W&B is not available or disabled.
    """

    def __init__(self, config: dict):
        """
        Initialize W&B logger.

        Args:
            config: Full experiment configuration dict
        """
        self.config = config
        log_cfg = config.get("logging", {}).get("wandb", {})
        self.enabled = log_cfg.get("enabled", False) and WANDB_AVAILABLE
        self.run = None

        if self.enabled:
            self._init_wandb(config, log_cfg)
        elif log_cfg.get("enabled", False) and not WANDB_AVAILABLE:
            print("⚠️  W&B logging enabled in config but 'wandb' package not installed.")
            print("   Install with: pip install wandb")
            print("   Falling back to console-only logging.")

    def _init_wandb(self, config: dict, log_cfg: dict) -> None:
        """Initialize W&B run."""
        exp_cfg = config.get("experiment", {})
        try:
            self.run = wandb.init(
                project=log_cfg.get("project", "tier1-image-classification"),
                entity=log_cfg.get("entity", None),
                name=exp_cfg.get("name", None),
                tags=exp_cfg.get("tags", []),
                config=config,  # Log full config
                reinit=True,
            )
            print(f"✅ W&B initialized: {self.run.url}")
        except Exception as e:
            print(f"⚠️  W&B init failed: {e}")
            print("   Continuing with console-only logging.")
            self.enabled = False

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log scalar metrics.

        Args:
            metrics: Dict of metric_name → value
            step: Global step (optional, W&B auto-increments if not given)
        """
        if self.enabled and self.run:
            wandb.log(metrics, step=step)

    def log_images(
        self,
        images: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        class_names: Optional[List[str]] = None,
        step: Optional[int] = None,
        caption_prefix: str = "pred",
        max_images: int = 16,
    ) -> None:
        """
        Log a grid of images with predictions to W&B.

        Args:
            images: Batch of images (B, C, H, W)
            predictions: Predicted class indices (B,)
            labels: True class indices (B,)
            class_names: List of class names for captions
            step: Global step
            caption_prefix: Prefix for image captions
            max_images: Maximum images to log
        """
        if not self.enabled or not self.run:
            return

        n = min(max_images, images.size(0))
        log_images = []

        for i in range(n):
            img = images[i].cpu()
            pred = predictions[i].item()
            true = labels[i].item()

            pred_name = class_names[pred] if class_names else str(pred)
            true_name = class_names[true] if class_names else str(true)
            correct = "✅" if pred == true else "❌"

            caption = f"{correct} pred={pred_name} | true={true_name}"

            # Denormalize for display
            img = self._denormalize(img)
            log_images.append(wandb.Image(img, caption=caption))

        wandb.log({f"{caption_prefix}_samples": log_images}, step=step)

    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        step: Optional[int] = None,
    ) -> None:
        """Log a confusion matrix to W&B."""
        if not self.enabled or not self.run:
            return

        wandb.log(
            {
                "confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=y_true,
                    preds=y_pred,
                    class_names=class_names,
                )
            },
            step=step,
        )

    def log_model(self, model_path: str, name: str = "model") -> None:
        """Log a model checkpoint as a W&B artifact."""
        if not self.enabled or not self.run:
            return

        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(model_path)
        self.run.log_artifact(artifact)

    def watch_model(self, model: torch.nn.Module, log_freq: int = 100) -> None:
        """Watch model gradients and parameters."""
        if self.enabled and self.run:
            wandb.watch(model, log="all", log_freq=log_freq)

    def finish(self) -> None:
        """Finish the W&B run."""
        if self.enabled and self.run:
            self.run.finish()
            print("✅ W&B run finished.")

    @staticmethod
    def _denormalize(img: torch.Tensor) -> torch.Tensor:
        """
        Denormalize an image tensor for display.
        Handles both ImageNet and [0.5, 0.5, 0.5] normalization.
        """
        img = img.clone()
        # Clamp to valid range
        for t in img:
            t.clamp_(-3, 3)  # Reasonable range for normalized images
        # Simple clamp to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img
