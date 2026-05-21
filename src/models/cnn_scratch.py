"""
Custom CNN From Scratch
========================
A deliberately hand-designed CNN for Food-101 classification.
NOT using any pretrained weights — everything trained from random init.

Architecture:
    5 ConvBlocks → Global Average Pooling → Classifier

Each ConvBlock:
    Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU → MaxPool → Dropout

Design choices (interview talking points):
    - BatchNorm after every conv: stabilizes training, allows higher LR
    - Two convs per block: more expressiveness before spatial reduction
    - Global Average Pooling: reduces overfitting vs flatten (no FC explosion)
    - Dropout between blocks: regularization for small dataset
    - ~3M parameters: intentionally constrained to show architecture skill
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ConvBlock(nn.Module):
    """
    Double convolution block with BatchNorm, ReLU, MaxPool, and Dropout.

    Conv → BN → ReLU → Conv → BN → ReLU → MaxPool → Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
        pool: bool = True,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(p=dropout_rate))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ScratchCNN(nn.Module):
    """
    Custom CNN for image classification, built entirely from scratch.

    Architecture:
        Input (3, H, W)
        → ConvBlock(3 → 64)    → (64, H/2, W/2)
        → ConvBlock(64 → 128)  → (128, H/4, W/4)
        → ConvBlock(128 → 256) → (256, H/8, W/8)
        → ConvBlock(256 → 512) → (512, H/16, W/16)
        → ConvBlock(512 → 512) → (512, H/32, W/32)
        → GlobalAvgPool         → (512,)
        → FC(512 → num_classes)

    For 64×64 input: final feature map is 2×2 before GAP
    For 224×224 input: final feature map is 7×7 before GAP

    Args:
        num_classes: Number of output classes
        channels: List of channel sizes for each block
        dropout_rate: Dropout probability between blocks
        use_batch_norm: Whether to use BatchNorm (should always be True)
        in_channels: Input image channels (3 for RGB)
    """

    def __init__(
        self,
        num_classes: int = 101,
        channels: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        in_channels: int = 3,
    ):
        super().__init__()

        if channels is None:
            channels = [64, 128, 256, 512, 512]

        # Build convolutional feature extractor
        blocks = []
        prev_channels = in_channels
        for i, out_ch in enumerate(channels):
            # Increase dropout progressively (deeper = more dropout)
            block_dropout = dropout_rate * (i + 1) / len(channels)
            blocks.append(
                ConvBlock(
                    in_channels=prev_channels,
                    out_channels=out_ch,
                    dropout_rate=block_dropout,
                    pool=True,
                )
            )
            prev_channels = out_ch

        self.features = nn.Sequential(*blocks)

        # Global Average Pooling — reduces (B, C, H, W) → (B, C)
        # This is KEY: no giant FC layer, much fewer parameters
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(channels[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )

        # Weight initialization — Kaiming for ReLU networks
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using Kaiming Normal (He initialization).
        This is the correct init for ReLU-based networks.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Logits (B, num_classes)
        """
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  # (B, C, 1, 1) → (B, C)
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature maps from the last conv block (for Grad-CAM).

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Feature maps from last conv block (B, C, H', W')
        """
        return self.features(x)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_parameters_millions(self) -> float:
        """Total trainable parameters in millions."""
        return self.num_parameters / 1e6


def build_scratch_cnn(config: dict) -> ScratchCNN:
    """
    Factory function to build ScratchCNN from config.

    Args:
        config: Full experiment config dict

    Returns:
        Initialized ScratchCNN model
    """
    model_cfg = config["model"]
    data_cfg = config["data"]

    model = ScratchCNN(
        num_classes=data_cfg["num_classes"],
        channels=model_cfg.get("channels", [64, 128, 256, 512, 512]),
        dropout_rate=model_cfg.get("dropout_rate", 0.3),
        use_batch_norm=model_cfg.get("use_batch_norm", True),
    )

    print(f"🏗️  Built ScratchCNN:")
    print(f"   Parameters: {model.num_parameters_millions:.2f}M")
    print(f"   Channels: {model_cfg.get('channels', [64, 128, 256, 512, 512])}")
    print(f"   Num classes: {data_cfg['num_classes']}")

    return model
