"""
Transfer Learning Module
==========================
Fine-tune pretrained ResNet-50 / EfficientNet-B0 on Food-101.

Supports:
- Feature extraction (freeze backbone, train classifier only)
- Gradual unfreezing (progressively unfreeze deeper layers)
- Full fine-tuning (everything trainable)
- Discriminative learning rates (backbone gets lower LR)

Interview talking points:
    - WHY transfer learning works: ImageNet features generalize
    - Feature extraction vs fine-tuning tradeoffs
    - Discriminative LR: early layers learn universal features (edges, textures),
      later layers learn task-specific features — so early layers need less change
    - Gradual unfreezing prevents catastrophic forgetting
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Optional, Tuple


class TransferLearningModel(nn.Module):
    """
    Transfer learning wrapper for torchvision pretrained models.

    Replaces the classifier head and supports fine-tuning strategies:
    1. Feature extraction: freeze all backbone layers
    2. Gradual unfreezing: unfreeze layers according to schedule
    3. Full fine-tuning: all layers trainable

    Supported backbones: resnet50, efficientnet_b0
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = 101,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # Load pretrained backbone
        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            # Replace classifier
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes),
            )
            self._classifier_name = "fc"

        elif backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            # Replace classifier
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes),
            )
            self._classifier_name = "classifier"

        else:
            raise ValueError(
                f"Unsupported backbone: {backbone_name}. "
                f"Supported: resnet50, efficientnet_b0"
            )

        # Optionally freeze backbone
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Freeze all backbone parameters except the classifier head."""
        for name, param in self.backbone.named_parameters():
            if self._classifier_name not in name:
                param.requires_grad = False
        self._print_trainable_status("After freeze")

    def unfreeze_layers(self, layer_names: List[str]):
        """
        Unfreeze specific layers by name.

        Args:
            layer_names: List of layer name prefixes to unfreeze.
                         Use "all" to unfreeze everything.
        """
        if "all" in layer_names:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            for name, param in self.backbone.named_parameters():
                if any(layer in name for layer in layer_names):
                    param.requires_grad = True

    def apply_unfreeze_schedule(self, epoch: int, schedule: List[dict]):
        """
        Apply gradual unfreezing based on epoch.

        Args:
            epoch: Current training epoch
            schedule: List of dicts with 'epoch' and 'layers' keys
                      e.g., [{"epoch": 0, "layers": ["fc"]},
                             {"epoch": 5, "layers": ["layer4", "fc"]}]
        """
        for entry in schedule:
            if epoch == entry["epoch"]:
                layers = entry["layers"]
                if isinstance(layers, str):
                    layers = [layers]
                self.unfreeze_layers(layers)
                trainable = sum(
                    p.numel() for p in self.backbone.parameters() if p.requires_grad
                )
                total = sum(p.numel() for p in self.backbone.parameters())
                print(
                    f"   🔓 Epoch {epoch}: Unfroze {layers} "
                    f"({trainable / 1e6:.2f}M / {total / 1e6:.2f}M params trainable)"
                )

    def get_param_groups(self, base_lr: float, backbone_lr_factor: float = 0.1) -> List[dict]:
        """
        Create parameter groups with discriminative learning rates.

        Backbone parameters get base_lr * backbone_lr_factor.
        Classifier parameters get base_lr.

        This is important because:
        - Backbone already has good features from ImageNet
        - Classifier needs to learn from scratch
        - Lower LR for backbone prevents catastrophic forgetting

        Args:
            base_lr: Learning rate for the classifier
            backbone_lr_factor: Multiplier for backbone LR (e.g., 0.1 = 10x smaller)

        Returns:
            List of param groups for the optimizer
        """
        classifier_params = []
        backbone_params = []

        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if self._classifier_name in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = []
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": base_lr * backbone_lr_factor,
                "name": "backbone",
            })
        if classifier_params:
            param_groups.append({
                "params": classifier_params,
                "lr": base_lr,
                "name": "classifier",
            })

        return param_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature maps from the last conv layer (for Grad-CAM).
        """
        if self.backbone_name == "resnet50":
            # ResNet: extract up to layer4
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            return x
        elif self.backbone_name == "efficientnet_b0":
            return self.backbone.features(x)

    def _print_trainable_status(self, label: str = ""):
        trainable = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in self.backbone.parameters())
        frozen = total - trainable
        print(
            f"   {label}: {trainable / 1e6:.2f}M trainable, "
            f"{frozen / 1e6:.2f}M frozen (total {total / 1e6:.2f}M)"
        )

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_parameters_millions(self) -> float:
        return self.num_parameters / 1e6


def build_transfer_model(config: dict) -> TransferLearningModel:
    """
    Factory function to build TransferLearningModel from config.
    """
    model_cfg = config["model"]
    data_cfg = config["data"]

    model = TransferLearningModel(
        backbone_name=model_cfg["backbone"],
        num_classes=data_cfg["num_classes"],
        pretrained=model_cfg.get("pretrained", True),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
    )

    print(f"🏗️  Built TransferLearningModel:")
    print(f"   Backbone: {model_cfg['backbone']}")
    print(f"   Pretrained: {model_cfg.get('pretrained', True)}")
    print(f"   Trainable params: {model.num_parameters_millions:.2f}M")

    return model
