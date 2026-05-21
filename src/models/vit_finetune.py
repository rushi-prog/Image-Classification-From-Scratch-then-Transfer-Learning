"""
Vision Transformer (ViT) Fine-Tuning Module
=============================================
Fine-tune pretrained ViT models using the `timm` library.

Supports:
- Multiple ViT variants (vit_small, vit_base, deit)
- Layer-wise learning rate decay (LLRD)
- Stochastic depth (drop_path)
- Attention dropout

Interview talking points:
    - ViT splits images into patches → treats them as tokens
    - Self-attention has O(n²) complexity with number of patches
    - ViT needs more data than CNNs (no inductive bias for locality)
    - Layer-wise LR decay: deeper layers changed more, shallow layers preserved
    - DeiT showed ViT can work with just ImageNet (not ImageNet-21k)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class ViTFineTune(nn.Module):
    """
    Vision Transformer fine-tuning wrapper using timm.

    Loads a pretrained ViT and replaces the classification head.
    Supports layer-wise learning rate decay (LLRD) for better fine-tuning.
    """

    def __init__(
        self,
        backbone_name: str = "vit_small_patch16_224",
        num_classes: int = 101,
        pretrained: bool = True,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for ViT fine-tuning. "
                "Install with: pip install timm"
            )

        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # Load pretrained ViT from timm
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_layer_groups(self) -> List[List[nn.Parameter]]:
        """
        Group model parameters by layer depth for LLRD.

        Groups (from shallowest to deepest):
        0: patch_embed + pos_embed + cls_token
        1..N: transformer blocks (one group per block)
        N+1: norm + head (classifier)

        Returns:
            List of parameter lists, ordered shallow → deep
        """
        groups = []

        # Group 0: Embeddings (patch + position + cls token)
        embed_params = []
        for name, param in self.model.named_parameters():
            if any(k in name for k in ["patch_embed", "pos_embed", "cls_token"]):
                embed_params.append(param)
        if embed_params:
            groups.append(embed_params)

        # Groups 1..N: Transformer blocks
        if hasattr(self.model, "blocks"):
            for block in self.model.blocks:
                groups.append(list(block.parameters()))
        elif hasattr(self.model, "layers"):
            for layer in self.model.layers:
                groups.append(list(layer.parameters()))

        # Final group: Norm + classifier head
        head_params = []
        for name, param in self.model.named_parameters():
            if any(k in name for k in ["norm", "head", "fc_norm"]):
                if not any(k in name for k in ["patch_embed", "blocks", "layers"]):
                    head_params.append(param)
        if head_params:
            groups.append(head_params)

        return groups

    def get_param_groups_with_llrd(
        self, base_lr: float, decay_rate: float = 0.75, weight_decay: float = 0.05
    ) -> List[dict]:
        """
        Create parameter groups with Layer-wise Learning Rate Decay (LLRD).

        The deepest layers (closest to output) get the base_lr.
        Each shallower layer gets lr * decay_rate.

        This is key for ViT fine-tuning because:
        - Shallow layers capture general visual features → preserve them
        - Deep layers + head need to adapt to new task → higher LR

        Args:
            base_lr: LR for the deepest layer (head)
            decay_rate: Multiply LR by this for each shallower layer
            weight_decay: Weight decay value

        Returns:
            List of param groups for optimizer
        """
        layer_groups = self.get_layer_groups()
        num_groups = len(layer_groups)

        param_groups = []
        for i, group_params in enumerate(layer_groups):
            # Deepest group (last) gets base_lr
            # Each shallower group gets lr * decay_rate
            depth = num_groups - 1 - i  # 0 for deepest, higher for shallower
            lr = base_lr * (decay_rate ** depth)

            # Separate weight decay: no WD for biases and LayerNorm
            decay_params = []
            no_decay_params = []
            for param in group_params:
                if param.dim() >= 2:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)

            if decay_params:
                param_groups.append({
                    "params": decay_params,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "name": f"layer_{i}_decay",
                })
            if no_decay_params:
                param_groups.append({
                    "params": no_decay_params,
                    "lr": lr,
                    "weight_decay": 0.0,
                    "name": f"layer_{i}_no_decay",
                })

        return param_groups

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract attention maps from all transformer blocks.
        Useful for visualization (attention rollout as Grad-CAM alternative).

        Args:
            x: Input images (B, 3, 224, 224)

        Returns:
            Attention weights from last block (B, num_heads, num_patches+1, num_patches+1)
        """
        # Register hooks to capture attention weights
        attention_weights = []

        def hook_fn(module, input, output):
            # timm's Attention module stores attn in forward
            if hasattr(module, 'attn_drop'):
                attention_weights.append(output)

        hooks = []
        if hasattr(self.model, 'blocks'):
            for block in self.model.blocks:
                if hasattr(block, 'attn'):
                    hook = block.attn.register_forward_hook(hook_fn)
                    hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            _ = self.model(x)

        # Clean up hooks
        for hook in hooks:
            hook.remove()

        if attention_weights:
            return attention_weights[-1]  # Last block attention
        return None

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_parameters_millions(self) -> float:
        return self.num_parameters / 1e6


def build_vit_model(config: dict) -> ViTFineTune:
    """
    Factory function to build ViTFineTune from config.
    """
    model_cfg = config["model"]
    data_cfg = config["data"]

    model = ViTFineTune(
        backbone_name=model_cfg.get("backbone", "vit_small_patch16_224"),
        num_classes=data_cfg["num_classes"],
        pretrained=model_cfg.get("pretrained", True),
        drop_rate=model_cfg.get("drop_rate", 0.1),
        attn_drop_rate=model_cfg.get("attn_drop_rate", 0.0),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.1),
    )

    # Print layer group info
    groups = model.get_layer_groups()
    print(f"🏗️  Built ViTFineTune:")
    print(f"   Backbone: {model_cfg.get('backbone', 'vit_small_patch16_224')}")
    print(f"   Total params: {model.num_parameters_millions:.2f}M")
    print(f"   Layer groups for LLRD: {len(groups)}")

    return model
