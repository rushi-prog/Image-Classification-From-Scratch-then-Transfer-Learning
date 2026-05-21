"""Phase 3-5 smoke test — verify transfer learning, ViT, evaluation, Grad-CAM."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

print("=" * 60)
print("Phase 3-5 Smoke Test")
print("=" * 60)

# ---- Phase 3: Transfer Learning ----
print("\n--- Phase 3: Transfer Learning ---")

from src.utils.config import load_config

# Test ResNet-50
config_resnet = load_config("configs/resnet50_finetune.yaml")
from src.models.transfer_learn import build_transfer_model, TransferLearningModel

model_resnet = build_transfer_model(config_resnet)
x = torch.randn(2, 3, 224, 224)
out = model_resnet(x)
assert out.shape == (2, 101), f"ResNet output shape wrong: {out.shape}"
print(f"[PASS] ResNet-50: input={x.shape} -> output={out.shape}")

# Test gradual unfreezing
model_resnet.apply_unfreeze_schedule(0, config_resnet["model"]["unfreeze_schedule"])
model_resnet.apply_unfreeze_schedule(3, config_resnet["model"]["unfreeze_schedule"])
print("[PASS] Gradual unfreezing works")

# Test discriminative LR
param_groups = model_resnet.get_param_groups(base_lr=0.001, backbone_lr_factor=0.1)
print(f"[PASS] Discriminative LR: {len(param_groups)} param groups")

# Test feature maps for Grad-CAM
feat = model_resnet.get_feature_maps(x)
print(f"[PASS] ResNet feature maps: {feat.shape}")

# Test EfficientNet-B0
config_eff = load_config("configs/efficientnet_finetune.yaml")
model_eff = build_transfer_model(config_eff)
out_eff = model_eff(x)
assert out_eff.shape == (2, 101)
feat_eff = model_eff.get_feature_maps(x)
print(f"[PASS] EfficientNet-B0: output={out_eff.shape}, features={feat_eff.shape}")

# ---- Phase 4: ViT ----
print("\n--- Phase 4: ViT Fine-Tuning ---")

try:
    import timm
    config_vit = load_config("configs/vit_finetune.yaml")
    from src.models.vit_finetune import build_vit_model

    model_vit = build_vit_model(config_vit)
    out_vit = model_vit(x)
    assert out_vit.shape == (2, 101), f"ViT output wrong: {out_vit.shape}"
    print(f"[PASS] ViT: input={x.shape} -> output={out_vit.shape}")

    # Test LLRD param groups
    llrd_groups = model_vit.get_param_groups_with_llrd(base_lr=0.00005, decay_rate=0.75)
    print(f"[PASS] LLRD: {len(llrd_groups)} param groups")

    # Test layer groups
    layer_groups = model_vit.get_layer_groups()
    print(f"[PASS] Layer groups: {len(layer_groups)} groups")

except ImportError:
    print("[SKIP] timm not installed - ViT tests skipped")
    print("       Install with: uv add timm")

# ---- Phase 5: Evaluation ----
print("\n--- Phase 5: Evaluation + Grad-CAM ---")

from src.evaluation.metrics import compute_metrics, print_metrics_summary

# Fake predictions for testing
y_true = np.random.randint(0, 101, size=100)
y_pred = y_true.copy()
y_pred[:20] = (y_pred[:20] + 1) % 101  # 80% accuracy

from src.data.dataset import FOOD101_CLASSES
metrics = compute_metrics(y_true, y_pred, class_names=FOOD101_CLASSES)
assert 79 < metrics["accuracy"] < 81, f"Expected ~80% accuracy, got {metrics['accuracy']}"
print(f"[PASS] Metrics: accuracy={metrics['accuracy']:.1f}%, f1_macro={metrics['f1_macro']:.1f}%")

# Test Grad-CAM
from src.evaluation.gradcam import GradCAM, get_target_layer, denormalize_image

# Use scratch CNN for Grad-CAM test (simpler)
from src.models.cnn_scratch import build_scratch_cnn
config_cnn = load_config("configs/cnn_scratch.yaml")
model_cnn = build_scratch_cnn(config_cnn)
target_layer = get_target_layer(model_cnn, config_cnn)
gradcam = GradCAM(model_cnn, target_layer)
print(f"[PASS] Grad-CAM initialized, target: {target_layer.__class__.__name__}")

# Generate heatmap
test_img = torch.randn(1, 3, 64, 64)
heatmap = gradcam.generate(test_img)
assert heatmap.shape[0] > 0 and heatmap.shape[1] > 0
assert 0 <= heatmap.min() and heatmap.max() <= 1.0
print(f"[PASS] Grad-CAM heatmap: shape={heatmap.shape}, range=[{heatmap.min():.2f}, {heatmap.max():.2f}]")

# Test denormalize
denorm = denormalize_image(test_img[0], mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
assert denorm.shape == (64, 64, 3)
print(f"[PASS] Denormalize: {denorm.shape}")

print(f"\n{'=' * 60}")
print("Phase 3-5 Smoke Test PASSED!")
print("=" * 60)
