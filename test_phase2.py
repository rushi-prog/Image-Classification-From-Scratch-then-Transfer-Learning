"""Phase 2 smoke test — verify model + training loop work on CPU."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# ---- Test 1: Config loading ----
print("=" * 60)
print("Phase 2 Smoke Test")
print("=" * 60)

from src.utils.config import load_config
config = load_config("configs/cnn_scratch.yaml")
print("[PASS] Config loaded")

# ---- Test 2: Model builds ----
from src.models.cnn_scratch import build_scratch_cnn
model = build_scratch_cnn(config)

# Verify parameter count
params_m = model.num_parameters_millions
print(f"[PASS] Model built: {params_m:.2f}M parameters")
assert 1.0 < params_m < 10.0, f"Unexpected param count: {params_m}M"

# ---- Test 3: Forward pass ----
batch_size = 4
img_size = config["data"]["image_size"]
x = torch.randn(batch_size, 3, img_size, img_size)
out = model(x)
assert out.shape == (batch_size, 101), f"Bad output shape: {out.shape}"
print(f"[PASS] Forward pass: input={x.shape} -> output={out.shape}")

# ---- Test 4: Feature maps for Grad-CAM ----
feat = model.get_feature_maps(x)
print(f"[PASS] Feature maps: {feat.shape}")

# ---- Test 5: Augmentations ----
from src.data.augmentations import get_train_transforms, get_val_transforms, MixupCutMix

train_tf = get_train_transforms(img_size, config.get("augmentation", {}))
val_tf = get_val_transforms(img_size, config.get("augmentation", {}))
print(f"[PASS] Train transforms: {len(train_tf.transforms)} ops")
print(f"[PASS] Val transforms: {len(val_tf.transforms)} ops")

# ---- Test 6: Mixup/CutMix ----
mc = MixupCutMix(
    mixup_alpha=0.2, cutmix_alpha=1.0,
    mix_prob=1.0,  # Force mixing for test
    num_classes=101, enabled=True,
)
images = torch.randn(8, 3, img_size, img_size)
targets = torch.randint(0, 101, (8,))
mixed_img, soft_targets = mc(images, targets)
assert soft_targets.shape == (8, 101), f"Bad soft target shape: {soft_targets.shape}"
assert soft_targets.sum(dim=1).allclose(torch.ones(8)), "Soft targets don't sum to 1"
print(f"[PASS] Mixup/CutMix: soft targets shape={soft_targets.shape}")

# ---- Test 7: Loss function ----
from src.training.losses import get_loss_function
criterion = get_loss_function(config)

# Test with hard labels
loss_hard = criterion(out, torch.randint(0, 101, (batch_size,)))
print(f"[PASS] Loss (hard labels): {loss_hard.item():.4f}")

# Test with soft labels
soft = torch.zeros(batch_size, 101)
soft[:, 0] = 0.7
soft[:, 1] = 0.3
loss_soft = criterion(out, soft)
print(f"[PASS] Loss (soft labels): {loss_soft.item():.4f}")

# ---- Test 8: Scheduler ----
from src.training.schedulers import get_scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler, mode = get_scheduler(optimizer, config, steps_per_epoch=100)
print(f"[PASS] Scheduler: mode={mode}")

# ---- Test 9: Callbacks ----
from src.training.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(patience=5, monitor="val_accuracy", mode="max")
assert not es(0.5)  # First call sets baseline
assert not es(0.6)  # Improvement
assert not es(0.55) # Worse but within patience
print("[PASS] EarlyStopping works")

# ---- Test 10: W&B logger (disabled mode) ----
# Override to disable for test
test_config = config.copy()
test_config["logging"] = {"wandb": {"enabled": False}}
from src.utils.logging_utils import WandBLogger
logger = WandBLogger(test_config)
logger.log_metrics({"test": 1.0})  # Should be no-op
logger.finish()
print("[PASS] W&B logger (disabled mode)")

print()
print("=" * 60)
print("Phase 2 Smoke Test PASSED! All 10 tests OK")
print("=" * 60)
print()
print("Ready for training! Run on Colab with:")
print("  python scripts/train.py --config configs/cnn_scratch.yaml")
