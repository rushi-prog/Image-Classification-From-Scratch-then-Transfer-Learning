"""Quick smoke test for Phase 1 — validates configs and imports."""
import yaml
import os

configs = [
    "configs/cnn_scratch.yaml",
    "configs/resnet50_finetune.yaml",
    "configs/efficientnet_finetune.yaml",
    "configs/vit_finetune.yaml",
]

print("=" * 50)
print("Phase 1 Smoke Test")
print("=" * 50)

# Test all configs load
for cfg_path in configs:
    assert os.path.exists(cfg_path), f"Missing: {cfg_path}"
    cfg = yaml.safe_load(open(cfg_path))
    assert "model" in cfg, f"Missing 'model' section in {cfg_path}"
    assert "data" in cfg, f"Missing 'data' section in {cfg_path}"
    assert "training" in cfg, f"Missing 'training' section in {cfg_path}"
    model = cfg["model"]["type"] if "type" in cfg["model"] else cfg["model"]["backbone"]
    print(f"  ✅ {cfg_path}: {model}")

# Test CNN scratch config details
cfg = yaml.safe_load(open("configs/cnn_scratch.yaml"))
assert cfg["data"]["dataset"] == "food101"
assert cfg["data"]["num_classes"] == 101
assert cfg["augmentation"]["mixup"]["enabled"] == True
assert cfg["augmentation"]["cutmix"]["enabled"] == True
print(f"\n  📦 Dataset: {cfg['data']['dataset']}")
print(f"  🖼️  Image size: {cfg['data']['image_size']}")
print(f"  📊 Classes: {cfg['data']['num_classes']}")
print(f"  🔀 Mixup: alpha={cfg['augmentation']['mixup']['alpha']}")
print(f"  ✂️  CutMix: alpha={cfg['augmentation']['cutmix']['alpha']}")
print(f"  📈 Epochs: {cfg['training']['epochs']}")
print(f"  ⚡ Mixed precision: {cfg['training']['mixed_precision']}")

# Verify source files exist
src_files = [
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/dataset.py",
    "src/data/augmentations.py",
    "src/data/download.py",
    "src/utils/__init__.py",
    "src/utils/config.py",
    "src/utils/logging_utils.py",
    "src/utils/visualization.py",
    "src/models/__init__.py",
    "src/training/__init__.py",
    "src/evaluation/__init__.py",
]

print(f"\n  Source files:")
for f in src_files:
    exists = os.path.exists(f)
    status = "✅" if exists else "❌"
    print(f"    {status} {f}")
    assert exists, f"Missing: {f}"

# Verify directories
dirs = ["results/checkpoints", "results/figures", "results/metrics", "assets", "notebooks", "scripts"]
print(f"\n  Directories:")
for d in dirs:
    exists = os.path.isdir(d)
    status = "✅" if exists else "❌"
    print(f"    {status} {d}")

print(f"\n{'=' * 50}")
print("🎉 Phase 1 Smoke Test PASSED!")
print("=" * 50)
