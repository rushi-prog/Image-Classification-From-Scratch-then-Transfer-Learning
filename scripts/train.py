"""
Training Script — CLI Entry Point
===================================
Run training from the command line:

    python scripts/train.py --config configs/cnn_scratch.yaml
    python scripts/train.py --config configs/resnet50_finetune.yaml
    python scripts/train.py --config configs/vit_finetune.yaml

Supports overriding any config value:
    python scripts/train.py --config configs/cnn_scratch.yaml --epochs 50 --lr 0.0005
"""

import os
import sys
import argparse
import random
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config, merge_configs
from src.data.dataset import get_dataloaders
from src.models.cnn_scratch import build_scratch_cnn
from src.training.trainer import Trainer
from src.utils.visualization import plot_training_curves, plot_lr_schedule


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎲 Seed: {seed}")


def build_model(config: dict) -> torch.nn.Module:
    """Build model based on config type."""
    model_type = config["model"]["type"]

    if model_type == "cnn_scratch":
        return build_scratch_cnn(config)
    elif model_type == "transfer_learning":
        # Will be added in Phase 3
        from src.models.transfer_learn import build_transfer_model
        return build_transfer_model(config)
    elif model_type == "vit":
        # Will be added in Phase 4
        from src.models.vit_finetune import build_vit_model
        return build_vit_model(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an image classification model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train CNN from scratch on Food-101
    python scripts/train.py --config configs/cnn_scratch.yaml

    # Train with custom epochs and learning rate
    python scripts/train.py --config configs/cnn_scratch.yaml --epochs 50 --lr 0.0005

    # Train ResNet-50 (transfer learning)
    python scripts/train.py --config configs/resnet50_finetune.yaml

    # Train ViT
    python scripts/train.py --config configs/vit_finetune.yaml
        """,
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Override learning rate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable W&B logging"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda/cpu/mps)"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override random seed"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    print("=" * 60)
    print("🖼️  Tier 1 — Image Classification Training")
    print("=" * 60)
    print(f"📄 Config: {args.config}")

    config = load_config(args.config)

    # Apply CLI overrides
    overrides = {}
    if args.epochs is not None:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.lr is not None:
        overrides.setdefault("training", {}).setdefault("optimizer", {})["lr"] = args.lr
    if args.batch_size is not None:
        overrides.setdefault("data", {})["batch_size"] = args.batch_size
    if args.no_wandb:
        overrides.setdefault("logging", {}).setdefault("wandb", {})["enabled"] = False
    if args.seed is not None:
        overrides.setdefault("experiment", {})["seed"] = args.seed

    if overrides:
        config = merge_configs(config, overrides)
        print(f"   CLI overrides applied: {list(overrides.keys())}")

    # Set seed
    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    # Build data
    dataloaders = get_dataloaders(config)

    # Build model
    model = build_model(config)

    # Train
    trainer = Trainer(
        model=model,
        config=config,
        dataloaders=dataloaders,
        device=args.device,
    )

    history = trainer.train()

    # Save history
    exp_name = config.get("experiment", {}).get("name", "experiment")
    history_path = f"./results/metrics/{exp_name}_history.json"
    trainer.save_history(history_path)

    # Plot training curves
    plot_training_curves(
        history,
        title=exp_name,
        save_path=f"./results/figures/{exp_name}_training_curves.png",
    )

    # Plot LR schedule
    plot_lr_schedule(
        history["learning_rate"],
        title=f"{exp_name} — LR Schedule",
        save_path=f"./results/figures/{exp_name}_lr_schedule.png",
    )

    # Final evaluation on test set
    print("\n📊 Running final evaluation on test set...")
    test_results = trainer.evaluate("test")
    print(f"🎯 Test Accuracy: {test_results['accuracy']:.2f}%")
    print(f"📉 Test Loss: {test_results['loss']:.4f}")

    print("\n✅ Training complete!")
    print(f"   Checkpoints: results/checkpoints/")
    print(f"   History: {history_path}")
    print(f"   Figures: results/figures/")


if __name__ == "__main__":
    main()
