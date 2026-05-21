"""
Evaluation Script
==================
Load a trained checkpoint and run full evaluation on the test set.

Usage:
    python scripts/evaluate.py --config configs/cnn_scratch.yaml --checkpoint results/checkpoints/cnn_scratch_food101_best.pth
"""

import os
import sys
import argparse
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.data.dataset import get_dataloaders, FOOD101_CLASSES
from src.evaluation.metrics import compute_metrics, get_classification_report, print_metrics_summary
from src.utils.visualization import plot_confusion_matrix
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path (.pth)")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def build_model(config):
    """Build model from config (same as train.py)."""
    model_type = config["model"]["type"]
    if model_type == "cnn_scratch":
        from src.models.cnn_scratch import build_scratch_cnn
        return build_scratch_cnn(config)
    elif model_type == "transfer_learning":
        from src.models.transfer_learn import build_transfer_model
        return build_transfer_model(config)
    elif model_type == "vit":
        from src.models.vit_finetune import build_vit_model
        return build_vit_model(config)
    raise ValueError(f"Unknown model type: {model_type}")


def main():
    args = parse_args()

    print("=" * 60)
    print("📊 Model Evaluation")
    print("=" * 60)

    # Load config and checkpoint
    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Build model and load weights
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"   Loaded checkpoint: {args.checkpoint}")
    print(f"   Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Device: {device}")

    # Load data
    dataloaders = get_dataloaders(config)
    loader = dataloaders[args.split]

    # Run evaluation
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc=f"Evaluating [{args.split}]"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names=FOOD101_CLASSES)
    print_metrics_summary(metrics, title=f"Test Results — {config['experiment']['name']}")

    # Classification report
    report = get_classification_report(y_true, y_pred, class_names=FOOD101_CLASSES)
    print("\nClassification Report (top/bottom shown):")
    lines = report.strip().split("\n")
    # Show header + first 10 + last 10 + summary
    for line in lines[:12]:
        print(f"  {line}")
    print("  ...")
    for line in lines[-12:]:
        print(f"  {line}")

    # Save metrics
    exp_name = config.get("experiment", {}).get("name", "model")
    metrics_path = f"./results/metrics/{exp_name}_eval_results.json"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    # Convert non-serializable items
    save_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
    with open(metrics_path, "w") as f:
        json.dump(save_metrics, f, indent=2)
    print(f"\n📁 Metrics saved: {metrics_path}")

    # Confusion matrix
    cm_path = f"./results/figures/{exp_name}_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names=FOOD101_CLASSES,
                          title=f"Confusion Matrix — {exp_name}", save_path=cm_path)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
