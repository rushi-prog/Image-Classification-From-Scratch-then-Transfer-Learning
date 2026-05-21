"""
Grad-CAM Visualization Script
===============================
Load a trained model and generate Grad-CAM heatmaps on test images.

Usage:
    python scripts/visualize_gradcam.py --config configs/cnn_scratch.yaml --checkpoint results/checkpoints/cnn_scratch_food101_best.pth --num-images 16
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.data.dataset import get_dataloaders, FOOD101_CLASSES
from src.evaluation.gradcam import (
    GradCAM,
    get_target_layer,
    denormalize_image,
    create_gradcam_grid,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="./results/figures")
    return parser.parse_args()


def build_model(config):
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
    print("🔍 Grad-CAM Visualization")
    print("=" * 60)

    # Setup
    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Build and load model
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Get target layer for Grad-CAM
    target_layer = get_target_layer(model, config)
    gradcam = GradCAM(model, target_layer)
    print(f"   Target layer: {target_layer.__class__.__name__}")

    # Load test data
    dataloaders = get_dataloaders(config)
    test_loader = dataloaders["test"]

    # Get normalization params for denormalization
    aug_cfg = config.get("augmentation", {})
    norm = aug_cfg.get("normalize", None)
    mean = norm["mean"] if norm else [0.5, 0.5, 0.5]
    std = norm["std"] if norm else [0.5, 0.5, 0.5]

    # Collect images
    images_list = []
    heatmaps_list = []
    preds_list = []
    trues_list = []

    print(f"   Generating Grad-CAM for {args.num_images} images...")

    collected = 0
    for images, targets in test_loader:
        for i in range(images.size(0)):
            if collected >= args.num_images:
                break

            img_tensor = images[i:i+1].to(device)

            # Generate heatmap
            with torch.enable_grad():
                heatmap = gradcam.generate(img_tensor)

            # Get prediction
            with torch.no_grad():
                output = model(img_tensor)
                pred_idx = output.argmax(dim=1).item()

            # Denormalize for display
            img_display = denormalize_image(images[i], mean=mean, std=std)

            images_list.append(img_display)
            heatmaps_list.append(heatmap)
            preds_list.append(FOOD101_CLASSES[pred_idx])
            trues_list.append(FOOD101_CLASSES[targets[i].item()])

            collected += 1

        if collected >= args.num_images:
            break

    # Create grid
    exp_name = config.get("experiment", {}).get("name", "model")
    save_path = os.path.join(args.save_dir, f"{exp_name}_gradcam.png")

    create_gradcam_grid(
        images=images_list,
        heatmaps=heatmaps_list,
        predictions=preds_list,
        true_labels=trues_list,
        title=f"Grad-CAM — {exp_name}",
        save_path=save_path,
        cols=4,
    )

    # Also save individual correct vs incorrect samples
    correct_imgs = [(img, hm, p, t) for img, hm, p, t in
                    zip(images_list, heatmaps_list, preds_list, trues_list) if p == t]
    incorrect_imgs = [(img, hm, p, t) for img, hm, p, t in
                      zip(images_list, heatmaps_list, preds_list, trues_list) if p != t]

    acc = len(correct_imgs) / len(images_list) * 100 if images_list else 0
    print(f"\n   Sample accuracy: {acc:.0f}% ({len(correct_imgs)}/{len(images_list)})")
    print(f"   ✅ Correct: {len(correct_imgs)} | ❌ Incorrect: {len(incorrect_imgs)}")
    print(f"\n✅ Grad-CAM visualization complete!")


if __name__ == "__main__":
    main()
