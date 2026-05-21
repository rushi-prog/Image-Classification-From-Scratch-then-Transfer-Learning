"""
Visualization Utilities
========================
Plotting functions for training curves, confusion matrices,
augmentation previews, and results comparison.

All plots are saved to results/figures/ and optionally displayed.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Use non-interactive backend for saving (works in Colab + headless)
matplotlib.use("Agg")

# Global style
plt.style.use("seaborn-v0_8-darkgrid")
SAVE_DIR = "./results/figures"


def _ensure_save_dir(save_dir: str = SAVE_DIR) -> str:
    """Create save directory if it doesn't exist."""
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Curves",
    save_path: Optional[str] = None,
    show: bool = False,
) -> str:
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Dict with keys like 'train_loss', 'val_loss',
                 'train_accuracy', 'val_accuracy'
        title: Plot title
        save_path: Where to save. Auto-generated if None.
        show: Whether to display (set False for Colab batch runs)

    Returns:
        Path to saved figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax = axes[0]
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="Train Loss", color="#FF6B6B", linewidth=2)
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val Loss", color="#4ECDC4", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Accuracy curve
    ax = axes[1]
    if "train_accuracy" in history:
        ax.plot(
            history["train_accuracy"],
            label="Train Accuracy",
            color="#FF6B6B",
            linewidth=2,
        )
    if "val_accuracy" in history:
        ax.plot(
            history["val_accuracy"],
            label="Val Accuracy",
            color="#4ECDC4",
            linewidth=2,
        )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    # Save
    if save_path is None:
        _ensure_save_dir()
        save_path = os.path.join(SAVE_DIR, f"training_curves_{title.replace(' ', '_')}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"📊 Training curves saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)

    return save_path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    top_n: int = 20,
    show: bool = False,
) -> str:
    """
    Plot a confusion matrix heatmap.

    For 101 classes, we show top-N most confused classes by default.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names for axis labels
        title: Plot title
        save_path: Where to save
        top_n: Show top N most confused classes (for readability)
        show: Whether to display

    Returns:
        Path to saved figure
    """
    cm = confusion_matrix(y_true, y_pred)

    if class_names is not None and len(class_names) > top_n:
        # Find top-N most confused classes (highest off-diagonal sum)
        off_diag = cm.copy()
        np.fill_diagonal(off_diag, 0)
        confusion_scores = off_diag.sum(axis=0) + off_diag.sum(axis=1)
        top_indices = np.argsort(confusion_scores)[-top_n:]

        cm = cm[np.ix_(top_indices, top_indices)]
        class_names = [class_names[i] for i in top_indices]
        title = f"{title} (Top {top_n} Most Confused)"

    fig, ax = plt.subplots(figsize=(max(10, len(cm) * 0.5), max(8, len(cm) * 0.4)))

    sns.heatmap(
        cm,
        annot=len(cm) <= 25,  # Show numbers only if manageable
        fmt="d",
        cmap="YlOrRd",
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto",
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("True", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save_path is None:
        _ensure_save_dir()
        save_path = os.path.join(SAVE_DIR, "confusion_matrix.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"📊 Confusion matrix saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)

    return save_path


def plot_augmentation_samples(
    original: np.ndarray,
    augmented_list: List[Tuple[str, np.ndarray]],
    title: str = "Augmentation Preview",
    save_path: Optional[str] = None,
    show: bool = False,
) -> str:
    """
    Show original image alongside augmented versions.

    Args:
        original: Original image (H, W, 3) in [0, 255] uint8
        augmented_list: List of (name, image) tuples
        title: Plot title
        save_path: Where to save
        show: Whether to display

    Returns:
        Path to saved figure
    """
    n = len(augmented_list) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    # Original
    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Augmented versions
    for i, (name, img) in enumerate(augmented_list):
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(name, fontsize=11)
        axes[i + 1].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path is None:
        _ensure_save_dir()
        save_path = os.path.join(SAVE_DIR, "augmentation_preview.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"📊 Augmentation preview saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)

    return save_path


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ["test_accuracy", "test_f1", "params_M"],
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
    show: bool = False,
) -> str:
    """
    Bar chart comparing multiple models across metrics.

    Args:
        results: {model_name: {metric_name: value}}
        metrics: Which metrics to compare
        title: Plot title
        save_path: Where to save
        show: Whether to display

    Returns:
        Path to saved figure
    """
    model_names = list(results.keys())
    n_metrics = len(metrics)
    x = np.arange(len(model_names))
    width = 0.8 / n_metrics

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        values = [results[model].get(metric, 0) for model in model_names]
        bars = ax.bar(x + i * width, values, width, label=metric, color=colors[i % len(colors)])

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Value", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path is None:
        _ensure_save_dir()
        save_path = os.path.join(SAVE_DIR, "model_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"📊 Model comparison saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)

    return save_path


def plot_lr_schedule(
    lrs: List[float],
    title: str = "Learning Rate Schedule",
    save_path: Optional[str] = None,
    show: bool = False,
) -> str:
    """Plot learning rate over training steps/epochs."""
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(lrs, color="#FF6B6B", linewidth=2)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        _ensure_save_dir()
        save_path = os.path.join(SAVE_DIR, "lr_schedule.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"📊 LR schedule saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)

    return save_path
