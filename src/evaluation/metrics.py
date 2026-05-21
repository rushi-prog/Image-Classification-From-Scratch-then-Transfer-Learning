"""
Evaluation Metrics
===================
Comprehensive metrics for image classification evaluation.

Computes:
- Top-1 and Top-5 accuracy
- Per-class accuracy
- F1 score (macro, weighted)
- Classification report
- Confusion matrix
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        y_prob: Prediction probabilities (N, num_classes) — needed for top-k
        class_names: Names for each class
        top_k: K for top-k accuracy

    Returns:
        Dict with all metrics
    """
    metrics = {}

    # Top-1 accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred) * 100

    # Top-5 accuracy (if probabilities available)
    if y_prob is not None and y_prob.shape[1] >= top_k:
        metrics[f"top{top_k}_accuracy"] = (
            top_k_accuracy_score(y_true, y_prob, k=top_k) * 100
        )

    # F1 scores
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro") * 100
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted") * 100

    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    metrics["per_class_accuracy_mean"] = per_class_acc.mean() * 100
    metrics["per_class_accuracy_std"] = per_class_acc.std() * 100

    # Worst / best classes
    if class_names:
        worst_idx = np.argsort(per_class_acc)[:5]
        best_idx = np.argsort(per_class_acc)[-5:][::-1]
        metrics["worst_classes"] = [
            {"class": class_names[i], "accuracy": per_class_acc[i] * 100}
            for i in worst_idx
        ]
        metrics["best_classes"] = [
            {"class": class_names[i], "accuracy": per_class_acc[i] * 100}
            for i in best_idx
        ]

    return metrics


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Generate a full sklearn classification report string.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names for display

    Returns:
        Formatted classification report string
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=3,
        zero_division=0,
    )


def print_metrics_summary(metrics: Dict, title: str = "Evaluation Results"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  Top-1 Accuracy:   {metrics['accuracy']:.2f}%")
    if "top5_accuracy" in metrics:
        print(f"  Top-5 Accuracy:   {metrics['top5_accuracy']:.2f}%")
    print(f"  F1 (macro):       {metrics['f1_macro']:.2f}%")
    print(f"  F1 (weighted):    {metrics['f1_weighted']:.2f}%")
    print(f"  Per-class Acc:    {metrics['per_class_accuracy_mean']:.2f}% "
          f"(+/- {metrics['per_class_accuracy_std']:.2f}%)")

    if "worst_classes" in metrics:
        print(f"\n  Worst 5 classes:")
        for item in metrics["worst_classes"]:
            print(f"    {item['class']:30s} {item['accuracy']:.1f}%")

    if "best_classes" in metrics:
        print(f"\n  Best 5 classes:")
        for item in metrics["best_classes"]:
            print(f"    {item['class']:30s} {item['accuracy']:.1f}%")

    print(f"{'=' * 60}")
