"""
Training Callbacks
===================
EarlyStopping and ModelCheckpoint callbacks for training control.
"""

import os
import json
import torch
from typing import Optional


class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Args:
        patience: Epochs to wait before stopping
        monitor: Metric name to monitor
        mode: 'min' (for loss) or 'max' (for accuracy)
        min_delta: Minimum change to qualify as improvement
    """

    def __init__(
        self,
        patience: int = 10,
        monitor: str = "val_accuracy",
        mode: str = "max",
        min_delta: float = 0.0,
        verbose: bool = True,
    ):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.should_stop = False

        if mode == "max":
            self.is_better = lambda curr, best: curr > best + min_delta
        else:
            self.is_better = lambda curr, best: curr < best - min_delta

    def __call__(self, current_value: float) -> bool:
        """
        Check if training should stop.

        Args:
            current_value: Current value of the monitored metric

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = current_value
            return False

        if self.is_better(current_value, self.best_score):
            self.best_score = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"   EarlyStopping: {self.counter}/{self.patience} "
                    f"(best {self.monitor}={self.best_score:.4f})"
                )
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"   EarlyStopping triggered after {self.patience} epochs without improvement.")
                return True

        return False


class ModelCheckpoint:
    """
    Save model checkpoints based on a monitored metric.

    Saves:
    - Best model (based on monitored metric)
    - Last model (every epoch, overwritten)
    - Checkpoint includes: model state, optimizer state, epoch, metrics

    Args:
        dirpath: Directory to save checkpoints
        monitor: Metric to monitor
        mode: 'min' or 'max'
        save_best: Whether to save best model
        save_last: Whether to save last model each epoch
        filename_prefix: Prefix for checkpoint filenames
    """

    def __init__(
        self,
        dirpath: str = "./results/checkpoints",
        monitor: str = "val_accuracy",
        mode: str = "max",
        save_best: bool = True,
        save_last: bool = True,
        filename_prefix: str = "",
        verbose: bool = True,
    ):
        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.filename_prefix = filename_prefix
        self.verbose = verbose
        self.best_score = None

        if mode == "max":
            self.is_better = lambda curr, best: curr > best
        else:
            self.is_better = lambda curr, best: curr < best

        os.makedirs(dirpath, exist_ok=True)

    def __call__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict,
        config: Optional[dict] = None,
    ) -> Optional[str]:
        """
        Save checkpoint if metric improved (or save_last).

        Args:
            model: The model to save
            optimizer: The optimizer to save
            epoch: Current epoch number
            metrics: Dict of current metrics
            config: Experiment config (saved with checkpoint)

        Returns:
            Path to saved best checkpoint, or None
        """
        current_value = metrics.get(self.monitor, None)
        saved_best_path = None

        # Checkpoint payload
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        }

        # Save last
        if self.save_last:
            last_path = os.path.join(
                self.dirpath, f"{self.filename_prefix}last.pth"
            )
            torch.save(checkpoint, last_path)

        # Save best
        if self.save_best and current_value is not None:
            if self.best_score is None or self.is_better(current_value, self.best_score):
                self.best_score = current_value
                best_path = os.path.join(
                    self.dirpath, f"{self.filename_prefix}best.pth"
                )
                torch.save(checkpoint, best_path)
                saved_best_path = best_path
                if self.verbose:
                    print(
                        f"   Checkpoint: saved best model "
                        f"({self.monitor}={current_value:.4f})"
                    )

        return saved_best_path

    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> dict:
        """
        Load the best checkpoint.

        Args:
            model: Model to load weights into
            optimizer: Optional optimizer to restore
            device: Device to load to

        Returns:
            Checkpoint dict with epoch, metrics, etc.
        """
        best_path = os.path.join(
            self.dirpath, f"{self.filename_prefix}best.pth"
        )
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"No best checkpoint found at {best_path}")

        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"   Loaded best checkpoint (epoch {checkpoint['epoch']}, "
              f"{self.monitor}={checkpoint['metrics'].get(self.monitor, 'N/A')})")

        return checkpoint
