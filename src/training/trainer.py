"""
Unified Training Loop
======================
Production-grade training loop with:
- Mixed precision (AMP) for faster training on GPU
- Gradient clipping for stability
- W&B logging integration
- Mixup/CutMix batch augmentation
- LR scheduler support (epoch-level and step-level)
- EarlyStopping + ModelCheckpoint
- Training history tracking

Designed to work with ALL model types (scratch CNN, transfer learning, ViT).
"""

import os
import time
import json
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.data.augmentations import MixupCutMix
from src.training.losses import get_loss_function
from src.training.schedulers import get_scheduler
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.logging_utils import WandBLogger


class Trainer:
    """
    Unified trainer for all model types.

    Handles the complete training lifecycle:
    1. Setup: loss, optimizer, scheduler, callbacks, logger
    2. Training loop with AMP + Mixup/CutMix
    3. Validation loop
    4. Checkpointing + early stopping
    5. History tracking + W&B logging

    Usage:
        trainer = Trainer(model, config, dataloaders)
        history = trainer.train()
        trainer.save_history("results/metrics/history.json")
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        dataloaders: Dict[str, DataLoader],
        device: Optional[str] = None,
    ):
        """
        Args:
            model: The model to train
            config: Full experiment config
            dataloaders: Dict with 'train', 'val', 'test' DataLoaders
            device: Device to train on (auto-detected if None)
        """
        self.model = model
        self.config = config
        self.dataloaders = dataloaders

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)
        print(f"🖥️  Device: {self.device}")

        # Setup components
        self._setup_loss()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_mixup_cutmix()
        self._setup_callbacks()
        self._setup_logger()
        self._setup_amp()

        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }
        self.current_epoch = 0

    def _setup_loss(self):
        """Initialize loss function."""
        self.criterion = get_loss_function(self.config)

    def _setup_optimizer(self):
        """Initialize optimizer with support for discriminative LR and LLRD."""
        opt_cfg = self.config["training"]["optimizer"]
        train_cfg = self.config["training"]
        model_cfg = self.config["model"]
        opt_type = opt_cfg.get("type", "adamw").lower()

        # Determine parameter groups based on model type
        params = self._get_param_groups(opt_cfg, train_cfg, model_cfg)

        if opt_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=opt_cfg["lr"],
                weight_decay=opt_cfg.get("weight_decay", 0.01),
                betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            )
        elif opt_type == "sgd":
            self.optimizer = torch.optim.SGD(
                params,
                lr=opt_cfg["lr"],
                momentum=opt_cfg.get("momentum", 0.9),
                weight_decay=opt_cfg.get("weight_decay", 0.01),
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

        print(f"⚙️  Optimizer: {opt_type} (lr={opt_cfg['lr']})")

    def _get_param_groups(self, opt_cfg, train_cfg, model_cfg):
        """Get parameter groups for discriminative LR / LLRD."""
        from src.models.transfer_learn import TransferLearningModel
        from src.models.vit_finetune import ViTFineTune

        # ViT with Layer-wise LR Decay
        if isinstance(self.model, ViTFineTune):
            llrd_cfg = train_cfg.get("layer_lr_decay", {})
            if llrd_cfg.get("enabled", False):
                decay_rate = llrd_cfg.get("decay_rate", 0.75)
                params = self.model.get_param_groups_with_llrd(
                    base_lr=opt_cfg["lr"],
                    decay_rate=decay_rate,
                    weight_decay=opt_cfg.get("weight_decay", 0.05),
                )
                print(f"📐 LLRD: decay_rate={decay_rate}, {len(params)} param groups")
                return params

        # Transfer learning with discriminative LR
        if isinstance(self.model, TransferLearningModel):
            disc_cfg = train_cfg.get("discriminative_lr", {})
            if disc_cfg.get("enabled", False):
                factor = disc_cfg.get("backbone_lr_factor", 0.1)
                params = self.model.get_param_groups(
                    base_lr=opt_cfg["lr"],
                    backbone_lr_factor=factor,
                )
                print(f"📐 Discriminative LR: backbone={opt_cfg['lr']*factor:.6f}, head={opt_cfg['lr']}")
                return params

        # Default: all params with same LR
        return self.model.parameters()

    def _setup_scheduler(self):
        """Initialize learning rate scheduler."""
        steps_per_epoch = len(self.dataloaders["train"])
        self.scheduler, self.scheduler_step_mode = get_scheduler(
            self.optimizer, self.config, steps_per_epoch
        )

    def _setup_mixup_cutmix(self):
        """Initialize Mixup/CutMix augmentation."""
        aug_cfg = self.config.get("augmentation", {})
        mixup_cfg = aug_cfg.get("mixup", {})
        cutmix_cfg = aug_cfg.get("cutmix", {})

        mixup_enabled = mixup_cfg.get("enabled", False)
        cutmix_enabled = cutmix_cfg.get("enabled", False)

        if mixup_enabled or cutmix_enabled:
            self.mixup_cutmix = MixupCutMix(
                mixup_alpha=mixup_cfg.get("alpha", 0.2) if mixup_enabled else 0,
                cutmix_alpha=cutmix_cfg.get("alpha", 1.0) if cutmix_enabled else 0,
                mix_prob=aug_cfg.get("mix_prob", 0.5),
                num_classes=self.config["data"]["num_classes"],
                enabled=True,
            )
            print(f"🔀 Mixup/CutMix: enabled (mix_prob={aug_cfg.get('mix_prob', 0.5)})")
        else:
            self.mixup_cutmix = None

    def _setup_callbacks(self):
        """Initialize callbacks."""
        cb_cfg = self.config.get("callbacks", {})
        exp_name = self.config.get("experiment", {}).get("name", "")

        # Early stopping
        es_cfg = cb_cfg.get("early_stopping", {})
        if es_cfg.get("enabled", False):
            self.early_stopping = EarlyStopping(
                patience=es_cfg.get("patience", 10),
                monitor=es_cfg.get("monitor", "val_accuracy"),
                mode=es_cfg.get("mode", "max"),
            )
            print(f"🛑 EarlyStopping: patience={es_cfg.get('patience', 10)}")
        else:
            self.early_stopping = None

        # Checkpointing
        cp_cfg = cb_cfg.get("checkpoint", {})
        self.checkpoint = ModelCheckpoint(
            dirpath=cp_cfg.get("dirpath", "./results/checkpoints"),
            monitor=cp_cfg.get("monitor", "val_accuracy"),
            mode=cp_cfg.get("mode", "max"),
            save_best=cp_cfg.get("save_best", True),
            save_last=cp_cfg.get("save_last", True),
            filename_prefix=f"{exp_name}_" if exp_name else "",
        )

    def _setup_logger(self):
        """Initialize W&B logger."""
        self.logger = WandBLogger(self.config)

    def _setup_amp(self):
        """Initialize automatic mixed precision."""
        self.use_amp = (
            self.config["training"].get("mixed_precision", False)
            and self.device.type == "cuda"
        )
        if self.use_amp:
            self.scaler = GradScaler()
            print("⚡ Mixed precision: enabled (FP16)")
        else:
            self.scaler = None

    def train(self) -> Dict[str, List[float]]:
        """
        Run the full training loop.

        Returns:
            Training history dict
        """
        epochs = self.config["training"]["epochs"]
        grad_clip = self.config["training"].get("gradient_clip", 0.0)

        print("\n" + "=" * 60)
        print(f"🚀 Starting training for {epochs} epochs")
        print("=" * 60)

        # Get unfreeze schedule for transfer learning
        unfreeze_schedule = self.config.get("model", {}).get("unfreeze_schedule", None)

        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # ---- Gradual Unfreezing (transfer learning) ----
            if unfreeze_schedule is not None:
                from src.models.transfer_learn import TransferLearningModel
                if isinstance(self.model, TransferLearningModel):
                    self.model.apply_unfreeze_schedule(epoch, unfreeze_schedule)

            # ---- TRAIN ----
            train_loss, train_acc = self._train_one_epoch(grad_clip)

            # ---- VALIDATE ----
            val_loss, val_acc = self._validate()

            # ---- LR tracking ----
            current_lr = self.optimizer.param_groups[0]["lr"]

            # ---- Step scheduler (epoch-level) ----
            if self.scheduler_step_mode == "epoch":
                self.scheduler.step()

            # ---- Record history ----
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)
            self.history["learning_rate"].append(current_lr)

            epoch_time = time.time() - epoch_start

            # ---- Print epoch summary ----
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
                f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% "
                f"| LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
            )

            # ---- Log to W&B ----
            self.logger.log_metrics(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "learning_rate": current_lr,
                    "epoch_time": epoch_time,
                },
                step=epoch,
            )

            # ---- Checkpoint ----
            metrics = {"val_accuracy": val_acc, "val_loss": val_loss}
            self.checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=metrics,
                config=self.config,
            )

            # ---- Early stopping ----
            if self.early_stopping is not None:
                monitor_val = metrics.get(
                    self.config.get("callbacks", {}).get("early_stopping", {}).get("monitor", "val_accuracy"),
                    val_acc,
                )
                if self.early_stopping(monitor_val):
                    print(f"\n🛑 Early stopping at epoch {epoch + 1}")
                    break

        # Finish
        print("\n" + "=" * 60)
        best_val_acc = max(self.history["val_accuracy"])
        best_epoch = self.history["val_accuracy"].index(best_val_acc) + 1
        print(f"🏆 Best Val Accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
        print("=" * 60)

        self.logger.finish()
        return self.history

    def _train_one_epoch(self, grad_clip: float) -> Tuple[float, float]:
        """Train for one epoch. Returns (avg_loss, accuracy%)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.dataloaders["train"],
            desc=f"Epoch {self.current_epoch + 1} [Train]",
            leave=False,
        )

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Apply Mixup/CutMix at batch level
            use_soft = False
            if self.mixup_cutmix is not None:
                images, targets = self.mixup_cutmix(images, targets)
                use_soft = targets.dim() > 1  # Soft labels are 2D

            # Forward pass with AMP
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                self.scaler.scale(loss).backward()

                if grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), grad_clip
                    )

                self.optimizer.step()

            # Step scheduler (batch-level, for OneCycleLR)
            if self.scheduler_step_mode == "step":
                self.scheduler.step()

            # Track metrics
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)

            if use_soft:
                # For soft labels, compare with argmax of soft target
                _, true_labels = targets.max(1)
                correct += predicted.eq(true_labels).sum().item()
            else:
                correct += predicted.eq(targets).sum().item()
            total += images.size(0)

            # Update progress bar
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * correct / total:.1f}%",
            )

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float]:
        """Validate on validation set. Returns (avg_loss, accuracy%)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, targets in self.dataloaders["val"]:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += images.size(0)

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, dataloader_key: str = "test") -> Dict[str, float]:
        """
        Run evaluation on a dataloader.

        Args:
            dataloader_key: 'test' or 'val'

        Returns:
            Dict with loss, accuracy, per-batch predictions
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        loader = self.dataloaders[dataloader_key]

        for images, targets in tqdm(loader, desc=f"Evaluating [{dataloader_key}]"):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += images.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        return {
            "loss": total_loss / total,
            "accuracy": 100.0 * correct / total,
            "predictions": all_preds,
            "targets": all_targets,
        }

    def save_history(self, path: str = "./results/metrics/history.json"):
        """Save training history to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"📊 History saved: {path}")
