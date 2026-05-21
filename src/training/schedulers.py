"""
Learning Rate Schedulers
=========================
Wrappers around PyTorch schedulers with warmup support.
"""

import math
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    _LRScheduler,
)


class LinearWarmupScheduler(_LRScheduler):
    """
    Linear warmup followed by another scheduler.

    During warmup:
        lr = start_lr + (target_lr - start_lr) * (step / warmup_steps)

    After warmup: delegates to the wrapped scheduler.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        start_lr: float,
        after_scheduler: _LRScheduler,
        steps_per_epoch: int = 1,
    ):
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.start_lr = start_lr
        self.after_scheduler = after_scheduler
        self.target_lrs = [group["lr"] for group in optimizer.param_groups]
        self.finished_warmup = False
        self._step_count = 0
        # Set initial_lr to avoid warning on first step
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        super().__init__(optimizer, last_epoch=0)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            # Linear warmup
            progress = self._step_count / max(self.warmup_steps, 1)
            return [
                self.start_lr + (target - self.start_lr) * progress
                for target in self.target_lrs
            ]
        else:
            if not self.finished_warmup:
                self.finished_warmup = True
                # Reset the after_scheduler
                for group, lr in zip(self.optimizer.param_groups, self.target_lrs):
                    group["lr"] = lr
            return self.after_scheduler.get_last_lr()

    def step(self, epoch=None):
        self._step_count += 1
        if self._step_count <= self.warmup_steps:
            super().step(epoch)
        else:
            self.after_scheduler.step(epoch)
            # Update our LR tracking
            self._last_lr = self.after_scheduler.get_last_lr()


def get_scheduler(optimizer: torch.optim.Optimizer, config: dict, steps_per_epoch: int = 1):
    """
    Build LR scheduler from config.

    Args:
        optimizer: The optimizer to schedule
        config: Full experiment config
        steps_per_epoch: Number of batches per epoch (for OneCycleLR)

    Returns:
        LR scheduler (with warmup if configured)
    """
    train_cfg = config["training"]
    sched_cfg = train_cfg.get("scheduler", {})
    warmup_cfg = train_cfg.get("warmup", {})

    sched_type = sched_cfg.get("type", "cosine_annealing")

    # Build the main scheduler
    if sched_type == "cosine_annealing":
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("T_max", train_cfg["epochs"]),
            eta_min=sched_cfg.get("eta_min", 1e-6),
        )
        step_mode = "epoch"
        print(f"📅 Scheduler: CosineAnnealing (T_max={sched_cfg.get('T_max', train_cfg['epochs'])})")

    elif sched_type == "cosine_annealing_warm_restarts":
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sched_cfg.get("T_0", 10),
            T_mult=sched_cfg.get("T_mult", 2),
            eta_min=sched_cfg.get("eta_min", 1e-5),
        )
        step_mode = "epoch"
        print(f"📅 Scheduler: CosineWarmRestarts (T_0={sched_cfg.get('T_0', 10)}, T_mult={sched_cfg.get('T_mult', 2)})")

    elif sched_type == "one_cycle":
        # OneCycleLR needs total_steps, stepped per batch
        total_steps = train_cfg["epochs"] * steps_per_epoch
        main_scheduler = OneCycleLR(
            optimizer,
            max_lr=train_cfg["optimizer"]["lr"],
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
        )
        step_mode = "step"
        print(f"📅 Scheduler: OneCycleLR (total_steps={total_steps})")

    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")

    # Wrap with warmup if configured
    if warmup_cfg.get("enabled", False) and sched_type != "one_cycle":
        warmup_epochs = warmup_cfg.get("epochs", 5)
        start_lr = warmup_cfg.get("start_lr", 1e-4)
        scheduler = LinearWarmupScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            start_lr=start_lr,
            after_scheduler=main_scheduler,
        )
        print(f"🔥 Warmup: {warmup_epochs} epochs (start_lr={start_lr})")
        return scheduler, step_mode
    else:
        return main_scheduler, step_mode
