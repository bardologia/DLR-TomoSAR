"""Training loop, optimizer, and scheduler for the dense MoE model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import MoEConfig, TrainingMode
from .losses import MoELoss
from .moe_model import DenseMoE


@dataclass
class TrainerConfig:
    epochs             : int            = 100
    lr_experts         : float          = 1e-4
    lr_gating          : float          = 1e-3
    weight_decay       : float          = 1e-5
    optimizer          : str            = "adamw"
    scheduler          : str            = "cosine"
    scheduler_kwargs   : dict[str, Any] = field(default_factory=dict)
    checkpoint_dir     : str            = "checkpoints/moe"
    save_every         : int            = 10
    use_per_expert_loss : bool          = True
    device             : str            = "cuda"


# ── optimizer / scheduler factories ──────────────────────────────────────────

def build_optimizer(model: DenseMoE, config: TrainerConfig) -> torch.optim.Optimizer:
    """Separate LR groups for experts and gating; only trainable params."""
    expert_params = [p for p in model.experts.parameters() if p.requires_grad]
    gating_params = [p for p in model.gating.parameters()  if p.requires_grad]

    groups: list[dict[str, Any]] = []
    if expert_params:
        groups.append({"params": expert_params, "lr": config.lr_experts})
    if gating_params:
        groups.append({"params": gating_params, "lr": config.lr_gating})
    if not groups:
        raise RuntimeError("No trainable parameters found.")

    name = config.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(groups, weight_decay=config.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, weight_decay=config.weight_decay)
    raise ValueError(f"Unknown optimizer '{name}'. Available: adam, adamw, sgd")


def build_scheduler(optimizer: torch.optim.Optimizer, config: TrainerConfig):
    name = config.scheduler.lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, **config.scheduler_kwargs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **config.scheduler_kwargs)
    raise ValueError(f"Unknown scheduler '{name}'. Available: cosine, step, none")


# ── trainer ──────────────────────────────────────────────────────────────────

class MoETrainer:
    def __init__(
        self,
        model          : DenseMoE,
        moe_config     : MoEConfig,
        trainer_config : TrainerConfig,
        train_loader   : DataLoader,
        val_loader     : DataLoader | None = None,
    ):
        self.device       = torch.device(trainer_config.device)
        self.model        = model.to(self.device)
        self.moe_config   = moe_config
        self.cfg          = trainer_config
        self.train_loader = train_loader
        self.val_loader   = val_loader

        self.criterion = MoELoss(
            moe_config.loss,
            max_out_channels    = moe_config.max_out_channels,
            use_per_expert_loss = trainer_config.use_per_expert_loss,
        )
        self.optimizer = build_optimizer(model, trainer_config)
        self.scheduler = build_scheduler(self.optimizer, trainer_config)

        self.ckpt_dir = Path(trainer_config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.history: list[dict[str, float]] = []

    # ── training loop ────────────────────────────────────────────────────

    def fit(self) -> list[dict[str, float]]:
        for epoch in range(1, self.cfg.epochs + 1):
            train_metrics = self._train_one_epoch()
            val_metrics   = self._validate() if self.val_loader is not None else {}

            record = {"epoch": epoch, **train_metrics, **val_metrics}
            self.history.append(record)

            if self.scheduler is not None:
                self.scheduler.step()
            if epoch % self.cfg.save_every == 0:
                self._save_checkpoint(epoch)

            self._log(record)

        self._save_checkpoint("final")
        return self.history

    def _train_one_epoch(self) -> dict[str, float]:
        self.model.train()
        accum = _MetricAccumulator()

        for inputs, targets in self.train_loader:
            inputs  = inputs.to(self.device)
            targets = targets.to(self.device)

            moe_out = self.model(inputs)
            losses  = self.criterion(moe_out, targets)

            self.optimizer.zero_grad()
            losses["total"].backward()
            self.optimizer.step()

            accum.update(losses, inputs.shape[0])

        return accum.compute(prefix="train/")

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        self.model.eval()
        accum = _MetricAccumulator()

        for inputs, targets in self.val_loader:
            inputs  = inputs.to(self.device)
            targets = targets.to(self.device)

            moe_out = self.model(inputs)
            losses  = self.criterion(moe_out, targets)
            accum.update(losses, inputs.shape[0])

        return accum.compute(prefix="val/")

    # ── checkpointing ────────────────────────────────────────────────────

    def _save_checkpoint(self, tag: int | str) -> None:
        path = self.ckpt_dir / f"moe_epoch_{tag}.pt"
        torch.save({
            "model_state_dict"     : self.model.state_dict(),
            "optimizer_state_dict" : self.optimizer.state_dict(),
            "moe_config"           : self.moe_config,
            "trainer_config"       : self.cfg,
            "history"              : self.history,
        }, path)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "history" in ckpt:
            self.history = ckpt["history"]

    # ── logging ──────────────────────────────────────────────────────────

    @staticmethod
    def _log(record: dict[str, float]) -> None:
        parts = [f"Epoch {record['epoch']:>4d}"]
        for k, v in record.items():
            if k != "epoch":
                parts.append(f"{k}={v:.6f}")
        print(" | ".join(parts))


class _MetricAccumulator:
    """Running average of loss components across mini-batches."""

    def __init__(self) -> None:
        self._sums  : dict[str, float] = {}
        self._count : int              = 0

    def update(self, losses: dict[str, torch.Tensor], batch_size: int) -> None:
        for k, v in losses.items():
            self._sums[k] = self._sums.get(k, 0.0) + v.item() * batch_size
        self._count += batch_size

    def compute(self, prefix: str = "") -> dict[str, float]:
        return {f"{prefix}{k}": v / self._count for k, v in self._sums.items()}
