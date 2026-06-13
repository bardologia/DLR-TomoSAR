from __future__ import annotations

import copy

import torch
import torch.nn as nn


class StageAMode:
    VALID = ("frozen", "finetune", "joint")

    def __init__(self, kind: str) -> None:
        if kind not in self.VALID:
            raise ValueError(f"Unknown stage_a_mode '{kind}'. Available: {self.VALID}")
        self.kind = kind

    @property
    def trainable(self) -> bool:
        return self.kind in ("finetune", "joint")

    def apply(self, autoencoder: nn.Module) -> None:
        autoencoder.requires_grad_(self.trainable)
        if not self.trainable:
            autoencoder.eval()

    def param_groups(self, autoencoder: nn.Module, lr: float, wd: float) -> list[dict]:
        if not self.trainable:
            return []

        params = [p for p in autoencoder.parameters() if p.requires_grad]
        if not params:
            return []
        return [{"params": params, "lr": lr, "weight_decay": wd, "name": "stage_a"}]


class TargetProvider:
    VALID = ("stopgrad", "live", "ema")

    def __init__(self, kind: str, encoder: nn.Module, decay: float = 0.996) -> None:
        if kind not in self.VALID:
            raise ValueError(f"Unknown target_provider '{kind}'. Available: {self.VALID}")
        self.kind  = kind
        self.decay = float(decay)

        self._ema = None
        if kind == "ema":
            self._ema = copy.deepcopy(encoder)
            self._ema.requires_grad_(False)
            self._ema.eval()

    def to(self, device) -> "TargetProvider":
        if self._ema is not None:
            self._ema.to(device)
        return self

    def target(self, encoder: nn.Module, curve: torch.Tensor) -> torch.Tensor:
        if self.kind == "live":
            return encoder(curve)

        if self.kind == "ema":
            with torch.no_grad():
                return self._ema(curve)

        with torch.no_grad():
            return encoder(curve)

    @torch.no_grad()
    def update(self, encoder: nn.Module) -> None:
        if self._ema is None:
            return

        for ema_p, online_p in zip(self._ema.parameters(), encoder.parameters()):
            ema_p.mul_(self.decay).add_(online_p.detach(), alpha=1.0 - self.decay)
        for ema_b, online_b in zip(self._ema.buffers(), encoder.buffers()):
            ema_b.copy_(online_b)
