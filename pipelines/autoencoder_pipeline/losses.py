from __future__ import annotations

import torch

from pipelines.training_pipeline.loss import LossComponents


def curve_loss(curve_hat: torch.Tensor, curve_ref: torch.Tensor, kind: str, huber_delta: float, charbonnier_eps: float) -> torch.Tensor:
    diff = curve_hat - curve_ref

    if kind == "mse":
        return LossComponents.mse_diff(diff)
    if kind == "l1":
        return LossComponents.l1_diff(diff)
    if kind == "huber":
        return LossComponents.huber_diff(diff, huber_delta)
    if kind == "charbonnier":
        return LossComponents.charbonnier_diff(diff, charbonnier_eps)
    raise ValueError(f"Unknown curve loss kind '{kind}'. Available: mse, l1, huber, charbonnier")


class ProfileAeLoss:
    def __init__(self, ae_cfg) -> None:
        self.ae_cfg = ae_cfg

    @property
    def loss_generation(self) -> int:
        return 0

    def __call__(self, curve_hat: torch.Tensor, input_curve: torch.Tensor) -> dict:
        val = curve_loss(curve_hat, input_curve, self.ae_cfg.curve_kind, self.ae_cfg.huber_delta, self.ae_cfg.charbonnier_eps)
        return {
            "total_loss": val,
            "components": {"curve_recon": val},
            "weighted":   {"curve_recon": val},
            "monitor":    {},
        }
