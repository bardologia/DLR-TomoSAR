from __future__ import annotations

import torch

from tools.loss.curve_loss import CurveLoss


class ProfileAeLoss:
    def __init__(self, ae_cfg) -> None:
        self.ae_cfg = ae_cfg

    @property
    def loss_generation(self) -> int:
        return 0

    def __call__(self, curve_hat: torch.Tensor, input_curve: torch.Tensor) -> dict:
        cfg  = self.ae_cfg
        diff = curve_hat - input_curve

        if   cfg.curve_kind == "mse":         val = CurveLoss.mse_diff(diff)
        elif cfg.curve_kind == "l1":          val = CurveLoss.l1_diff(diff)
        elif cfg.curve_kind == "huber":       val = CurveLoss.huber_diff(diff, cfg.huber_delta)
        elif cfg.curve_kind == "charbonnier": val = CurveLoss.charbonnier_diff(diff, cfg.charbonnier_eps)
        else:                                 raise ValueError(f"Unknown curve loss kind '{cfg.curve_kind}'. Available: mse, l1, huber, charbonnier")

        return {
            "total_loss": val,
            "components": {"curve_recon": val},
            "weighted":   {"curve_recon": val},
            "monitor":    {},
        }
