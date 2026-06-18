from __future__ import annotations

import torch

from tools.loss.curve_loss import CurveLoss


class Loss:
    def __init__(self, ae_loss_cfg) -> None:
        self.cfg             = ae_loss_cfg
        self.loss_generation = 0

    def __call__(self, image_hat: torch.Tensor, image: torch.Tensor) -> dict:
        diff = image_hat - image

        if   self.cfg.recon_kind == "mse":         val = CurveLoss.mse_diff(diff)
        elif self.cfg.recon_kind == "l1":          val = CurveLoss.l1_diff(diff)
        elif self.cfg.recon_kind == "huber":       val = CurveLoss.huber_diff(diff, self.cfg.huber_delta)
        elif self.cfg.recon_kind == "charbonnier": val = CurveLoss.charbonnier_diff(diff, self.cfg.charbonnier_eps)
        else:                                       raise ValueError(f"Unknown recon_kind '{self.cfg.recon_kind}'. Available: mse, l1, huber, charbonnier")

        return {
            "total_loss" : val,
            "components"  : {"image_recon": val},
            "weighted"    : {"image_recon": val},
            "monitor"     : {},
            "occupancy"   : {},
        }
