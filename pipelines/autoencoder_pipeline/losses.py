from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossConfig, ReconLossName
from .model  import AutoencoderOutput


class CharbonnierLoss(nn.Module):

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target) ** 2 + self.eps ** 2).mean()


class CompositeLoss(nn.Module):

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config        = config
        self.recon_loss_fn = self._build_recon_loss(config.reconstruction_loss, config.charbonnier_eps)

    @staticmethod
    def _build_recon_loss(name: ReconLossName, charbonnier_eps: float) -> nn.Module:
        if name == ReconLossName.mse        : return nn.MSELoss()
        if name == ReconLossName.l1         : return nn.L1Loss()
        if name == ReconLossName.smooth_l1  : return nn.SmoothL1Loss()
        if name == ReconLossName.charbonnier: return CharbonnierLoss(eps=charbonnier_eps)
        raise ValueError(f"Unknown reconstruction loss '{name}'.")

    @staticmethod
    def variance_term(z: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
        std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
        return F.relu(target_std - std).mean()

    @staticmethod
    def covariance_term(z: torch.Tensor) -> torch.Tensor:
        b, d = z.shape
        if b < 2:
            return z.new_tensor(0.0)
        z_cent = z - z.mean(dim=0, keepdim=True)
        cov    = (z_cent.t() @ z_cent) / (b - 1)
        off    = cov - torch.diag(torch.diagonal(cov))
        return (off ** 2).sum() / d

    @staticmethod
    def contrastive_term(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        b = z_a.shape[0]
        if b < 2:
            return z_a.new_tensor(0.0)
        z_a    = F.normalize(z_a, dim=1)
        z_b    = F.normalize(z_b, dim=1)
        logits = (z_a @ z_b.t()) / temperature
        labels = torch.arange(b, device=z_a.device)
        loss_ab = F.cross_entropy(logits,     labels)
        loss_ba = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_ab + loss_ba)

    @staticmethod
    def _representation(output: AutoencoderOutput) -> torch.Tensor:
        return output.projection if output.projection is not None else output.latent

    def forward(
        self,
        output_a : AutoencoderOutput,
        target_a : torch.Tensor,
        output_b : AutoencoderOutput | None = None,
    ) -> dict[str, torch.Tensor]:
        cfg  = self.config
        zero = target_a.new_tensor(0.0)

        recon = self.recon_loss_fn(output_a.reconstruction, target_a) if cfg.use_reconstruction else zero

        rep_a = self._representation(output_a)
        rep_b = self._representation(output_b) if output_b is not None else None

        if cfg.use_variance:
            var = self.variance_term(rep_a, target_std=cfg.variance_target_std)
            if rep_b is not None:
                var = 0.5 * (var + self.variance_term(rep_b, target_std=cfg.variance_target_std))
        else:
            var = zero

        if cfg.use_covariance:
            cov = self.covariance_term(rep_a)
            if rep_b is not None:
                cov = 0.5 * (cov + self.covariance_term(rep_b))
        else:
            cov = zero

        if cfg.use_contrastive and rep_b is not None:
            con = self.contrastive_term(rep_a, rep_b, temperature=cfg.contrastive_temperature)
        else:
            con = zero

        total = (cfg.reconstruction_weight * recon
                 + cfg.variance_weight     * var
                 + cfg.covariance_weight   * cov
                 + cfg.contrastive_weight  * con)

        return {
            "total"          : total,
            "reconstruction" : recon.detach(),
            "variance"       : var.detach(),
            "covariance"     : cov.detach(),
            "contrastive"    : con.detach(),
        }
