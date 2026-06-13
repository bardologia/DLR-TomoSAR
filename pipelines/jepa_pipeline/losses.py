from __future__ import annotations

import torch
import torch.nn.functional as F

from pipelines.training_pipeline.loss import LossComponents
from tools.gaussians                  import GaussianCurve


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


class JepaLoss:
    def __init__(self, autoencoder, target_provider, embedding_cfg, x_axis, norm_stats, params_per_gaussian: int) -> None:
        self.autoencoder     = autoencoder
        self.target_provider = target_provider
        self.emb_cfg         = embedding_cfg
        self.x_axis          = x_axis
        self.norm_stats      = norm_stats
        self.ppg             = params_per_gaussian

    @property
    def loss_generation(self) -> int:
        return 0

    def _embedding_terms(self, z_hat, z_star) -> tuple[torch.Tensor, dict, dict]:
        cfg        = self.emb_cfg
        components = {}
        weighted   = {}
        total      = torch.zeros((), dtype=z_hat.dtype, device=z_hat.device)

        if cfg.use_embedding_mse:
            val = LossComponents.mse_diff(z_hat - z_star)
            components["embedding_mse"] = val
            weighted["embedding_mse"]   = cfg.weight_embedding_mse * val
            total = total + cfg.weight_embedding_mse * val

        if cfg.use_embedding_cosine:
            val = LossComponents.cosine(z_hat, z_star, axis=1)
            components["embedding_cosine"] = val
            weighted["embedding_cosine"]   = cfg.weight_embedding_cosine * val
            total = total + cfg.weight_embedding_cosine * val

        if cfg.use_embedding_smoothl1:
            val = F.smooth_l1_loss(z_hat, z_star, beta=cfg.smoothl1_beta)
            components["embedding_smoothl1"] = val
            weighted["embedding_smoothl1"]   = cfg.weight_embedding_smoothl1 * val
            total = total + cfg.weight_embedding_smoothl1 * val

        return total, components, weighted

    def __call__(self, z_hat, gt_params_norm) -> dict:
        with torch.no_grad():
            gt_phys    = self.norm_stats.denormalize_output(gt_params_norm.float())
            gt_curve   = GaussianCurve.reconstruct(gt_phys, self.x_axis, self.ppg).to(z_hat.dtype)
            gt_curve_n = self.autoencoder.normalize_curve(gt_curve)

        z_hat_n    = self.autoencoder.normalize_embedding(z_hat)
        z_star_raw = self.target_provider.target(self.autoencoder.encoder, gt_curve_n)
        z_star_n   = self.autoencoder.normalize_embedding(z_star_raw)

        total, components, weighted = self._embedding_terms(z_hat_n, z_star_n)

        if self.emb_cfg.use_curve_recon:
            curve_hat = self.autoencoder.decode(z_hat_n)
            c_val     = curve_loss(curve_hat, gt_curve_n, self.emb_cfg.curve_kind, self.emb_cfg.huber_delta, self.emb_cfg.charbonnier_eps)
            w         = self.emb_cfg.weight_curve_recon
            components["curve_recon"] = c_val
            weighted["curve_recon"]   = w * c_val
            total = total + w * c_val

        return {"total_loss": total, "components": components, "weighted": weighted, "monitor": {}}
