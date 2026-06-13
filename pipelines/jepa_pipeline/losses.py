from __future__ import annotations

import torch
import torch.nn.functional as F

from pipelines.training_pipeline.loss import Loss, LossComponents


class RunningStandardizer:
    def __init__(self, dim: int, momentum: float = 0.01) -> None:
        self.momentum     = float(momentum)
        self.mean         = torch.zeros(dim)
        self.std          = torch.ones(dim)
        self._initialized = False

    def to(self, device) -> "RunningStandardizer":
        self.mean = self.mean.to(device)
        self.std  = self.std.to(device)
        return self

    @torch.no_grad()
    def update(self, z: torch.Tensor) -> None:
        flat = z.transpose(0, 1).reshape(z.shape[1], -1)
        m    = flat.mean(dim=1)
        s    = flat.std(dim=1).clamp(min=1e-6)

        if not self._initialized:
            self.mean = m.clone()
            self.std  = s.clone()
            self._initialized = True
        else:
            self.mean.mul_(1 - self.momentum).add_(m, alpha=self.momentum)
            self.std.mul_(1 - self.momentum).add_(s, alpha=self.momentum)

    def apply(self, z: torch.Tensor) -> torch.Tensor:
        shape = (1, -1, 1, 1)
        return (z - self.mean.reshape(shape)) / self.std.reshape(shape)


class ProfileAeLoss:
    def __init__(self, inner_loss: Loss, ae_cfg) -> None:
        self.inner  = inner_loss
        self.ae_cfg = ae_cfg

    @staticmethod
    def curve_term(curve_hat: torch.Tensor, curve_ref: torch.Tensor, cfg) -> torch.Tensor:
        diff = curve_hat - curve_ref
        kind = cfg.ae_curve_kind

        if kind == "mse":
            return LossComponents.mse_diff(diff)
        if kind == "l1":
            return LossComponents.l1_diff(diff)
        if kind == "huber":
            return LossComponents.huber_diff(diff, cfg.ae_huber_delta)
        if kind == "charbonnier":
            return LossComponents.charbonnier_diff(diff, cfg.ae_charbonnier_eps)
        raise ValueError(f"Unknown ae_curve_kind '{kind}'. Available: mse, l1, huber, charbonnier")

    def set_curriculum(self, complete_cfg) -> None:
        self.inner.set_curriculum(complete_cfg)

    @property
    def loss_generation(self) -> int:
        return self.inner.loss_generation

    def __call__(self, params_hat_norm, curve_hat, input_curve, gt_params_norm) -> dict:
        inner = self.inner(params_hat_norm, gt_params_norm)

        total      = inner["total_loss"]
        components = dict(inner["components"])
        weighted   = dict(inner["weighted"])
        monitor    = dict(inner["monitor"])

        if self.ae_cfg.use_ae_curve:
            ae_val = self.curve_term(curve_hat, input_curve, self.ae_cfg)
            w      = self.ae_cfg.weight_ae_curve
            components["ae_curve"] = ae_val
            weighted["ae_curve"]   = w * ae_val
            total = total + w * ae_val
        elif self.inner.log_all_losses:
            with torch.no_grad():
                monitor["ae_curve_denorm"] = self.curve_term(curve_hat, input_curve, self.ae_cfg)

        return {"total_loss": total, "components": components, "weighted": weighted, "monitor": monitor}


class JepaLoss:
    def __init__(self, autoencoder, inner_loss: Loss, target_provider, embedding_cfg, norm_stats) -> None:
        self.autoencoder     = autoencoder
        self.inner           = inner_loss
        self.target_provider = target_provider
        self.emb_cfg         = embedding_cfg
        self.norm_stats      = norm_stats
        self.standardizer    = RunningStandardizer(autoencoder.config.embedding_dim, embedding_cfg.standardize_momentum)
        self._std_on_device  = False

    def set_curriculum(self, complete_cfg) -> None:
        self.inner.set_curriculum(complete_cfg)

    @property
    def loss_generation(self) -> int:
        return self.inner.loss_generation

    def _embedding_terms(self, z_hat, z_star) -> tuple[torch.Tensor, dict, dict]:
        cfg        = self.emb_cfg
        components = {}
        weighted   = {}
        total      = torch.zeros((), dtype=z_hat.dtype, device=z_hat.device)

        if cfg.standardize_target:
            if not self._std_on_device:
                self.standardizer.to(z_hat.device)
                self._std_on_device = True
            self.standardizer.update(z_star.detach())
            z_hat_s  = self.standardizer.apply(z_hat)
            z_star_s = self.standardizer.apply(z_star)
        else:
            z_hat_s, z_star_s = z_hat, z_star

        if cfg.use_embedding_mse:
            val = LossComponents.mse_diff(z_hat_s - z_star_s)
            components["embedding_mse"] = val
            weighted["embedding_mse"]   = cfg.weight_embedding_mse * val
            total = total + cfg.weight_embedding_mse * val

        if cfg.use_embedding_cosine:
            val = LossComponents.cosine(z_hat_s, z_star_s, axis=1)
            components["embedding_cosine"] = val
            weighted["embedding_cosine"]   = cfg.weight_embedding_cosine * val
            total = total + cfg.weight_embedding_cosine * val

        if cfg.use_embedding_smoothl1:
            val = F.smooth_l1_loss(z_hat_s, z_star_s, beta=cfg.smoothl1_beta)
            components["embedding_smoothl1"] = val
            weighted["embedding_smoothl1"]   = cfg.weight_embedding_smoothl1 * val
            total = total + cfg.weight_embedding_smoothl1 * val

        return total, components, weighted

    def decode_params(self, z_hat) -> torch.Tensor:
        return self.autoencoder.heads(z_hat)

    def __call__(self, z_hat, gt_params_norm) -> dict:
        with torch.no_grad():
            gt_phys  = self.norm_stats.denormalize_output(gt_params_norm.float())
            gt_curve = self.inner.reconstruct(gt_phys).to(z_hat.dtype)

        z_star = self.target_provider.target(self.autoencoder.encoder, gt_curve)

        emb_total, emb_components, emb_weighted = self._embedding_terms(z_hat, z_star)

        params_hat = self.decode_params(z_hat)
        inner      = self.inner(params_hat, gt_params_norm)

        total      = emb_total + inner["total_loss"]
        components = {**emb_components, **inner["components"]}
        weighted   = {**emb_weighted,   **inner["weighted"]}
        monitor    = dict(inner["monitor"])

        return {"total_loss": total, "components": components, "weighted": weighted, "monitor": monitor}
