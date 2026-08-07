from __future__ import annotations

import torch

from pipelines.autoencoder_common.trainer        import AutoencoderTrainer
from pipelines.profile_autoencoder.training.loss import Loss


class Trainer(AutoencoderTrainer):
    stage_name    = "Profile Autoencoder"
    section_title = "[Profile Autoencoder Training]"

    def _build_criterion(self):
        cfg = self.config.ae_loss

        self.logger.section("[Loss Function]")
        self.logger.kv_table({
            "Curve kind"      : cfg.curve_kind,
            "Huber delta"     : f"{cfg.huber_delta:g} (used only when kind=huber)",
            "Charbonnier eps" : f"{cfg.charbonnier_eps:g} (used only when kind=charbonnier)",
            "Sobolev"         : f"weight={cfg.weight_sobolev:g}" if cfg.use_sobolev else "off",
            "Latent noise"    : f"std={cfg.latent_noise_std:g} weight={cfg.weight_latent_noise:g}" if cfg.use_latent_noise else "off",
        })

        return Loss(cfg)

    def _compute_loss(self, batch):
        noisy, clean = batch
        noisy        = noisy.to(self.device).unsqueeze(-1).unsqueeze(-1)
        clean        = clean.to(self.device).unsqueeze(-1).unsqueeze(-1)

        curve_hat, z = self.model.reconstruct(noisy)
        loss_dict    = self.criterion(curve_hat, clean)

        cfg = self.config.ae_loss
        if cfg.use_latent_noise:
            jittered     = self.model.decode(z + torch.randn_like(z) * cfg.latent_noise_std)
            value, terms = self.criterion.evaluate(jittered - clean)

            for name, term in terms.items():
                loss_dict["components"][f"latent_{name}"] = term

            loss_dict["total_loss"] = loss_dict["total_loss"] + cfg.weight_latent_noise * value

        return loss_dict
