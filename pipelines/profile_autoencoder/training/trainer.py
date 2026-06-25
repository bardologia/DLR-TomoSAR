from __future__ import annotations

from pipelines.autoencoder_common.trainer        import AutoencoderTrainer
from pipelines.profile_autoencoder.training.loss import Loss


class Trainer(AutoencoderTrainer):
    stage_name    = "Profile Autoencoder"
    section_title = "[Profile Autoencoder Training]"

    def _build_criterion(self):
        return Loss(self.config.ae_loss)

    def _compute_loss(self, batch):
        curve        = batch.to(self.device).unsqueeze(-1).unsqueeze(-1)
        curve_hat, _ = self.model.reconstruct(curve)
        return self.criterion(curve_hat, curve)
