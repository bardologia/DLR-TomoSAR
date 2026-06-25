from __future__ import annotations

from pipelines.autoencoder_common.trainer      import AutoencoderTrainer
from pipelines.image_autoencoder.training.loss import Loss


class Trainer(AutoencoderTrainer):
    stage_name    = "Image Autoencoder"
    section_title = "[Image Autoencoder Training]"

    def _build_criterion(self):
        return Loss(self.config.ae_loss)

    def _compute_loss(self, batch):
        image        = batch[0].to(self.device)
        image_hat, _ = self.model.reconstruct(image)
        return self.criterion(image_hat, image)
