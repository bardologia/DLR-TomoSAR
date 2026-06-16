from __future__ import annotations

from tools.training                              import BaseTrainer
from pipelines.image_autoencoder.training.loss   import Loss


class Trainer(BaseTrainer):
    stage_name    = "Image Autoencoder"
    section_title = "[Image Autoencoder Training]"

    def __init__(self, model, model_cfg, x_axis, config, run_dir, logger):
        self.model_cfg = model_cfg
        super().__init__(model, config, run_dir, logger, x_axis)

    def _build_param_groups(self):
        return self.model_cfg.get_param_groups(self.model)

    def _build_criterion(self):
        return Loss(self.config.ae_loss)

    def _compute_loss(self, batch):
        image        = batch[0].to(self.device)
        image_hat, _ = self.model.reconstruct(image)
        return self.criterion(image_hat, image)
