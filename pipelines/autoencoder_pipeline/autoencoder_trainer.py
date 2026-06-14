from __future__ import annotations

from tools.training         import BaseTrainer
from pipelines.autoencoder_pipeline.loss import ProfileAeLoss


class ProfileAeTrainer(BaseTrainer):
    stage_name    = "Stage-A"
    section_title = "[Stage-A Autoencoder Training]"

    def __init__(self, model, model_cfg, x_axis, config, run_dir, logger):
        self.model_cfg = model_cfg
        super().__init__(model, config, run_dir, logger, x_axis)

    def _build_param_groups(self):
        return self.model_cfg.get_param_groups(self.model)

    def _build_criterion(self):
        return ProfileAeLoss(self.config.ae_loss)

    def _compute_loss(self, batch):
        curve        = batch.to(self.device).unsqueeze(-1).unsqueeze(-1)
        curve_hat, _ = self.model.reconstruct(curve)
        target       = self.model.normalize_curve(curve)
        return self.criterion(curve_hat, target)
