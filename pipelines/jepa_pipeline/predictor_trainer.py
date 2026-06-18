from __future__ import annotations

import torch.nn as nn

from tools.training       import BaseTrainer
from pipelines.jepa_pipeline.coupling    import StageAMode, TargetProvider
from pipelines.jepa_pipeline.loss        import JepaLoss


class JepaModule(nn.Module):
    def __init__(self, backbone: nn.Module, autoencoder: nn.Module) -> None:
        super().__init__()
        self.backbone    = backbone
        self.autoencoder = autoencoder

    def forward(self, images):
        return self.backbone(images)


class JepaPredictorTrainer(BaseTrainer):
    stage_name    = "Stage-B"
    section_title = "[Stage-B JEPA Predictor Training]"

    def __init__(self, model: JepaModule, backbone_cfg, x_axis, config, run_dir, logger, norm_stats):
        self.backbone_cfg = backbone_cfg
        self.gaussian_cfg = config.gaussian
        self.norm_stats   = norm_stats

        self.stage_a_mode = StageAMode(config.stage_a_mode)
        self.stage_a_mode.apply(model.autoencoder)

        super().__init__(model, config, run_dir, logger, x_axis)

    def _build_param_groups(self):
        param_groups  = self.backbone_cfg.get_param_groups(self.model.backbone)
        param_groups += self.stage_a_mode.param_groups(self.model.autoencoder, self.config.ae_finetune_lr, self.config.ae_finetune_wd)
        return param_groups

    def _build_criterion(self):
        target_provider = TargetProvider(self.config.target_provider, self.model.autoencoder.encoder, self.config.ema_decay).to(self.device)
        return JepaLoss(self.model.autoencoder, target_provider, self.config.embedding_loss, self.x_axis, self.norm_stats, self.gaussian_cfg.params_per_gaussian)

    def _set_train_mode(self):
        self.model.backbone.train()
        if self.stage_a_mode.trainable:
            self.model.autoencoder.train()

    def _on_optimizer_step(self):
        if self.stage_a_mode.trainable:
            self.criterion.target_provider.update(self.model.autoencoder.encoder)

    def _compute_loss(self, batch):
        images    = batch[0].to(self.device)
        gt_params = batch[1].to(self.device)
        z_hat     = self.model(images)
       
        return self.criterion(z_hat, gt_params)
