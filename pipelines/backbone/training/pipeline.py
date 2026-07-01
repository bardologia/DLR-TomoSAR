from __future__ import annotations

from pathlib import Path

import numpy as np

from configuration.dataset  import DatasetConfig
from configuration.training import BackboneTrainerConfig
from models                 import BACKBONE_IMAGE_SIZE_MODELS
from pipelines.backbone.dataset.pipeline  import DatasetPipeline
from pipelines.backbone.training.trainer  import Trainer
from pipelines.shared.config.run_metadata import TrainingRunMetadata
from tools.data.gaussians          import GaussianAxis, GaussianHead
from tools.runtime.reproducibility import Reproducibility


class TrainingPipeline:
    def __init__(
        self,
        trainer_config : BackboneTrainerConfig,
        dataset_config : DatasetConfig,
        backbone_name  : str,
        model_config   = None,
        seed           : int = 0,
        run_name       : str | None = None,
    ) -> None:

        patch_height, patch_width = dataset_config.patch.size

        self.trainer_config = trainer_config
        self.dataset_config = dataset_config
        self.backbone_name     = backbone_name
        self.model_config   = model_config
        self.image_size     = patch_height
        self.seed           = seed

        Reproducibility.seed_everything(self.seed)

        self.run_metadata = TrainingRunMetadata(
            trainer_config = trainer_config,
            model_name     = backbone_name,
            base_logdir    = Path(trainer_config.io.logdir),
            run_name       = run_name,
        )
        self.logger = self.run_metadata.logger

        self.dataset_pipeline = DatasetPipeline(
            config                 = dataset_config,
            training_run_directory = self.run_metadata.run_directory,
            logger                 = self.logger,
            seed                   = self.seed,
            height_axis_convention = trainer_config.geometry.height_axis_convention,
            build_geometry_field   = self.physics_geometry_active(trainer_config),
        )

    @staticmethod
    def physics_geometry_active(trainer_config) -> bool:
        curriculum = trainer_config.curriculum
        loss_cfgs  = [curriculum.warmup] + ([curriculum.complete] if curriculum.enabled else [])
        flags      = ("use_coherence_resyn", "use_covariance_match", "use_capon_cycle")

        return any(getattr(cfg, flag, False) for cfg in loss_cfgs for flag in flags)

    def _build_model(self, in_channels: int, out_channels: int):
        from models import get_backbone

        overrides = {"in_channels": in_channels, "out_channels": out_channels}
        if self.backbone_name in BACKBONE_IMAGE_SIZE_MODELS:
            overrides["image_size"] = self.image_size

        model, model_cfg = get_backbone(self.backbone_name, config=self.model_config, **overrides)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.section("[Model Built]")
        self.logger.subsection(f"Architecture : {self.backbone_name}")
        self.logger.subsection(f"In Channels  : {in_channels}")
        self.logger.subsection(f"Out Channels : {out_channels}")
        self.logger.subsection(f"Parameters   : {n_params:,}")
        return model, model_cfg

    def _make_trainer(self, model, model_cfg, x_axis, norm_stats):
        return Trainer(
            model      = model,
            model_cfg  = model_cfg,
            x_axis     = x_axis,
            config     = self.trainer_config,
            run_dir    = self.run_metadata.run_directory,
            logger     = self.logger,
            norm_stats = norm_stats,
        )

    def run(self, probe_config=None):
        self.logger.section("[PyTorch Training Pipeline Execution]")

        gaussian_cfg                    = self.trainer_config.gaussian
        self.dataset_config.n_gaussians = gaussian_cfg.n_default_gaussians

        x_axis_length = self.dataset_pipeline.layout.profile_length

        self.dataset_config.x_axis = GaussianAxis.build(gaussian_cfg.x_min, gaussian_cfg.x_max, x_axis_length)

        train_loader, val_loader, test_loader, datasets = self.dataset_pipeline.run()

        train_dataset = datasets["train"]
        in_channels   = train_dataset.input_channels
        n_gaussians   = gaussian_cfg.n_default_gaussians
        out_channels  = GaussianHead.total_channels(gaussian_cfg.params_per_gaussian, n_gaussians, gaussian_cfg.predict_presence)
        x_axis        = np.asarray(self.dataset_config.x_axis, dtype=np.float32)

        model, model_cfg = self._build_model(in_channels=in_channels, out_channels=out_channels)

        self.run_metadata.save_trainer_config()
        self.run_metadata.save_model_config(model_cfg, self.backbone_name)

        self.run_metadata.save_run_summary(
            model_name       = self.backbone_name,
            in_channels      = in_channels,
            out_channels     = out_channels,
            x_axis_length    = x_axis_length,
            n_gaussians      = n_gaussians,
            predict_presence = gaussian_cfg.predict_presence,
        )

        trainer = self._make_trainer(model, model_cfg, x_axis, train_dataset.normalizer)

        try:
            trainer.maybe_run_loss_probe(train_loader, probe_config)
            results = trainer.train(train_loader, val_loader, test_loader)
        finally:
            self.run_metadata.close()

        return results
