from __future__ import annotations

from pathlib import Path

import numpy as np

from configuration.dataset  import DatasetConfig
from configuration.training import BackboneTrainerConfig, OverfitCheckConfig
from models                 import BACKBONE_IMAGE_SIZE_MODELS, get_backbone
from pipelines.backbone.dataset.pipeline     import DatasetPipeline
from pipelines.backbone.training.trainer     import Trainer
from pipelines.shared.config.run_metadata    import TrainingRunMetadata
from pipelines.shared.model.model_builder    import ModelBuilder
from pipelines.shared.training.overfit_check import OverfitCheck
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
        overfit_check  : OverfitCheckConfig | None = None,
    ) -> None:

        patch_height, patch_width = dataset_config.patch.size

        self.trainer_config = trainer_config
        self.dataset_config = dataset_config
        self.backbone_name  = backbone_name
        self.model_config   = model_config
        self.image_size     = patch_height
        self.seed           = seed
        self.overfit_check  = overfit_check if overfit_check is not None else OverfitCheckConfig()

        Reproducibility.seed_everything(self.seed)

        self.run_metadata = TrainingRunMetadata(
            trainer_config = trainer_config,
            model_name     = backbone_name,
            base_logdir    = Path(trainer_config.io.logdir),
            run_name       = run_name,
        )
        self.logger = self.run_metadata.logger

        gaussian_cfg                    = trainer_config.gaussian
        self.dataset_config.n_gaussians = gaussian_cfg.n_default_gaussians

        self.dataset_pipeline = DatasetPipeline(
            config                 = dataset_config,
            training_run_directory = self.run_metadata.run_directory,
            logger                 = self.logger,
            seed                   = self.seed,
            height_axis_convention = trainer_config.geometry.height_axis_convention,
            build_geometry_field   = self.physics_geometry_active(trainer_config),
        )

        self.dataset_config.x_axis = GaussianAxis.build(gaussian_cfg.x_min, gaussian_cfg.x_max, self.dataset_pipeline.layout.profile_length)

    @staticmethod
    def physics_geometry_active(trainer_config) -> bool:
        curriculum = trainer_config.curriculum
        loss_cfgs  = [curriculum.warmup] + ([curriculum.complete] if curriculum.enabled else [])
        flags      = ("use_coherence_resyn", "use_covariance_match", "use_capon_cycle")

        return any(getattr(cfg, flag) for cfg in loss_cfgs for flag in flags)

    def _run_overfit_check(self, train_dataset, in_channels: int, out_channels: int, x_axis) -> None:
        check = OverfitCheck(self.overfit_check, self.run_metadata.run_directory, self.logger)
        if not check.enabled:
            return

        gate_trainer_config                    = check.sanitized_trainer_config(self.trainer_config)
        gate_trainer_config.curriculum.enabled = False

        for loss_cfg in (gate_trainer_config.curriculum.warmup, gate_trainer_config.curriculum.complete):
            loss_cfg.use_active_normalization = False

        check.record("curriculum.enabled", False)
        check.record("curriculum.warmup.use_active_normalization",   False)
        check.record("curriculum.complete.use_active_normalization", False)

        base_config       = self.model_config if self.model_config is not None else ModelBuilder.config_from_registry(self.backbone_name, {})
        gate_model_config = check.sanitized_model_config(base_config)

        gate_model, gate_model_cfg = get_backbone(self.backbone_name, config=gate_model_config, **self._model_overrides(in_channels, out_channels))

        gate_trainer = Trainer(
            model      = gate_model,
            model_cfg  = gate_model_cfg,
            x_axis     = x_axis,
            config     = gate_trainer_config,
            run_dir    = check.work_directory,
            logger     = self.logger,
            norm_stats = train_dataset.normalizer,
            emit_docs  = False,
        )

        check.run(gate_trainer, train_dataset)

    def _build_model(self, in_channels: int, out_channels: int):
        model, model_cfg = get_backbone(self.backbone_name, config=self.model_config, **self._model_overrides(in_channels, out_channels))

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.section("[Model Built]")
        self.logger.subsection(f"Architecture : {self.backbone_name}")
        self.logger.subsection(f"In Channels  : {in_channels}")
        self.logger.subsection(f"Out Channels : {out_channels}")
        self.logger.subsection(f"Parameters   : {n_params:,}")
        return model, model_cfg

    def _model_overrides(self, in_channels: int, out_channels: int) -> dict:
        overrides = {"in_channels": in_channels, "out_channels": out_channels}
        if self.backbone_name in BACKBONE_IMAGE_SIZE_MODELS:
            overrides["image_size"] = self.image_size

        return overrides

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

        gaussian_cfg  = self.trainer_config.gaussian
        x_axis_length = int(np.asarray(self.dataset_config.x_axis).size)

        train_loader, val_loader, test_loader, datasets = self.dataset_pipeline.run()

        train_dataset = datasets["train"]
        in_channels   = train_dataset.input_channels
        n_gaussians   = gaussian_cfg.n_default_gaussians
        out_channels  = GaussianHead.total_channels(gaussian_cfg.params_per_gaussian, n_gaussians)
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
            seed             = self.seed,
        )

        try:
            self._run_overfit_check(train_dataset, in_channels, out_channels, x_axis)

            trainer = self._make_trainer(model, model_cfg, x_axis, train_dataset.normalizer)

            trainer.maybe_run_loss_probe(train_loader, probe_config)
            results = trainer.train(train_loader, val_loader, test_loader)
            self.run_metadata.save_test_metrics(trainer.test_metrics)
        finally:
            self.run_metadata.close()

        return results
