from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from configuration.jepa_config              import JepaTrainerConfig
from models                                 import get_model
from models.profile_autoencoder             import ProfileAutoencoder
from pipelines.benchmark_pipeline.config_factory import ConfigFactory
from pipelines.dataset_pipeline.pipeline    import DatasetPipeline
from pipelines.training_pipeline.pipeline   import TrainingRunMetadata
from pipelines.autoencoder_pipeline.pipeline import AutoencoderPipelineSupport
from pipelines.jepa_pipeline.predictor_trainer   import JepaModule, JepaPredictorTrainer
from tools.reproducibility                  import Reproducibility

_IMAGE_SIZE_MODELS = {"swin_unet", "transunet", "unetr"}


class JepaPipeline:
    def __init__(self, entry_config) -> None:
        self.entry   = entry_config
        self.factory = ConfigFactory(entry_config)
        Reproducibility.seed_everything(entry_config.seed)

        base = self.factory.training_trainer_config(logdir=entry_config.logdir)

        if entry_config.stage_a_run is not None:
            self.autoencoder_cfg = AutoencoderPipelineSupport.load_autoencoder_config(Path(entry_config.stage_a_run) / "meta")
        else:
            self.autoencoder_cfg = entry_config.autoencoder

        self.trainer_config = JepaTrainerConfig(
            gaussian           = base.gaussian,
            autoencoder        = self.autoencoder_cfg,
            embedding_loss     = entry_config.embedding_loss,
            stage_a_mode       = entry_config.stage_a_mode,
            target_provider    = entry_config.target_provider,
            stage_a_checkpoint = (str(Path(entry_config.stage_a_run) / "best_model.pt") if entry_config.stage_a_run else None),
            overfit            = entry_config.overfit,
        )
        AutoencoderPipelineSupport.copy_base_subconfigs(self.trainer_config, base)
        self.trainer_config.geometry = entry_config.geometry.resolved(entry_config.paths.dataset_path, secondary_labels=self.factory._secondary_labels())

        self.dataset_config = self.factory.training_dataset_config()
        self.model_name     = entry_config.model_name

    def run(self):
        run_meta = TrainingRunMetadata(self.trainer_config, self.model_name, Path(self.trainer_config.io.logdir), self.entry.run_name)
        logger   = run_meta.logger

        gaussian_cfg = self.trainer_config.gaussian
        self.dataset_config.n_gaussians = gaussian_cfg.n_default_gaussians

        dataset_pipeline = DatasetPipeline(self.dataset_config, run_meta.run_directory, logger=logger, seed=self.entry.seed)
        x_len            = AutoencoderPipelineSupport.x_axis_length(dataset_pipeline)
        x_axis           = np.linspace(gaussian_cfg.x_min, gaussian_cfg.x_max, x_len, dtype=np.float32)
        self.dataset_config.x_axis = x_axis

        train_loader, val_loader, test_loader, datasets = dataset_pipeline.run()
        norm_stats  = datasets["train"].normalizer
        in_channels = datasets["train"].input_channels

        self.autoencoder_cfg.profile_length = x_len
        embedding_dim = self.autoencoder_cfg.embedding_dim

        backbone, backbone_cfg = self._build_backbone(in_channels, embedding_dim, x_len)
        autoencoder            = self._load_autoencoder()
        model                  = JepaModule(backbone, autoencoder)

        run_meta.save_trainer_config()
        run_meta.save_model_config(backbone_cfg, self.model_name)
        AutoencoderPipelineSupport.save_autoencoder_config(self.autoencoder_cfg, run_meta.metadata_directory)
        run_meta.save_run_summary(self.model_name, in_channels=in_channels, out_channels=gaussian_cfg.params_per_gaussian * gaussian_cfg.n_default_gaussians, x_axis_length=x_len)

        trainer = JepaPredictorTrainer(model, backbone_cfg, x_axis, self.trainer_config, run_meta.run_directory, logger, norm_stats)
        try:
            results = trainer.train(train_loader, val_loader, test_loader)
        finally:
            run_meta.close()
            logger.close()
        return results, run_meta.run_directory

    def _build_backbone(self, in_channels: int, embedding_dim: int, image_size: int):
        overrides = {"in_channels": in_channels, "out_channels": embedding_dim}
        if self.model_name in _IMAGE_SIZE_MODELS:
            overrides["image_size"] = image_size
        for k, v in self.entry.model_overrides.items():
            overrides[k] = v
        return get_model(self.model_name, **overrides)

    @staticmethod
    def validate_stage_a_checkpoint(ckpt_path, stage_a_mode: str) -> None:
        if ckpt_path is None:
            if stage_a_mode == "frozen":
                raise ValueError("stage_a_mode 'frozen' requires pretrained autoencoder weights from a Stage-A run; pass --stage_a_run, or use 'finetune'/'joint' to train the autoencoder from scratch.")
            return

        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"Stage-A checkpoint '{ckpt_path}' does not exist; expected 'best_model.pt' under the --stage_a_run directory.")

    def _load_autoencoder(self) -> ProfileAutoencoder:
        autoencoder = ProfileAutoencoder(self.autoencoder_cfg)
        ckpt_path   = self.trainer_config.stage_a_checkpoint
        self.validate_stage_a_checkpoint(ckpt_path, self.trainer_config.stage_a_mode)

        if ckpt_path is None:
            return autoencoder

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        autoencoder.load_state_dict(ckpt["params"])
        return autoencoder


class SingleJepaRunner:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        results, run_directory = JepaPipeline(self.config).run()
        if self.config.infer_after:
            from pipelines.jepa_pipeline.inference import JepaInferencePipeline
            JepaInferencePipeline(self.config.inference, run_directory).run()
