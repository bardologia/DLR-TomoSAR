from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from configuration.dataset_config                     import DatasetCreationConfiguration, TargetMode
from configuration.training_config                    import TrainerConfig
from pipelines.dataset_creation_pipeline.pipeline     import DatasetCreationPipeline
from models                                           import get_model
from pipelines.training_pipeline.metadata             import TrainingRunMetadata
from pipelines.training_pipeline.trainer              import Trainer


_IMAGE_SIZE_MODELS = {"swin_unet", "transunet", "unetr"}


class TrainingPipeline:
    def __init__(
        self,
        trainer_config : TrainerConfig,
        dataset_config : DatasetCreationConfiguration,
        model_name     : str,
    ) -> None:

        patch_height, patch_width = dataset_config.patch.size
   
        self.trainer_config = trainer_config
        self.dataset_config = dataset_config
        self.model_name     = model_name
        self.image_size     = patch_height

        self.run_metadata = TrainingRunMetadata(
            trainer_config = trainer_config,
            model_name     = model_name,
            base_logdir    = Path(trainer_config.io.logdir),
        )
        self.logger = self.run_metadata.logger

        self.dataset_pipeline = DatasetCreationPipeline(
            config                 = dataset_config,
            training_run_directory = self.run_metadata.run_directory,
            logger                 = self.logger,
        )

    def _build_model(self, in_channels: int, out_channels: int) -> torch.nn.Module:
        overrides = {"in_channels": in_channels, "out_channels": out_channels}
        if self.model_name in _IMAGE_SIZE_MODELS:
            overrides["image_size"] = self.image_size

        model = get_model(self.model_name, **overrides)
        self.logger.section("[Model Built]")
        self.logger.subsection(f"Architecture : {self.model_name}")
        self.logger.subsection(f"In Channels  : {in_channels}")
        self.logger.subsection(f"Out Channels : {out_channels}")
        return model

    def run(self):
        self.logger.section("[Training Pipeline Execution]")

        gaussian_cfg = self.trainer_config.gaussian

        self.dataset_config.n_gaussians = gaussian_cfg.n_default_gaussians

        if self.dataset_config.target_mode == TargetMode.GAUSSIAN_FIT and self.dataset_config.x_axis is None:
            tomo_path      = self.dataset_pipeline.layout.artifact_path("full_tomogram")
            tomo_mmap      = np.load(str(tomo_path), mmap_mode="r", allow_pickle=False)
            x_axis_length  = int(tomo_mmap.shape[0])
            self.dataset_config.x_axis = np.linspace(gaussian_cfg.x_min, gaussian_cfg.x_max, x_axis_length, dtype=np.float32)

        train_loader, val_loader, test_loader, datasets = self.dataset_pipeline.run()

        train_dataset = datasets["train"]
        in_channels   = train_dataset.input_channels
        n_gaussians   = gaussian_cfg.n_default_gaussians
        out_channels  = gaussian_cfg.params_per_gaussian * n_gaussians

        x_axis_length = train_dataset.target_channels
        x_axis        = torch.from_numpy(self.dataset_config.x_axis).float() \
                        if self.dataset_config.x_axis is not None \
                        else torch.linspace(gaussian_cfg.x_min, gaussian_cfg.x_max, x_axis_length)

        model = self._build_model(in_channels=in_channels, out_channels=out_channels)

        self.run_metadata.save_trainer_config()
        
        self.run_metadata.save_run_summary(
            model_name    = self.model_name,
            in_channels   = in_channels,
            out_channels  = out_channels,
            x_axis_length = x_axis_length,
        )

        trainer = Trainer(
            model      = model,
            x_axis     = x_axis,
            config     = self.trainer_config,
            run_dir    = self.run_metadata.run_directory,
            logger     = self.logger,
            norm_stats = getattr(train_dataset, "norm_stats", None),
        )

        try:
            results = trainer.train(train_loader, val_loader, test_loader)
        finally:
            self.run_metadata.close()
            self.logger.close()

        return results
