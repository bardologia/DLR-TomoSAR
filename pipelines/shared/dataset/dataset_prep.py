from __future__ import annotations

import numpy as np

from pipelines.backbone.dataset.pipeline import DatasetPipeline


class BackboneDatasetPreparation:
    def __init__(self, dataset_config, trainer_config, run_meta, logger, seed) -> None:
        self.dataset_config = dataset_config
        self.trainer_config = trainer_config
        self.run_meta       = run_meta
        self.logger         = logger
        self.seed           = seed

    def run(self):
        gaussian_cfg                    = self.trainer_config.gaussian
        self.dataset_config.n_gaussians = gaussian_cfg.n_default_gaussians

        dataset_pipeline = DatasetPipeline(self.dataset_config, self.run_meta.run_directory, logger=self.logger, seed=self.seed)
        x_len            = dataset_pipeline.profile_length
        x_axis           = np.linspace(gaussian_cfg.x_min, gaussian_cfg.x_max, x_len, dtype=np.float32)

        self.dataset_config.x_axis = x_axis

        train_loader, val_loader, test_loader, datasets = dataset_pipeline.run()
        
        return (train_loader, val_loader, test_loader), datasets, x_axis, x_len
