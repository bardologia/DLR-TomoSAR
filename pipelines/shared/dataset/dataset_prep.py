from __future__ import annotations

from pipelines.backbone.dataset.pipeline import DatasetPipeline
from tools.data.gaussians                import GaussianAxis


class BackboneDatasetPreparation:
    def __init__(self, dataset_config, trainer_config, run_meta, logger, seed, build_geometry_field: bool = False, height_axis_convention: str = "height") -> None:
        self.dataset_config         = dataset_config
        self.trainer_config         = trainer_config
        self.run_meta               = run_meta
        self.logger                 = logger
        self.seed                   = seed
        self.build_geometry_field   = build_geometry_field
        self.height_axis_convention = height_axis_convention

    def run(self):
        gaussian_cfg                    = self.trainer_config.gaussian
        self.dataset_config.n_gaussians = gaussian_cfg.n_default_gaussians

        dataset_pipeline = DatasetPipeline(self.dataset_config, self.run_meta.run_directory, logger=self.logger, seed=self.seed, height_axis_convention=self.height_axis_convention, build_geometry_field=self.build_geometry_field)
        x_len            = dataset_pipeline.profile_length
        x_axis           = GaussianAxis.build(gaussian_cfg.x_min, gaussian_cfg.x_max, x_len)

        self.dataset_config.x_axis = x_axis

        train_loader, val_loader, test_loader, datasets = dataset_pipeline.run()
        
        return (train_loader, val_loader, test_loader), datasets, x_axis, x_len
