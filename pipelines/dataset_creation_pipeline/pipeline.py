from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from torch.utils.data import DataLoader

from configuration.dataset_config                    import DatasetCreationConfiguration
from pipelines.dataset_creation_pipeline.crop        import Cropper
from pipelines.dataset_creation_pipeline.load        import Loader, PatchDataset
from pipelines.dataset_creation_pipeline.metadata    import DatasetLayout, DatasetMetadataWriter
from pipelines.dataset_creation_pipeline.normalize   import Stats
from pipelines.dataset_creation_pipeline.patch       import Patcher
from tools.logger                                    import Logger

class DatasetCreationPipeline:
    def __init__(self, config : DatasetCreationConfiguration, training_run_directory : Path, logger : Logger | None = None) -> None:
        self.config                 = config
        self.training_run_directory = Path(training_run_directory)

        log_dir = self.training_run_directory / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logger or Logger(log_dir = str(log_dir), name = "dataset_creation", level = "INFO")

        self.layout          = DatasetLayout(config.preprocessing_run_directory, logger=self.logger, parameters_path=config.parameters_path)
        self.cropper         = Cropper(self.layout, config.split_regions, logger=self.logger)
        self.metadata_writer = DatasetMetadataWriter(self.training_run_directory, logger=self.logger)

        ic = config.input_config
        self.logger.section("[DatasetCreationPipeline Initialized]")
        self.logger.kv_table(
            {
                "Pre-processing Run"  : str(config.preprocessing_run_directory),
                "Training Run"        : str(self.training_run_directory),
                "Master"              : f"use={ic.use_master} rep={ic.master_representation.value}",
                "Slaves"              : f"use={ic.use_slaves} rep={ic.slaves_representation.value}",
                "Interferograms"      : f"use={ic.use_interferograms} rep={ic.interferograms_representation.value}",
                "Patch Size"          : config.patch.size,
                "Patch Stride"        : config.patch.stride,
            },
            title="Dataset Creation",
        )

    def _build_dataset(self, split_name : str, norm_stats : Optional[Stats] = None) -> Tuple[PatchDataset, Patcher]:
        region   = dict(self.config.split_regions.items())[split_name]
        arrays   = self.cropper.load_split(region)
        spatial  = (region.azimuth_size, region.range_size)
        
        patcher  = Patcher.build(
            spatial_size           = spatial,
            patch_size             = self.config.patch.size,
            stride                 = self.config.patch.stride,
            use_reflective_padding = self.config.patch.use_reflective_padding,
        )

        gt_parameters = arrays["parameters"]
        
        dataset       = PatchDataset(
            inputs           = arrays["inputs"],
            gt_parameters    = gt_parameters,
            grid             = patcher,
            input_config     = self.config.input_config,
            split_name       = split_name,
            logger           = self.logger,
            norm_stats       = norm_stats,
            x_axis           = self.config.x_axis,
            n_gaussians      = self.config.n_gaussians,
        )
        
        return dataset, patcher

    def run(self) -> Tuple[DataLoader, DataLoader, DataLoader, dict[str, PatchDataset]]:
        self.logger.section("[Dataset Creation Pipeline Execution]")

        train_ds, train_patcher = self._build_dataset("train")

        norm_stats = Stats.compute_from_dataset(
            dataset             = train_ds,
            logger              = self.logger,
            input_config        = self.config.input_config,
            n_slaves            = train_ds.n_slaves,
            params_per_gaussian = 3,
            input_mode          = self.config.input_normalization_mode,
            output_mode         = self.config.output_normalization_mode,
            num_workers         = self.config.num_workers,
        )
        
        norm_stats.save(self.training_run_directory / "meta", self.logger)

        train_ds, train_patcher = self._build_dataset("train", norm_stats=norm_stats)
        val_ds,   val_patcher   = self._build_dataset("val",   norm_stats=norm_stats)
        test_ds,  test_patcher  = self._build_dataset("test",  norm_stats=norm_stats)

        train_loader, val_loader, test_loader = Loader.build(
            train_dataset = train_ds,
            val_dataset   = val_ds,
            test_dataset  = test_ds,
            batch_size    = self.config.batch_size,
            num_workers   = self.config.num_workers,
            pin_memory    = self.config.pin_memory,
            shuffle_train = self.config.shuffle_train,
            logger        = self.logger,
        )

        self.metadata_writer.save_dataset_configuration(self.config)
        
        self.metadata_writer.save_crop_metadata(
            global_crop = self.layout.global_crop,
            splits      = dict(self.config.split_regions.items()),
        )
        
        self.metadata_writer.save_patch_metadata({
            "train" : train_patcher.grid,
            "val"   : val_patcher.grid,
            "test"  : test_patcher.grid,
        })

        datasets = {"train": train_ds, "val": val_ds, "test": test_ds}
        
        return train_loader, val_loader, test_loader, datasets
