from __future__ import annotations

from pathlib                                 import Path
from typing                                  import Optional, Tuple
from torch.utils.data                        import DataLoader
from configuration.dataset_config            import DatasetConfiguration
from pipelines.dataset_pipeline.crop           import Cropper
from pipelines.dataset_pipeline.dataset         import PatchDataset
from pipelines.dataset_pipeline.loader          import Loader
from pipelines.dataset_pipeline.layout          import Layout
from pipelines.dataset_pipeline.metadata        import MetadataWriter
from pipelines.dataset_pipeline.stats           import Stats
from pipelines.dataset_pipeline.stats_computer  import StatsComputer
from pipelines.dataset_pipeline.patch           import Patcher
from tools.logger                               import Logger
from pipelines.dataset_pipeline.normalizer      import Normalizer
from pipelines.dataset_pipeline.augmentation    import SpatialAugmenter


class DatasetPipeline:
    def __init__(self, config : DatasetConfiguration, training_run_directory : Path, logger : Logger | None = None) -> None:
        self.config                 = config
        self.training_run_directory = Path(training_run_directory)

        log_dir = self.training_run_directory / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or Logger(log_dir=str(log_dir), name="dataset_pipeline", level="INFO")

        self.layout          = Layout(config.preprocessing_run_directory,  logger=self.logger, parameters_path=config.parameters_path)
        self.cropper         = Cropper(self.layout, config.split_regions,  logger=self.logger)
        self.metadata_writer = MetadataWriter(self.training_run_directory, logger=self.logger)
        self.augmenter       = SpatialAugmenter(config.augmentation,       logger=self.logger)

        ic = config.input_config
        oc = config.output_config
        
        self.logger.section("[DatasetPipeline Initialized]")
        self.logger.kv_table(
            {
                "Pre-processing Run"  : str(config.preprocessing_run_directory),
                "Training Run"        : str(self.training_run_directory),
                "Primary"             : f"use={ic.use_primary} rep={ic.primary_representation.value}",
                "Secondaries"         : f"use={ic.use_secondaries} rep={ic.secondaries_representation.value}",
                "Interferograms"      : f"use={ic.use_interferograms} rep={ic.interferograms_representation.value}",
                "DEM"                 : f"use={ic.use_dem}",
                "Output Parameters"   : ",".join(oc.role_names),
                "Patch Size"          : config.patch.size,
                "Patch Stride"        : config.patch.stride,
            },
            title="Dataset Creation",
        )

    def _build_dataset(self, split_name : str, normalizer : Optional[Normalizer] = None) -> Tuple[PatchDataset, Patcher]:
        region   = dict(self.config.split_regions.items())[split_name]
        arrays   = self.cropper.load_split(region)
        spatial  = (region.azimuth_size, region.range_size)
        
        patcher  = Patcher.build(
            spatial_size           = spatial,
            patch_size             = self.config.patch.size,
            stride                 = self.config.patch.stride,
            use_reflective_padding = self.config.patch.use_reflective_padding,
        )

        dataset = PatchDataset(
            inputs           = arrays["inputs"],
            gt_parameters    = arrays["parameters"],
            grid             = patcher,
            input_config     = self.config.input_config,
            output_config    = self.config.output_config,
            split_name       = split_name,
            normalizer       = normalizer,
            x_axis           = self.config.x_axis,
            n_gaussians      = self.config.n_gaussians,
            augmenter        = self.augmenter,
            dem              = arrays.get("dem") if self.config.input_config.use_dem else None,
        )
        
        return dataset, patcher

    def run(self) -> Tuple[DataLoader, DataLoader, DataLoader, dict[str, PatchDataset]]:
      
        train_ds, train_patcher = self._build_dataset("train")

        norm_stats = StatsComputer.compute(
            dataset       = train_ds,
            params_path   = self.config.parameters_path,
            logger        = self.logger,
            input_config  = self.config.input_config,
            output_config = self.config.output_config,
            n_slaves      = train_ds.n_slaves,
            n_gaussians   = self.config.n_gaussians,
            num_workers   = self.config.num_workers,
            max_samples   = 4000,
        )

        norm_stats.save(self.training_run_directory / "meta")

        normalizer        = Normalizer(norm_stats)
        train_ds.normalizer = normalizer

        val_ds,   val_patcher   = self._build_dataset("val",  normalizer=normalizer)
        test_ds,  test_patcher  = self._build_dataset("test", normalizer=normalizer)

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
