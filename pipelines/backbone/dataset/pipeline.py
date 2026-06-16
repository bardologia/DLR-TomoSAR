from __future__ import annotations

from dataclasses import asdict
from pathlib     import Path
from typing      import Optional, Tuple

from torch.utils.data import DataLoader

from configuration.dataset import DatasetConfig
from tools.data.regions                       import CropRegion
from pipelines.backbone.dataset.augmentation  import SpatialAugmenter
from pipelines.backbone.dataset.datasets      import MultiRegionDataset, PatchDataset
from pipelines.shared.loaders                 import Loader
from pipelines.backbone.dataset.normalization import Normalizer, StatsComputer
from pipelines.backbone.dataset.spatial       import Cropper, GridInfo, Layout, Patcher
from tools.data.io                            import FileIO
from tools.monitoring.logger                  import Logger


class MetadataWriter:
    def __init__(self, run_directory: Path, logger: Logger) -> None:
        self.run_directory      = Path(run_directory)
        self.logger             = logger
        self.metadata_directory = self.run_directory / "meta"

        self.outpaths           = {
            "dataset_configuration" : self.metadata_directory / "dataset_creation_config.json",
            "crop"                  : self.metadata_directory / "crop.json",
            "patch"                 : self.metadata_directory / "patch.json",
        }

        FileIO.ensure_dir(self.metadata_directory)

        self.logger.section("[MetadataWriter Initialized]")
        self.logger.subsection(f"Metadata Directory : {self.metadata_directory} \n")

    def save_dataset_configuration(self, config: DatasetConfig) -> Path:
        out_path = self.outpaths["dataset_configuration"]
        payload  = asdict(config)

        payload["preprocessing_run_directory"] = str(config.preprocessing_run_directory)
        payload["input_config"]                = config.input_config.as_dict()
        payload["output_config"]               = config.output_config.as_dict()
        payload.pop("x_axis", None)

        return FileIO.save_json(payload, out_path)

    def save_crop_metadata(self, global_crop: CropRegion, splits: dict[str, CropRegion]) -> Path:
        out_path = self.outpaths["crop"]
        payload  = {"global_crop" : list(global_crop.as_tuple()), "splits" : {name: self._region_payload(value) for name, value in splits.items()}}

        return FileIO.save_json(payload, out_path)

    def save_patch_metadata(self, grids: dict[str, GridInfo]) -> Path:
        out_path = self.outpaths["patch"]
        payload  = {name: self._grid_payload(value) for name, value in grids.items()}

        return FileIO.save_json(payload, out_path)

    def _region_payload(self, value):
        if isinstance(value, (list, tuple)):
            return [list(region.as_tuple()) for region in value]
        return list(value.as_tuple())

    def _grid_payload(self, value):
        if isinstance(value, (list, tuple)):
            return [grid.as_dict() for grid in value]
        return value.as_dict()


class DatasetPipeline:
    def __init__(self, config : DatasetConfig, training_run_directory : Path, logger : Logger | None = None, seed : int = 0) -> None:
        self.config                 = config
        self.training_run_directory = Path(training_run_directory)
        self.seed                   = int(seed)

        log_dir = self.training_run_directory / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or Logger(log_dir=str(log_dir), name="dataset_pipeline", level="INFO")

        self.layout          = Layout(config.preprocessing_run_directory,  logger=self.logger, parameters_path=config.parameters_path)
        self.cropper         = Cropper(self.layout, config.split_regions,  logger=self.logger, secondary_labels=config.secondary_labels)
        self.metadata_writer = MetadataWriter(self.training_run_directory, logger=self.logger)
        self.augmenter       = SpatialAugmenter(config.augmentation,       logger=self.logger, seed=self.seed)

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

    @property
    def profile_length(self) -> int:
        return self.layout.profile_length

    def _build_dataset(self, split_name : str, normalizer : Optional[Normalizer] = None):
        regions  = self.config.split_regions.regions(split_name)
        parts    = []
        patchers = []

        for region in regions:
            arrays  = self.cropper.load_split(region)
            spatial = (region.azimuth_size, region.range_size)

            patcher = Patcher.build(
                spatial_size           = spatial,
                patch_size             = self.config.patch.size,
                stride                 = self.config.patch.stride,
                use_reflective_padding = self.config.patch.use_reflective_padding,
            )

            part = PatchDataset(
                inputs           = arrays["inputs"],
                gt_parameters    = arrays["parameters"],
                grid             = patcher,
                input_config     = self.config.input_config,
                output_config    = self.config.output_config,
                split_name       = split_name,
                n_secondaries    = arrays["n_secondaries"],
                n_interferograms = arrays["n_interferograms"],
                normalizer       = normalizer,
                x_axis           = self.config.x_axis,
                n_gaussians      = self.config.n_gaussians,
                augmenter        = self.augmenter,
                dem              = arrays.get("dem") if self.config.input_config.use_dem else None,
            )

            parts.append(part)
            patchers.append(patcher)

        if len(parts) == 1:
            return parts[0], patchers[0]

        self.logger.subsection(f"Split '{split_name}': {len(parts)} disjoint regions, {sum(len(p) for p in parts)} patches total")

        return MultiRegionDataset(parts), patchers

    def _patch_grids(self, patcher):
        if isinstance(patcher, list):
            return [p.grid for p in patcher]
        return patcher.grid

    def run(self) -> Tuple[DataLoader, DataLoader, DataLoader, dict[str, PatchDataset]]:

        train_ds, train_patcher = self._build_dataset("train")

        norm_stats = StatsComputer.compute(
            dataset          = train_ds,
            logger           = self.logger,
            input_config     = self.config.input_config,
            output_config    = self.config.output_config,
            n_secondaries    = train_ds.n_secondaries,
            n_interferograms = train_ds.n_interferograms,
            n_gaussians      = self.config.n_gaussians,
            num_workers      = self.config.num_workers,
            max_samples      = 4000,
        )

        norm_stats.save(self.training_run_directory / "meta")

        normalizer          = Normalizer(norm_stats)
        train_ds.normalizer = normalizer

        val_ds,   val_patcher  = self._build_dataset("val",  normalizer=normalizer)
        test_ds,  test_patcher = self._build_dataset("test", normalizer=normalizer)

        train_loader, val_loader, test_loader = Loader.build(
            train_dataset   = train_ds,
            val_dataset     = val_ds,
            test_dataset    = test_ds,
            batch_size      = self.config.batch_size,
            num_workers     = self.config.num_workers,
            pin_memory      = self.config.pin_memory,
            shuffle_train   = self.config.shuffle_train,
            prefetch_factor = self.config.prefetch_factor,
            seed            = self.seed,
            logger          = self.logger,
        )

        self.metadata_writer.save_dataset_configuration(self.config)

        self.metadata_writer.save_crop_metadata(
            global_crop = self.layout.global_crop,
            splits      = dict(self.config.split_regions.items()),
        )

        self.metadata_writer.save_patch_metadata({
            "train" : self._patch_grids(train_patcher),
            "val"   : self._patch_grids(val_patcher),
            "test"  : self._patch_grids(test_patcher),
        })

        datasets = {"train": train_ds, "val": val_ds, "test": test_ds}

        return train_loader, val_loader, test_loader, datasets
