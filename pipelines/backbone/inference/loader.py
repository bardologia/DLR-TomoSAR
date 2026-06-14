from __future__ import annotations

from dataclasses import dataclass
from datetime    import datetime
from pathlib     import Path
from typing      import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from configuration.data.dataset_config    import DatasetConfiguration, InputConfig, OutputConfig, PatchConfiguration, SplitRegions
from configuration.inference.inference_config  import InferenceConfig
from tools.data.regions                   import CropRegion
from configuration.sar.gaussian_config import GaussianConfig
from models                          import get_model
from pipelines.backbone.dataset.datasets      import PatchDataset
from pipelines.backbone.dataset.normalization import Normalizer, Stats
from pipelines.backbone.dataset.spatial       import Cropper, GridInfo, Layout, Patcher
from tools.data.io             import FileIO, ModelConfigIO
from tools.data.gaussians                 import GaussianClamp
from tools.monitoring.logger                    import Logger
from tools.baselines           import TrackBaselines, TrackProfiles


_IMAGE_SIZE_MODELS = {"swin_unet", "transunet", "unetr"}


class InferenceMetadata:
    def __init__(self, config: InferenceConfig) -> None:
        self.config  = config
        paths        = config.paths

        base = config.run_directory / "inference"
        self.output_dir     = base / config.output_subdir if config.output_subdir else base / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.figures_dir    = self.output_dir / paths.figures_subdir
        self.animations_dir = self.output_dir / paths.animations_subdir
        self.logs_dir       = self.output_dir / paths.logs_subdir
        self.cube_dir       = self.output_dir / paths.cubes_subdir
        self.metrics_path   = self.output_dir / paths.metrics_filename
        self.report_path    = self.output_dir / paths.report_filename

    def figure_path(self, name: str, ext: str = "png") -> Path:
        return self.figures_dir / f"{name}.{ext}"

    def create_dirs(self) -> None:
        FileIO.ensure_dirs(
            self.output_dir,
            self.figures_dir,
            self.animations_dir,
            self.logs_dir,
            self.cube_dir,
        )


class ModelWrapper:
    def __init__(
        self,
        model,
        device,
        *,
        params_per_gaussian: int = 3,
        normalizer=None,
        x_axis: torch.Tensor | None = None,
        amp_max: float | None = None,
    ) -> None:

        self._model               = model
        self._device              = device
        self._params_per_gaussian = params_per_gaussian
        self._normalizer          = normalizer
        self._x_axis              = x_axis
        self._amp_max             = amp_max

    def denormalize_output(self, out: torch.Tensor) -> torch.Tensor:
        if self._normalizer is not None:
            out = self._normalizer.denormalize_output(out)

        if self._x_axis is not None and self._amp_max is not None:
            out = GaussianClamp.apply(
                out,
                x_axis      = self._x_axis.to(out.device),
                amp_max     = self._amp_max,
                ppg         = self._params_per_gaussian,
                leaky_slope = 0.0,
            )

        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self._device)

        with torch.no_grad():
            out = self._model(x_t)

        out = self.denormalize_output(out)

        return out.cpu().numpy()


@dataclass
class Run:
    model            : object
    model_name       : str
    in_channels      : int
    out_channels     : int
    x_axis           : np.ndarray
    x_axis_length    : int
    n_gaussians      : int
    dataset_config   : DatasetConfiguration
    split_name       : str
    split_region     : CropRegion
    global_crop      : CropRegion
    grid             : GridInfo
    dataset          : PatchDataset
    loader           : DataLoader
    checkpoint_meta  : dict
    complex_inputs   : np.ndarray | None = None
    n_secondaries    : int               = 0
    secondary_labels : list | None       = None
    full_curves      : np.ndarray | None = None
    track_baselines  : TrackBaselines | None = None
    track_profiles   : TrackProfiles  | None = None


class RunLoader:
    def __init__(self, run_directory: Path, logger: Logger) -> None:
        self.run_directory  = Path(run_directory)
        self.logger         = logger
        self.meta_directory = self.run_directory / "meta"

    def _read_json(self, name: str) -> dict:
        return FileIO.load_json(self.meta_directory / name)

    def _parse_split_payload(self, value):
        if isinstance(value, list):
            return [CropRegion(**region) for region in value]
        return CropRegion(**value)

    def _build_dataset_config(self, payload: dict, batch_size: Optional[int], num_workers: int) -> DatasetConfiguration:
        splits = payload["split_regions"]

        split_regions = SplitRegions(
            train = self._parse_split_payload(splits["train"]),
            val   = self._parse_split_payload(splits["val"]),
            test  = self._parse_split_payload(splits["test"]),
        )

        patch = PatchConfiguration(
            size                   = tuple(payload["patch"]["size"]),
            stride                 = int(payload["patch"]["stride"]),
            use_reflective_padding = bool(payload["patch"]["use_reflective_padding"]),
        )

        secondary_labels = payload["secondary_labels"]

        return DatasetConfiguration(
            preprocessing_run_directory = Path(payload["preprocessing_run_directory"]),
            parameters_path             = Path(payload["parameters_path"]),
            split_regions               = split_regions,
            secondary_labels            = tuple(secondary_labels) if secondary_labels is not None else None,
            patch                       = patch,
            input_config                = InputConfig.from_dict(payload["input_config"]),
            output_config               = OutputConfig.from_dict(payload["output_config"]),
            batch_size                  = batch_size if batch_size is not None else int(payload["batch_size"]),
            num_workers                 = int(num_workers),
            shuffle_train               = False,
            pin_memory                  = bool(payload["pin_memory"]),
        )

    def _build_model(self, model_name: str, in_channels: int, out_channels: int, image_size: int):
        model_config, _ = ModelConfigIO.load(self.meta_directory)

        overrides = {"in_channels": in_channels, "out_channels": out_channels}

        if model_name in _IMAGE_SIZE_MODELS:
            overrides["image_size"] = image_size

        model, _ = get_model(model_name, config=model_config, **overrides)

        return model

    def _load_checkpoint(self, ckpt_path: Path, device: str) -> tuple[dict, np.ndarray, dict]:
        ckpt   = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        raw    = ckpt["x_axis"]
        x_axis = raw.cpu().numpy().astype(np.float32) if hasattr(raw, "cpu") else np.asarray(raw, dtype=np.float32)

        meta = {
            "epoch"         : int(ckpt["epoch"]),
            "best_val_loss" : float(ckpt["best_val_loss"]),
            "best_epoch"    : int(ckpt["best_epoch"]),
        }

        return ckpt, x_axis, meta

    def _wrap_model(self, model, device: str, norm_stats: Stats, x_axis: np.ndarray, amp_max: float) -> ModelWrapper:
        return ModelWrapper(
            model               = model,
            device              = device,
            params_per_gaussian = 3,
            normalizer          = Normalizer(norm_stats),
            x_axis              = torch.from_numpy(x_axis),
            amp_max             = amp_max,
        )

    def _build_dataset(self, dataset_config : DatasetConfiguration, split_name : str, x_axis : np.ndarray, n_gaussians : int, norm_stats : Stats) -> Tuple[PatchDataset, GridInfo, CropRegion, CropRegion, dict]:
        layout  = Layout(dataset_config.preprocessing_run_directory, logger=self.logger, parameters_path=dataset_config.parameters_path)
        cropper = Cropper(layout, dataset_config.split_regions, logger=self.logger, secondary_labels=dataset_config.secondary_labels)

        regions = dataset_config.split_regions.regions(split_name)
        if len(regions) != 1:
            raise ValueError(f"Inference requires a single contiguous region for split '{split_name}'; found {len(regions)} disjoint regions. Stitching is only defined over one rectangular crop.")

        region = regions[0]
        arrays = cropper.load_split(region, load_tomogram=True)

        grid = Patcher.build(
            spatial_size           = (region.azimuth_size, region.range_size),
            patch_size             = dataset_config.patch.size,
            stride                 = dataset_config.patch.stride,
            use_reflective_padding = dataset_config.patch.use_reflective_padding,
        )

        inputs        = arrays["inputs"]
        gt_parameters = arrays["parameters"]

        dataset = PatchDataset(
            inputs           = inputs,
            gt_parameters    = gt_parameters,
            grid             = grid,
            input_config     = dataset_config.input_config,
            output_config    = dataset_config.output_config,
            split_name       = split_name,
            n_secondaries    = arrays["n_secondaries"],
            n_interferograms = arrays["n_interferograms"],
            normalizer       = Normalizer(norm_stats),
            x_axis           = x_axis,
            n_gaussians      = n_gaussians,
        )

        return dataset, grid, region, layout.global_crop, arrays

    def _load_track_info(self, dataset_config: DatasetConfiguration) -> Tuple[TrackBaselines | None, TrackProfiles | None]:
        dataset_dir = Path(dataset_config.preprocessing_run_directory)
        labels      = dataset_config.secondary_labels

        baselines_path = dataset_dir / "meta" / TrackBaselines.FILENAME
        profiles_path  = TrackProfiles.profiles_file(dataset_dir)

        if not baselines_path.is_file():
            raise FileNotFoundError(f"Track baselines file is required for inference but was not found: {baselines_path}")
        if not profiles_path.is_file():
            raise FileNotFoundError(f"Track profiles file is required for inference but was not found: {profiles_path}")

        baselines = TrackBaselines.load(baselines_path).subset(labels)
        profiles  = TrackProfiles.load(profiles_path).subset(labels)

        self.logger.kv_table(baselines.describe(), title="Tracks Used in This Run")

        return baselines, profiles

    def load(
        self,
        *,
        split           : str,
        batch_size      : Optional[int],
        num_workers     : int,
        device          : str,
        checkpoint_name : str,
    ) -> Run:

        self.logger.section("[Inference: Load Run]")
        self.logger.subsection(f"Run Directory : {self.run_directory} \n")

        run_summary    = self._read_json("run_summary.json")

        dataset_config = self._build_dataset_config(
            payload     = self._read_json("dataset_creation_config.json"),
            batch_size  = batch_size,
            num_workers = num_workers,
        )

        model_name     = str(run_summary["model_name"])
        in_channels    = int(run_summary["in_channels"])
        out_channels   = int(run_summary["out_channels"])
        n_gaussians    = out_channels // 3

        ckpt_path = self.run_directory / checkpoint_name
        model     = self._build_model(model_name, in_channels, out_channels, dataset_config.patch.size[0])
        model     = model.to(device)

        ckpt, x_axis, ckpt_meta = self._load_checkpoint(ckpt_path, device)
        model.load_state_dict(ckpt["params"])

        model.eval()
        norm_stats  = Stats.load(self.run_directory / "meta", self.logger)
        gauss_cfg   = GaussianConfig.from_dataset(dataset_config.preprocessing_run_directory, n_gaussians)
        model       = self._wrap_model(model, device, norm_stats, x_axis, gauss_cfg.amp_max)

        dataset, grid, region, global_crop, arrays = self._build_dataset(
            dataset_config = dataset_config,
            split_name     = split,
            x_axis         = x_axis,
            n_gaussians    = n_gaussians,
            norm_stats     = norm_stats,
        )

        track_baselines, track_profiles = self._load_track_info(dataset_config)

        loader = DataLoader(
            dataset,
            batch_size  = dataset_config.batch_size,
            shuffle     = False,
            num_workers = dataset_config.num_workers,
            pin_memory  = True,
            drop_last   = False,
        )

        self.logger.section(f"[Model]         : '{model_name}'")
        self.logger.kv_table({
            "Checkpoint":  ckpt_path,
            "In channels":  in_channels,
            "Out channels": out_channels,
            "K gaussians":  n_gaussians,
        })

        self.logger.section(f"[Split]         : '{split}'")
        self.logger.kv_table({
            "Patches":      grid.grid.number_of_patches,
            "Azimuth size": region.azimuth_size,
            "Range size":   region.range_size,
            "X-axis length": x_axis.size,
        })

        return Run(
            model            = model,
            model_name       = model_name,
            in_channels      = in_channels,
            out_channels     = out_channels,
            x_axis           = x_axis,
            x_axis_length    = int(x_axis.size),
            n_gaussians      = n_gaussians,
            dataset_config   = dataset_config,
            split_name       = split,
            split_region     = region,
            global_crop      = global_crop,
            grid             = grid.grid,
            dataset          = dataset,
            loader           = loader,
            checkpoint_meta  = ckpt_meta,
            complex_inputs   = arrays["inputs"],
            n_secondaries    = arrays["n_secondaries"],
            secondary_labels = arrays["secondary_labels"],
            full_curves      = arrays["tomogram"],
            track_baselines  = track_baselines,
            track_profiles   = track_profiles,
        )
