from __future__ import annotations

import json
import torch
from dataclasses import dataclass
from pathlib     import Path
from typing      import Optional, Tuple
import numpy as np

from torch.utils.data                                import DataLoader
from pipelines.inference_pipeline.wrapper            import ModelWrapper
from tools.logger                                    import Logger
from configuration.dataset_config                    import DatasetConfiguration, InputConfig, OutputConfig, PatchConfiguration, SplitRegions
from configuration.processing_config                 import CropRegion
from pipelines.dataset_pipeline.crop                 import Cropper
from pipelines.dataset_pipeline.layout               import Layout
from pipelines.dataset_pipeline.dataset              import PatchDataset
from pipelines.dataset_pipeline.stats                import Stats
from pipelines.dataset_pipeline.normalizer           import Normalizer
from pipelines.dataset_pipeline.patch                import Patcher, GridInfo
from models                                          import get_model
from configuration.training_config                   import GaussianConfig


_IMAGE_SIZE_MODELS = {"swin_unet", "transunet", "unetr"}


@dataclass
class Run:
    model           : object
    model_name      : str
    in_channels     : int
    out_channels    : int
    x_axis          : np.ndarray
    x_axis_length   : int
    n_gaussians     : int
    dataset_config  : DatasetConfiguration
    split_name      : str
    split_region    : CropRegion
    global_crop     : CropRegion
    grid            : GridInfo
    dataset         : PatchDataset
    loader          : DataLoader
    checkpoint_meta : dict
    used_ema        : bool


class RunLoader:
    def __init__(self, run_directory: Path, logger: Logger) -> None:
        self.run_directory  = Path(run_directory)
        self.logger         = logger
        self.meta_directory = self.run_directory / "meta"

    def _read_json(self, name: str) -> dict:
        path = self.meta_directory / name
        
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_dataset_config(self, payload: dict, batch_size: Optional[int], num_workers: int) -> DatasetConfiguration:
        splits = payload["split_regions"]
        
        split_regions = SplitRegions(
            train = CropRegion(**splits["train"]),
            val   = CropRegion(**splits["val"]),
            test  = CropRegion(**splits["test"]),
        )
       
        patch = PatchConfiguration(
            size                   = tuple(payload["patch"]["size"]),
            stride                 = int(payload["patch"]["stride"]),
            use_reflective_padding = bool(payload["patch"]["use_reflective_padding"]),
        )

        return DatasetConfiguration(
            preprocessing_run_directory = Path(payload["preprocessing_run_directory"]),
            parameters_path             = Path(payload["parameters_path"]),
            split_regions               = split_regions,
            patch                       = patch,
            input_config                = InputConfig.from_dict(payload["input_config"]),
            output_config               = OutputConfig.from_dict(payload["output_config"]),
            batch_size                  = batch_size if batch_size is not None else int(payload["batch_size"]),
            num_workers                 = int(num_workers),
            shuffle_train               = False,
            pin_memory                  = bool(payload["pin_memory"]),
        )

    def _build_dataset(self, dataset_config : DatasetConfiguration, split_name : str, x_axis : np.ndarray, n_gaussians : int, norm_stats : Stats) -> Tuple[PatchDataset, GridInfo, CropRegion, CropRegion]:
        layout  = Layout(dataset_config.preprocessing_run_directory, logger=self.logger, parameters_path=dataset_config.parameters_path)
        cropper = Cropper(layout, dataset_config.split_regions, logger=self.logger)
        region  = dict(dataset_config.split_regions.items())[split_name]
        arrays  = cropper.load_split(region)

        grid = Patcher.build(
            spatial_size           = (region.azimuth_size, region.range_size),
            patch_size             = dataset_config.patch.size,
            stride                 = dataset_config.patch.stride,
            use_reflective_padding = dataset_config.patch.use_reflective_padding,
        )

        inputs        = arrays["inputs"]
        gt_parameters = arrays["parameters"]

        dataset = PatchDataset(
            inputs        = inputs,
            gt_parameters = gt_parameters,
            grid          = grid,
            input_config  = dataset_config.input_config,
            output_config = dataset_config.output_config,
            split_name    = split_name,
            normalizer    = Normalizer(norm_stats),
            x_axis        = x_axis,
            n_gaussians   = n_gaussians,
        )

        return dataset, grid, region, layout.global_crop

    def _build_model(self, model_name: str, in_channels: int, out_channels: int, image_size: int):
        overrides = {"in_channels": in_channels, "out_channels": out_channels}
        
        if model_name in _IMAGE_SIZE_MODELS:
            overrides["image_size"] = image_size
        
        model, _ = get_model(model_name, **overrides)
        
        return model

    def _apply_ema(self, model, ckpt: dict, use_ema: bool) -> bool:
        if not use_ema:
            return False
        ema_state = ckpt.get("ema_shadow", {})
        if not ema_state or not ema_state.get("shadow"):
            return False
        shadow  = ema_state["shadow"]
        applied = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in shadow:
                    param.data.copy_(shadow[name].to(param.device, dtype=param.dtype))
                    applied += 1
        
        self.logger.subsection(f"EMA : applied to {applied} parameters")
        return True

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

    def load(
        self,
        *,
        split           : str,
        batch_size      : Optional[int],
        num_workers     : int,
        device          : str,
        use_ema         : bool,
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
        used_ema = self._apply_ema(model, ckpt, use_ema)

        model.eval()
        norm_stats  = Stats.load(self.run_directory / "meta", self.logger)
        gauss_cfg   = GaussianConfig.from_dataset(dataset_config.preprocessing_run_directory, n_gaussians)
        model       = self._wrap_model(model, device, norm_stats, x_axis, gauss_cfg.amp_max)

        dataset, grid, region, global_crop = self._build_dataset(
            dataset_config = dataset_config,
            split_name     = split,
            x_axis         = x_axis,
            n_gaussians    = n_gaussians,
            norm_stats     = norm_stats,
        )

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
            model           = model,
            model_name      = model_name,
            in_channels     = in_channels,
            out_channels    = out_channels,
            x_axis          = x_axis,
            x_axis_length   = int(x_axis.size),
            n_gaussians     = n_gaussians,
            dataset_config  = dataset_config,
            split_name      = split,
            split_region    = region,
            global_crop     = global_crop,
            grid            = grid.grid,
            dataset         = dataset,
            loader          = loader,
            checkpoint_meta = ckpt_meta,
            used_ema        = used_ema,
        )
