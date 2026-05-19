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
from configuration.preprocessing_config              import CropRegion
from pipelines.dataset_creation_pipeline.crop        import Cropper, Layout
from pipelines.dataset_creation_pipeline.load        import PatchDataset
from pipelines.dataset_creation_pipeline.normalize   import Stats, Normalizer
from pipelines.dataset_creation_pipeline.patch       import Patcher, GridInfo
from models                                          import get_model


_IMAGE_SIZE_MODELS = {"swin_unet", "transunet", "unetr"}


@dataclass
class LoadedRun:
    run_directory   : Path
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


class DirectoryLoader:
    def __init__(self, run_directory: Path, logger: Logger) -> None:
        self.run_directory  = Path(run_directory)
        self.logger         = logger
        self.meta_directory = self.run_directory / "meta"

    def _read_json(self, name: str) -> dict:
        path = self.meta_directory / name
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_dataset_config(self, payload: dict, batch_size: Optional[int], num_workers: int) -> DatasetConfiguration:
        splits        = payload["split_regions"]
        
        split_regions = SplitRegions(
            train = CropRegion(**splits["train"]),
            val   = CropRegion(**splits["val"]),
            test  = CropRegion(**splits["test"]),
        )

        patch_payload = payload["patch"]
        
        patch         = PatchConfiguration(
            size                   = tuple(patch_payload["size"]),
            stride                 = int(patch_payload["stride"]),
            use_reflective_padding = bool(patch_payload["use_reflective_padding"]),
        )

        raw_params_path = payload["parameters_path"]
        parameters_path = Path(raw_params_path) if raw_params_path != "None" else None

        return DatasetConfiguration(
            preprocessing_run_directory  = Path(payload["preprocessing_run_directory"]),
            parameters_path              = parameters_path,
            split_regions                = split_regions,
            patch                        = patch,
            input_config                 = InputConfig.from_dict(payload["input_config"]),
            output_config                = OutputConfig.from_dict(payload["output_config"]),
            batch_size                   = batch_size if batch_size is not None else int(payload["batch_size"]),
            num_workers                  = int(num_workers),
            shuffle_train                = False,
            pin_memory                   = bool(payload["pin_memory"]),
        )

    def _build_split_dataset(
        self,
        dataset_config : DatasetConfiguration,
        split_name     : str,
        x_axis         : np.ndarray,
        n_gaussians    : int,
    ) -> Tuple[PatchDataset, GridInfo, CropRegion, CropRegion]:
        
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

        norm_stats = Stats.load(self.run_directory / "meta", self.logger)

        dataset = PatchDataset(
            inputs        = arrays["inputs"],
            gt_parameters = arrays["parameters"],
            grid          = grid,
            input_config  = dataset_config.input_config,
            output_config = dataset_config.output_config,
            split_name    = split_name,
            logger        = self.logger,
            norm_stats    = norm_stats,
            x_axis        = x_axis,
            n_gaussians   = n_gaussians,
        )

        return dataset, grid, region, layout.global_crop

    def _build_model(self, model_name: str, in_channels: int, out_channels: int, image_size: int):
        overrides = {"in_channels": in_channels, "out_channels": out_channels}
        if model_name in _IMAGE_SIZE_MODELS:
            overrides["image_size"] = image_size
        return get_model(model_name, **overrides)

    def _apply_ema(self, model, ema_state: dict) -> int:
        shadow  = ema_state["shadow"]
        applied = 0
      
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in shadow:
                    param.data.copy_(shadow[name].to(param.device, dtype=param.dtype))
                    applied += 1
      
        return applied

    def _load_checkpoint(self, ckpt_path: Path, device: str) -> dict:
        return torch.load(str(ckpt_path), map_location=device, weights_only=False)

    def _extract_axis(self, ckpt: dict) -> np.ndarray:
        raw = ckpt["x_axis"]
        return raw.cpu().numpy().astype(np.float32) if hasattr(raw, "cpu") else np.asarray(raw, dtype=np.float32)

    def _extract_ckpt_meta(self, ckpt: dict) -> dict:
        return {
            "epoch"         : int(ckpt["epoch"]),
            "best_val_loss" : float(ckpt["best_val_loss"]),
            "best_epoch"    : int(ckpt["best_epoch"]),
            "best_metrics"  : dict(ckpt["best_metrics"]),
        }

    def _wrap_model(self, model, device: str) -> ModelWrapper:
        norm_stats = Stats.load(self.run_directory / "meta", self.logger)
        return ModelWrapper(
            model               = model,
            device              = device,
            params_per_gaussian = 3,
            normalizer          = Normalizer(norm_stats),
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
    ) -> LoadedRun:
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

        ckpt = self._load_checkpoint(ckpt_path, device)
        model.load_state_dict(ckpt["model_state_dict"])

        used_ema = False
        if use_ema:
            n_applied = self._apply_ema(model, ckpt["ema_state_dict"])
            used_ema  = n_applied > 0
            self.logger.subsection(f"EMA           : applied to {n_applied} parameters")

        model.eval()
        model = self._wrap_model(model, device)

        x_axis    = self._extract_axis(ckpt)
        ckpt_meta = self._extract_ckpt_meta(ckpt)

        dataset, grid, region, global_crop = self._build_split_dataset(
            dataset_config = dataset_config,
            split_name     = split,
            x_axis         = x_axis,
            n_gaussians    = n_gaussians,
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
        self.logger.subsection(f"Checkpoint    : {ckpt_path}")
        self.logger.subsection(f"In channels   : {in_channels}")
        self.logger.subsection(f"Out channels  : {out_channels}")
        self.logger.subsection(f"K gaussians   : {n_gaussians} \n")
        
        self.logger.section(f"[Split]         : '{split}'")
        self.logger.subsection(f"Patches       : {grid.grid.number_of_patches}")
        self.logger.subsection(f"Azimuth size  : {region.azimuth_size}")
        self.logger.subsection(f"Range size    : {region.range_size}\n")
        self.logger.subsection(f"X-axis length : {x_axis.size}")


        return LoadedRun(
            run_directory   = self.run_directory,
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
