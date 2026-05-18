from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib     import Path
from typing      import Optional, Tuple
import numpy as np

from torch.utils.data                                import DataLoader
from pipelines.inference_pipeline.wrapper            import ModelWrapper
from tools.logger                                    import Logger
from configuration.dataset_config                    import DatasetCreationConfiguration,InputConfig, PatchConfiguration, SplitRegions
from configuration.preprocessing_config              import CropRegion
from pipelines.dataset_creation_pipeline.crop        import Cropper, DatasetLayout
from pipelines.dataset_creation_pipeline.load        import PatchDataset
from pipelines.dataset_creation_pipeline.normalize   import NormalizationStats, Normalizer
from pipelines.dataset_creation_pipeline.patch       import Patcher, GridInfo
from models                                          import MODEL_REGISTRY, get_model


_IMAGE_SIZE_MODELS = {"swin_unet", "transunet", "unetr"}


@dataclass
class LoadedRun:
    run_directory   : Path
    model           : object          # ModelWrapper (JAX) or torch.nn.Module (legacy)
    model_name      : str
    in_channels     : int
    out_channels    : int
    x_axis          : np.ndarray
    x_axis_length   : int
    n_gaussians     : int
    has_noise_head  : bool
    dataset_config  : DatasetCreationConfiguration
    split_name      : str
    split_region    : CropRegion
    global_crop     : CropRegion
    grid            : GridInfo
    dataset         : PatchDataset
    loader          : DataLoader
    checkpoint_meta : dict
    used_ema        : bool


class RunDirectoryLoader:
    PARAMS_PER_GAUSSIAN = 3

    def __init__(self, run_directory: Path, logger: Logger) -> None:
        self.run_directory  = Path(run_directory)
        self.logger         = logger
        self.meta_directory = self.run_directory / "meta"
        if not self.meta_directory.exists():
            raise FileNotFoundError(f"meta/ folder missing in run dir: {self.run_directory}")

    def _read_json(self, name: str) -> dict:
        path = self.meta_directory / name
        if not path.exists():
            raise FileNotFoundError(f"Missing metadata file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_dataset_config(self, payload: dict, batch_size: Optional[int], num_workers: int) -> DatasetCreationConfiguration:
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

        raw_params_path = payload.get("parameters_path")
        parameters_path = Path(raw_params_path) if raw_params_path and raw_params_path != "None" else None

        
        return DatasetCreationConfiguration(
            preprocessing_run_directory  = Path(payload["preprocessing_run_directory"]),
            parameters_path              = parameters_path,
            split_regions                = split_regions,
            patch                        = patch,
            input_config                 = InputConfig.from_dict(payload["input_config"]),
            batch_size                   = batch_size if batch_size is not None else int(payload["batch_size"]),
            num_workers                  = int(num_workers),
            shuffle_train                = False,
            pin_memory                   = bool(payload["pin_memory"]),
        )

    def _build_split_dataset(
        self,
        dataset_config : DatasetCreationConfiguration,
        split_name     : str,
        run_directory  : Optional[Path] = None,
        x_axis         : Optional[np.ndarray] = None,
        n_gaussians    : int = 1,
    ) -> Tuple[PatchDataset, GridInfo, CropRegion, CropRegion]:
        
        layout  = DatasetLayout(dataset_config.preprocessing_run_directory, logger=self.logger, parameters_path=dataset_config.parameters_path)
        cropper = Cropper(layout, dataset_config.split_regions, logger=self.logger)
        region  = dict(dataset_config.split_regions.items())[split_name]
        arrays  = cropper.load_split(region)
        spatial = (region.azimuth_size, region.range_size)

        grid = Patcher.build(
            spatial_size           = spatial,
            patch_size             = dataset_config.patch.size,
            stride                 = dataset_config.patch.stride,
            use_reflective_padding = dataset_config.patch.use_reflective_padding,
        )

        gt_parameters = arrays["parameters"]
        norm_stats    = NormalizationStats.load(run_directory / "meta", self.logger) if run_directory else None
        if norm_stats is not None and norm_stats.input_stats is None:
            norm_stats = None

        dataset = PatchDataset(
            inputs           = arrays["inputs"],
            gt_parameters    = gt_parameters,
            grid             = grid,
            input_config     = dataset_config.input_config,
            split_name       = split_name,
            logger           = self.logger,
            norm_stats       = norm_stats,
            x_axis           = x_axis,
            n_gaussians      = n_gaussians,
        )
        
        return dataset, grid, region, layout.global_crop

    def _build_model(self, model_name: str, in_channels: int, out_channels: int, image_size: int):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY)}")
        overrides = {"in_channels": in_channels, "out_channels": out_channels}

        if model_name in _IMAGE_SIZE_MODELS:
            overrides["image_size"] = image_size

        return get_model(model_name, **overrides)

    @staticmethod
    def _apply_ema(model, ema_state: dict, logger: Logger) -> int:
        import torch
        if not ema_state or not ema_state.get("enabled", False):
            logger.warning("EMA requested but checkpoint has no enabled EMA state. Falling back to raw weights.")
            return 0

        shadow  = ema_state.get("shadow", {})
        applied = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in shadow:
                    param.data.copy_(shadow[name].to(param.device, dtype=param.dtype))
                    applied += 1
        return applied

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
        self.logger.subsection(f"Run Directory : {self.run_directory}")

        run_summary     = self._read_json("run_summary.json")
        dataset_payload = self._read_json("dataset_creation_config.json")
        _crop_payload   = self._read_json("crop.json")
        _patch_payload  = self._read_json("patch.json")

        model_name    = str(run_summary["model_name"])
        in_channels   = int(run_summary["in_channels"])
        out_channels  = int(run_summary["out_channels"])
        x_axis_length = int(run_summary["x_axis_length"])

        n_gaussians    = out_channels // self.PARAMS_PER_GAUSSIAN
        has_noise_head = (out_channels % self.PARAMS_PER_GAUSSIAN) != 0

        if n_gaussians < 1:
            raise ValueError(f"out_channels={out_channels} is not consistent with at least 1 Gaussian (3 params each).")

        dataset_config = self._build_dataset_config(dataset_payload, batch_size=batch_size, num_workers=num_workers)

        image_size = dataset_config.patch.size[0]

        ckpt_path = self.run_directory / checkpoint_name

        self.logger.subsection(f"Checkpoint    : {ckpt_path}")

        norm_stats = NormalizationStats.load(self.run_directory / "meta", self.logger)
        if norm_stats is not None and norm_stats.input_stats is None:
            norm_stats = None
            print("[WARNING] Loaded normalization stats has no input_stats; ignoring normalization stats.")

        is_jax_ckpt = ckpt_path.suffix == ".pkl"
        used_ema    = False

        if is_jax_ckpt:
            self.logger.subsection("Backend : JAX")
            with open(ckpt_path, "rb") as _f:
                ckpt = pickle.load(_f)

            params      = ckpt["params"]
            batch_stats = ckpt.get("batch_stats")

            if use_ema and ckpt.get("ema_shadow") is not None:
                params   = ckpt["ema_shadow"]
                used_ema = True
                self.logger.subsection("EMA : using ema_shadow params")
            else:
                self.logger.subsection("EMA : not applied (using raw params)")

            from models.jax_models import get_jax_model
            
            jax_model = get_jax_model(
                model_name,
                in_channels  = in_channels,
                out_channels = out_channels,
            ) 
            
            model = ModelWrapper(
                apply_fn    = jax_model.apply,
                params      = params,
                batch_stats = batch_stats,
                params_per_gaussian      = self.PARAMS_PER_GAUSSIAN,
                normalizer               = Normalizer(norm_stats) if norm_stats is not None else None,
            )

            x_axis_raw = ckpt.get("x_axis", None)
            if x_axis_raw is None:
                x_axis_np = np.linspace(-20.0, 60.0, x_axis_length, dtype=np.float32)
                self.logger.warning("No x_axis in checkpoint; falling back to linspace(-20, 60, x_axis_length).")
            else:
                x_axis_np = np.asarray(x_axis_raw, dtype=np.float32).copy()

            ckpt_meta = {
                "epoch"         : int(ckpt.get("epoch", -1)),
                "best_val_loss" : float(ckpt.get("best_val_loss", float("nan"))),
                "best_epoch"    : int(ckpt.get("best_epoch", -1)),
                "best_metrics"  : dict(ckpt.get("best_metrics", {}) or {}),
            }

        else:
            import torch
            self.logger.subsection("Backend : PyTorch")
            model = self._build_model(model_name, in_channels=in_channels, out_channels=out_channels, image_size=image_size)
            model = model.to(device)

            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])

            if use_ema:
                n_applied = self._apply_ema(model, ckpt.get("ema_state_dict", {}) or {}, self.logger)
                used_ema  = n_applied > 0
                self.logger.subsection(f"EMA : applied to {n_applied} parameters")

            model.eval()

            x_axis_raw = ckpt.get("x_axis", None)
            if x_axis_raw is None:
                x_axis_np = np.linspace(-20.0, 60.0, x_axis_length, dtype=np.float32)
                self.logger.warning("No x_axis in checkpoint; falling back to linspace(-20, 60, x_axis_length).")
            else:
                t = x_axis_raw
                x_axis_np = t.cpu().numpy().astype(np.float32) if hasattr(t, "cpu") else np.asarray(t, dtype=np.float32)

            ckpt_meta = {
                "epoch"         : int(ckpt.get("epoch", -1)),
                "best_val_loss" : float(ckpt.get("best_val_loss", float("nan"))),
                "best_epoch"    : int(ckpt.get("best_epoch", -1)),
                "best_metrics"  : dict(ckpt.get("best_metrics", {}) or {}),
            }

        dataset, grid, region, global_crop = self._build_split_dataset(
            dataset_config,
            split,
            run_directory = self.run_directory,
            x_axis        = x_axis_np,
            n_gaussians   = n_gaussians,
        )

        from pipelines.dataset_creation_pipeline.load import _numpy_collate
        loader = DataLoader(
            dataset,
            batch_size  = dataset_config.batch_size,
            shuffle     = False,
            num_workers = dataset_config.num_workers,
            pin_memory  = False,
            multiprocessing_context = "spawn" if dataset_config.num_workers > 0 else None,
            drop_last   = False,
            collate_fn  = _numpy_collate,
        )

        self.logger.subsection(f"Model : '{model_name}' in_ch={in_channels} out_ch={out_channels} K_gauss={n_gaussians} noise_head={has_noise_head}")
        self.logger.subsection(f"Split : '{split}' patches={grid.grid.number_of_patches} az={region.azimuth_size} rg={region.range_size}\n")

        return LoadedRun(
            run_directory   = self.run_directory,
            model           = model,
            model_name      = model_name,
            in_channels     = in_channels,
            out_channels    = out_channels,
            x_axis          = x_axis_np,
            x_axis_length   = int(x_axis_np.size),
            n_gaussians     = n_gaussians,
            has_noise_head  = has_noise_head,
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
