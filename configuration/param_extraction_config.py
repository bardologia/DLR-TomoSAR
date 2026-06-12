from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union


class FitMode:
    @dataclass
    class SigmaOnly:
        threshold_factor   : float = 0.25
        truncation_index   : int   = 170
        k_max              : int   = 5
        lambda_k           : float = 1e-2
        prominence_frac    : float = 0.05
        sigma_init_divisor : float = 4.0
        activity_threshold : float = 5e-3


FitConfig = FitMode.SigmaOnly


@dataclass
class FitSettings:
    max_fit_iterations  : int       = 5000
    fit_config          : FitConfig = field(default_factory=FitMode.SigmaOnly)

    @property
    def parameters_per_profile(self) -> int:
        return 3 * self.fit_config.k_max

    @property
    def fitting_method(self) -> str:
        return "sigma_only_adam"


@dataclass
class ExtractionConfig:
    processed_data_path  : Path
    pyrat_directory      : Path                          = field(default_factory=lambda: Path("/ste/rnd/User/vice_vi/pyrat"))
    output_prefix        : str                           = "params"
    output_suffix        : Optional[str]                 = None
    height_range         : Optional[Tuple[float, float]] = None
    fit_settings         : FitSettings                   = field(default_factory=FitSettings)
    parameter_workers    : int                           = 20

    range_batch_size     : int                           = 3500
    gpu_pixel_batch_size : int                           = 24576
    adam_steps           : int                           = 3000
    adam_lr              : float                         = 2e-1
    adam_b1              : float                         = 0.95
    adam_b2              : float                         = 0.999
    gpu_device_ids       : Optional[List[int]]           = field(default_factory=lambda: [0, 1, 2, 3])


    @property
    def data_directory(self) -> Path:
        return Path(self.processed_data_path) / "data"

    @property
    def metadata_directory(self) -> Path:
        return Path(self.processed_data_path) / "meta"

    @property
    def output_suffix_value(self) -> str:
        if self.output_suffix:
            return self.output_suffix

        cfg      = self.fit_settings.fit_config
        k_max    = cfg.k_max
        divisor  = cfg.sigma_init_divisor
        lambda_k = cfg.lambda_k
        div_tag  = f"{divisor:g}".replace(".", "p")
        lam_tag  = f"{lambda_k:g}".replace(".", "p").replace("-", "m")
        return f"sigmaonly_k{k_max}_sig{div_tag}_lam{lam_tag}"

    @property
    def output_subdir_name(self) -> str:
        return f"{self.output_prefix}_{self.output_suffix_value}"

    @property
    def output_directory(self) -> Path:
        return Path(self.processed_data_path) / "params" / self.output_subdir_name

    @property
    def parameters_npy_path(self) -> Path:
        return self.output_directory / "parameters.npy"

    @property
    def diagnostics_npz_path(self) -> Path:
        return self.output_directory / "fit_diagnostics.npz"

    def discover_tomogram_path(self) -> Path:
        tomogram_path = self.data_directory / "tomogram_full.npy"

        if not tomogram_path.is_file():
            raise FileNotFoundError(f"No tomogram_full.npy in {self.data_directory}")

        return tomogram_path

    def discover_height_range(self) -> Tuple[float, float]:
        if self.height_range is not None:
            return tuple(self.height_range)

        meta_dir   = self.metadata_directory
        state_path = meta_dir / "config_state.json"
        if state_path.is_file():
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            hr = state["tomogram_config"]["height_range"]
            if isinstance(hr, (list, tuple)) and len(hr) == 2:
                return (float(hr[0]), float(hr[1]))

        raise FileNotFoundError(f"No height_range override given and no config_state.json with a tomogram height_range under {meta_dir}")


@dataclass
class ExtractParamsEntryConfig:
    dataset_base_path : Path         = Path("/ste/rnd/User/vice_vi/Dataset")
    dataset_filter    : list         = field(default_factory=list)
    pyrat_directory   : Path         = Path("/ste/rnd/User/vice_vi/pyrat")

    output_prefix     : str          = "params"
    output_suffix     : str | None   = None
    height_range      : tuple | None = None

    fit_k_max              : int    = 5
    fit_lambda_k           : float  = 1e-2
    fit_sigma_init_divisor : float  = 4.0

    gpu_device_ids    : list         = field(default_factory=lambda: [0, 1, 2, 3])
    range_batch_size  : int          = 3500
    parameter_workers : int          = 100
