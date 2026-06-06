from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union


class FitMode:
    @dataclass
    class SigmaOnly:
        threshold_factor : float = 0.25
        truncation_index : int   = 170
        k_max            : int   = 5
        lambda_k         : float = 3e-3
        prominence_frac  : float = 0.05


FitConfig = FitMode.SigmaOnly


@dataclass
class FitSettings:
    max_fit_iterations  : int       = 5000
    fit_config          : FitConfig = field(default_factory=FitMode.SigmaOnly)

    @property
    def number_of_gaussians(self) -> int:
        return self.fit_config.k_max

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
    tomogram_filename    : Optional[str]                 = None
    height_range         : Optional[Tuple[float, float]] = None
    fit_settings         : FitSettings                   = field(default_factory=FitSettings)
    parameter_workers    : int                           = 20

    use_gpu              : bool                          = True
    gpu_batch_size       : int                           = 256
    gpu_pixel_batch_size : int                           = 24576
    adam_steps           : int                           = 3000
    adam_lr              : float                         = 2e-1
    adam_b1              : float                         = 0.95
    adam_b2              : float                         = 0.999
    gpu_device_ids       : Optional[List[int]]           = field(default_factory=lambda: [0, 1, 3])


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
        fs = self.fit_settings
        method = fs.fitting_method
        if method.startswith("sigma_only"):
            cfg = self.fit_settings.fit_config
            k_max = getattr(cfg, "k_max", self.fit_settings.number_of_gaussians)
            method_short = f"sigonly_k{k_max}"
        else:
            method_short = "filt"
        return f"Ng{fs.number_of_gaussians}_{method_short}"

    @property
    def output_subdir_name(self) -> str:
        return f"{self.output_prefix}_{self.output_suffix_value}"

    @property
    def output_directory(self) -> Path:
        return Path(self.processed_data_path) / "params" / self.output_subdir_name

    @property
    def parameters_npy_path(self) -> Path:
        return self.output_directory / f"parameters_{self.output_suffix_value}.npy"

    @property
    def diagnostics_npz_path(self) -> Path:
        return self.output_directory / f"fit_diagnostics_{self.output_suffix_value}.npz"

    def discover_tomogram_path(self) -> Path:
        data_dir = self.data_directory

        if self.tomogram_filename:
            candidate = data_dir / self.tomogram_filename
            if candidate.is_file():
                return candidate

        layout_path = data_dir / "dataset.json"
        if layout_path.is_file():
            with open(layout_path, "r", encoding="utf-8") as f:
                layout = json.load(f)
            fname = layout.get("artifacts", {}).get("tomogram_full")
            if fname:
                candidate = data_dir / fname
                if candidate.is_file():
                    return candidate

        params_matches = sorted(data_dir.glob("tomogram_full_*params*.npy"))
        if params_matches:
            return params_matches[0]

        any_matches = sorted(data_dir.glob("tomogram_full_*.npy"))
        if any_matches:
            return any_matches[0]

    def discover_height_range(self) -> Tuple[float, float]:
        if self.height_range is not None:
            return tuple(self.height_range)

        meta_dir = self.metadata_directory
        if meta_dir.is_dir():
            for state_path in sorted(meta_dir.glob("config_state_*.json")):
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)

                for key in ("output_configs", "input_configs"):
                    cfg = state.get(key)
                    if isinstance(cfg, dict) and cfg.get("height_range"):
                        hr = cfg["height_range"]
                        if isinstance(hr, (list, tuple)) and len(hr) == 2:
                            return (float(hr[0]), float(hr[1]))

        

@dataclass
class ExtractParamsEntryConfig:
    dataset_base_path : Path         = Path("/ste/rnd/User/vice_vi/Dataset")
    dataset_filter    : list         = field(default_factory=list)
    pyrat_directory   : Path         = Path("/ste/rnd/User/vice_vi/pyrat")
    tomogram_filename : str          = "tomogram_full_1000a16000a500a4000_1_Xtomo_id2X.npy"

    output_prefix     : str          = "params"
    output_suffix     : str | None   = None
    height_range      : tuple | None = None

    fit_k_max         : int          = 5
    fit_lambda_k      : float        = 3e-3
    parameter_workers : int          = 50
