from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib     import Path


@dataclass
class GaussianConfig:
    n_default_gaussians : int
    x_min               : float
    x_max               : float
    amp_max             : float = 1000
    params_per_gaussian : int   = 3
    clamp_leaky_slope   : float = 0.01

    @classmethod
    def from_dataset(cls, dataset_dir: str | Path, n_gaussians: int) -> "GaussianConfig":
        meta_dir     = Path(dataset_dir) / "meta"
        cfg          = json.loads((meta_dir / "config_state.json").read_text())
        height_range = cfg["tomogram_config"]["height_range"]

        return cls(
            n_default_gaussians = n_gaussians,
            x_min               = float(height_range[0]),
            x_max               = float(height_range[1]),
        )
