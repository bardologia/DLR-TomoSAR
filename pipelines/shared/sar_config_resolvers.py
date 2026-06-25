from __future__ import annotations

import json
from pathlib import Path

from configuration.sar.gaussian_config import GaussianConfig


class GaussianConfigLoader:
    @staticmethod
    def from_dataset(dataset_dir: str | Path, n_gaussians: int, predict_presence: bool = False) -> GaussianConfig:
        meta_dir     = Path(dataset_dir) / "meta"
        cfg          = json.loads((meta_dir / "config_state.json").read_text())
        height_range = cfg["tomogram_config"]["height_range"]

        return GaussianConfig(
            n_default_gaussians = n_gaussians,
            x_min               = float(height_range[0]),
            x_max               = float(height_range[1]),
            predict_presence    = predict_presence,
        )
