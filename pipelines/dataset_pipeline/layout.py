from __future__ import annotations

import json
from pathlib import Path

from configuration.processing_config import CropRegion
from tools.logger                    import Logger


class Layout:
    def __init__(self, run_directory: Path, logger: Logger, parameters_path: Path) -> None:
        self.run_directory    = Path(run_directory)
        self.logger           = logger
        self.data_directory   = self.run_directory / "data"
        self.parameters_path  = Path(parameters_path)

        layout_path = self.data_directory / "dataset.json"
        with open(layout_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.global_crop    : CropRegion = CropRegion(*payload["global_crop"])
        self.dataset_type   : str        = payload["dataset_type"]
        self.tomogram_tag   : str        = payload["tomogram_tag"]
        self.parameter_tag  : str        = payload["parameter_tag"]
        self.artifacts      : dict       = payload["artifacts"]

        self.logger.section("[Layout Loaded]")
        self.logger.kv_table({
            "Run Directory":   self.run_directory,
            "Global Crop":     self.global_crop.as_tuple(),
            "Azimuth (lines)": self.global_crop.azimuth_size,
            "Range (samples)": self.global_crop.range_size,
            "Tomogram Tag":    self.tomogram_tag,
            "Parameters":      self.parameters_path,
        })

    def artifact_path(self, artifact_key: str) -> Path:
        if artifact_key == "parameters":
            return self.parameters_path

        return self.data_directory / self.artifacts[artifact_key]
