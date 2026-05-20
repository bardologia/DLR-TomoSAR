from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from configuration.param_extraction_config import ExtractionConfig
from tools.logger                          import Logger


class ExtractionMetadataManager:
    def __init__(self, config: ExtractionConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger

    def save_run_metadata(self, npy_path: Path) -> Path:
        meta_path = self.config.output_directory / "param_extraction_meta.json"
        ext       = self.config.fit_settings

        payload = {
            "timestamp"           : datetime.now().isoformat(timespec="seconds"),
            "processed_data_path" : str(self.config.processed_data_path),
            "source_tomogram"     : str(self.config.discover_tomogram_path()),
            "height_range"        : list(self.config.discover_height_range()),
            "output_directory"    : str(self.config.output_directory),
            "output_prefix"       : self.config.output_prefix,
            "output_suffix"       : self.config._output_suffix_value,
            "parameters_npy"      : npy_path.name,
            "number_of_gaussians" : ext.k_max,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, default=str)

        self.logger.subsection(f"-> Metadata written: {meta_path}")
        return meta_path
