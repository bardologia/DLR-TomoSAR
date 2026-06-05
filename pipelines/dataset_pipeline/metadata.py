from __future__ import annotations

import json
from dataclasses import asdict
from pathlib     import Path

from configuration.dataset_config            import DatasetConfiguration
from configuration.processing_config         import CropRegion
from pipelines.dataset_pipeline.patch        import GridInfo
from tools.logger                            import Logger


class MetadataWriter:
    def __init__(self, run_directory: Path, logger: Logger) -> None:
        self.run_directory      = Path(run_directory)
        self.logger             = logger
        self.metadata_directory = self.run_directory / "meta"
        
        self.outpaths           = {
            "dataset_configuration" : self.metadata_directory / "dataset_creation_config.json",
            "crop"                  : self.metadata_directory / "crop.json",
            "patch"                 : self.metadata_directory / "patch.json",
        }
        
        self.metadata_directory.mkdir(parents=True, exist_ok=True)

        self.logger.section("[MetadataWriter Initialized]")
        self.logger.subsection(f"Metadata Directory : {self.metadata_directory} \n")
        
    def save_dataset_configuration(self, config: DatasetConfiguration) -> Path:
        out_path = self.outpaths["dataset_configuration"]
        payload  = asdict(config)
        
        payload["preprocessing_run_directory"] = str(config.preprocessing_run_directory)
        payload["input_config"]                = config.input_config.as_dict()
        payload["output_config"]               = config.output_config.as_dict()
        payload.pop("x_axis", None)   
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, default=str)
        
        return out_path

    def save_crop_metadata(self, global_crop: CropRegion, splits: dict[str, CropRegion]) -> Path:
        out_path = self.outpaths["crop"]
        payload  = {"global_crop" : list(global_crop.as_tuple()), "splits" : {name: self._region_payload(value) for name, value in splits.items()}}

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

        return out_path

    def save_patch_metadata(self, grids: dict[str, GridInfo]) -> Path:
        out_path = self.outpaths["patch"]
        payload  = {name: self._grid_payload(value) for name, value in grids.items()}

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

        return out_path

    def _region_payload(self, value):
        if isinstance(value, (list, tuple)):
            return [list(region.as_tuple()) for region in value]
        return list(value.as_tuple())

    def _grid_payload(self, value):
        if isinstance(value, (list, tuple)):
            return [grid.as_dict() for grid in value]
        return value.as_dict()
