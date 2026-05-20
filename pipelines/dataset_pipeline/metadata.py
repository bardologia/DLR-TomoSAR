from __future__ import annotations

import json
from dataclasses import asdict
from pathlib     import Path

from configuration.dataset_config            import DatasetConfiguration
from configuration.processing_config         import CropRegion
from pipelines.dataset_pipeline.patch        import GridInfo
from tools.logger                            import Logger


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
        self.logger.subsection(f"Run Directory  : {self.run_directory}")
        self.logger.subsection(f"Global Crop    : {self.global_crop.as_tuple()}")
        self.logger.subsection(f"Tomogram Tag   : {self.tomogram_tag}")
        self.logger.subsection(f"Parameters     : {self.parameters_path} \n")

    def artifact_path(self, artifact_key: str) -> Path:
        if artifact_key == "parameters":
            return self.parameters_path

        return self.data_directory / self.artifacts[artifact_key]


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
        payload  = {"global_crop" : list(global_crop.as_tuple()), "splits"      : {name: list(region.as_tuple()) for name, region in splits.items()}}
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        
        return out_path

    def save_patch_metadata(self, grids: dict[str, GridInfo]) -> Path:
        out_path = self.outpaths["patch"]
        payload  = {name: grid.as_dict() for name, grid in grids.items()}
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        
        return out_path
