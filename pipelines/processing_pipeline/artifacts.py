from __future__ import annotations

from pathlib import Path
from typing import Literal

from configuration.processing_config import ProcessingConfiguration
from tools.logger                    import Logger


ArtifactType = Literal["tomogram_full", "tomogram_reduced", "dem_full", "dem_reduced", "primary_reduced", "secondaries_reduced", "interferograms_reduced"]


class ArtifactRegistry:
    def __init__(self, config: ProcessingConfiguration, logger: Logger) -> None:
        self.config = config
        self.logger = logger

    def ensure_directory_structure(self) -> None:
        paths = self.config.paths
        self.logger.section("[Directory Validation]")
        for directory in (paths.data_directory, paths.metadata_directory, paths.temporary_directory):
            target_dir = directory.resolve()
            target_dir.mkdir(parents=True, exist_ok=True)
            self.logger.subsection(f"Ensured path : {target_dir}")

    def _artifact_filenames(self) -> dict[str, str]:
        tomo_tag  = self.config.tomogram_tag
        param_tag = self.config.parameter_tag
        
        return {
            "tomogram_full"          : f"tomogram_full_{param_tag}.npy",
            "tomogram_reduced"       : f"tomogram_reduced_{tomo_tag}.npy",
            "dem_full"               : f"dem_full_{param_tag}.npy",
            "dem_reduced"            : f"dem_reduced_{tomo_tag}.npy",
            "primary_reduced"        : f"primary_reduced_{tomo_tag}.npy",
            "secondaries_reduced"    : f"secondaries_reduced_{tomo_tag}.npy",
            "interferograms_reduced" : f"interferograms_reduced_{tomo_tag}.npy",
        }

    def artifact_path(self, artifact_type: ArtifactType) -> Path:
        filenames = self._artifact_filenames()
        return self.config.paths.data_directory / filenames[artifact_type]

    def existence_map(self) -> dict[str, bool]:
        self.ensure_directory_structure()
        filenames = self._artifact_filenames()
        existence = {key: self.artifact_path(key).exists() for key in filenames}

        self.logger.section("[Pipeline State Check]")
        for key, exists in existence.items():
            path = self.artifact_path(key)
            self.logger.subsection(f"{key:<16} ({path.name}) : {'FOUND' if exists else 'MISSING'}")
        
        return existence
