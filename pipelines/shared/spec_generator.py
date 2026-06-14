from __future__ import annotations

from pathlib import Path

from configuration.sar.processing_config import PathConfiguration, TomogramConfiguration

from tools                   import FileIO
from tools.monitoring.logger import Logger


class GeneratorBase:
    def __init__(self, spec: dict, logger: Logger) -> None:
        self.spec   = spec
        self.logger = logger

    @classmethod
    def from_spec_file(cls, spec_path: str | Path, logger: Logger) -> "GeneratorBase":
        return cls(FileIO.load_json(Path(spec_path)), logger)

    def _paths(self) -> PathConfiguration:
        return PathConfiguration(
            main_directory   = Path(self.spec["main_directory"]),
            pyrat_directory  = Path(self.spec["pyrat_directory"]),
            run_subdirectory = self.spec["run_subdirectory"],
        )

    def _tomogram_config(self) -> TomogramConfiguration:
        return TomogramConfiguration(**self.spec["tomogram_config"])
