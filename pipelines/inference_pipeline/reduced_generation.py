from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path
from typing      import List

from configuration.processing_config        import (
    ParallelConfiguration,
    PathConfiguration,
    ProcessingConfiguration,
    TomogramConfiguration,
)
from pipelines.processing_pipeline.tomogram import TomogramProcessor
from pipelines.shared                       import FileIO
from tools.logger                           import Logger
from tools.regions                          import CropRegion


@dataclass
class ReducedTomogramSpec:
    tomogram_config  : dict
    stack_identifier : str
    dataset_type     : str
    pyrat_directory  : str
    main_directory   : str
    effort           : str
    crop             : List[int]
    tomogram_path    : str
    dem_path         : str


class ReducedTomogramGenerator:
    def __init__(self, spec: ReducedTomogramSpec, logger: Logger) -> None:
        self.spec   = spec
        self.logger = logger

    @classmethod
    def from_spec_file(cls, spec_path: str | Path, logger: Logger) -> "ReducedTomogramGenerator":
        payload = FileIO.load_json(Path(spec_path))
        return cls(ReducedTomogramSpec(**payload), logger)

    def _build_config(self) -> ProcessingConfiguration:
        tomogram_config = TomogramConfiguration(**self.spec.tomogram_config)

        paths = PathConfiguration(
            main_directory   = Path(self.spec.main_directory),
            pyrat_directory  = Path(self.spec.pyrat_directory),
            run_subdirectory = "reduced_work",
        )

        return ProcessingConfiguration(
            crop             = CropRegion(*self.spec.crop),
            tomogram_config  = tomogram_config,
            parallel         = ParallelConfiguration(effort=self.spec.effort),
            paths            = paths,
            dataset_type     = self.spec.dataset_type,
            stack_identifier = self.spec.stack_identifier,
        )

    def run(self) -> None:
        config = self._build_config()

        self.logger.section("[Reduced Tomogram Generation]")
        self.logger.kv_table({
            "Stack id"       : config.stack_identifier,
            "Track selection": config.tomogram_config.track_selection,
            "Crop"           : config.crop.as_tuple(),
            "Output"         : self.spec.tomogram_path,
        })

        TomogramProcessor(config, logger=self.logger).run(
            tomogram_path    = Path(self.spec.tomogram_path),
            dem_path         = Path(self.spec.dem_path),
            stack_identifier = config.stack_identifier,
            tomogram_config  = config.tomogram_config,
        )
