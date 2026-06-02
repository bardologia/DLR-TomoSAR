from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tools.crop_region import CropRegion

__all__ = ["CropRegion"]


@dataclass
class TomogramConfiguration:
    fusar_project_path     : str                 = ""
    base_directory         : str                 = "/ste/rnd/"
    polarisation           : str                 = "hv"
    track_selection        : str                 = "*"
    height_range           : Tuple[float, float] = (-20.0, 80.0)
    filter_method          : str                 = "Boxcar"
    filter_arguments       : Dict                = field(default_factory=lambda: {"win": [20, 10]})
    beamforming_method     : str                 = "Capon"
    beamforming_arguments  : List                = field(default_factory=list)
    max_crop_azimuth_width : int                 = 1000
    apply_resampling       : bool                = False
    apply_presumming       : bool                = False
    max_amplitude_clip     : float               = 1.25


@dataclass
class ParallelConfiguration:
    tomogram_workers : Optional[int] = None
    pyrat_threads    : int           = 15

    @staticmethod
    def available_cores() -> int:
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:
            return os.cpu_count() or 1

    def resolve_workers(self, subsection_count: int) -> int:
        if self.tomogram_workers is not None:
            return max(1, min(subsection_count, self.tomogram_workers))

        cores  = self.available_cores()
        budget = max(1, cores // max(1, self.pyrat_threads))

        return max(1, min(subsection_count, budget))


@dataclass
class PathConfiguration:
    main_directory         : Path          = field(default_factory=lambda: Path("/ste/rnd/User/vice_vi/Dataset"))
    pyrat_directory        : Path          = field(default_factory=lambda: Path("/ste/rnd/User/vice_vi/pyrat"))
    data_subdirectory      : str           = "data"
    metadata_subdirectory  : str           = "meta"
    temporary_subdirectory : str           = "tmp"
    run_subdirectory       : Optional[str] = None

    @property
    def run_directory(self) -> Path:
        if self.run_subdirectory is None:
            return self.main_directory
        return self.main_directory / self.run_subdirectory

    @property
    def data_directory(self) -> Path:
        return self.run_directory / self.data_subdirectory

    @property
    def metadata_directory(self) -> Path:
        return self.run_directory / self.metadata_subdirectory

    @property
    def temporary_directory(self) -> Path:
        return self.run_directory / self.temporary_subdirectory


@dataclass
class ProcessingConfiguration:
    crop           : CropRegion
    input_configs  : TomogramConfiguration              = field(default_factory=TomogramConfiguration)
    output_configs : Optional[TomogramConfiguration]    = None
    parallel       : ParallelConfiguration              = field(default_factory=ParallelConfiguration)
    paths          : PathConfiguration                  = field(default_factory=PathConfiguration)

    dataset_type             : str = "FSAR"
    full_stack_identifier    : str = "flaca"
    reduced_stack_identifier : str = "flaca_2"
    tomogram_output_tag      : str = "Xtomo_id2X"
    parameter_output_tag     : str = "Xparams_id2X"

    @property
    def tomogram_tag(self) -> str:
        return f"{self.crop.as_identifier_string()}_{self.reduced_stack_identifier}_{self.tomogram_output_tag}"

    @property
    def parameter_tag(self) -> str:
        return f"{self.crop.as_identifier_string()}_{self.full_stack_identifier}_{self.parameter_output_tag}"

    @property
    def output_config(self) -> TomogramConfiguration:
        return self.output_configs if self.output_configs is not None else self.input_configs

    @property
    def has_split_configs(self) -> bool:
        return self.output_configs is not None

    def __post_init__(self) -> None:
        if self.paths.run_subdirectory is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.paths.run_subdirectory = f"run_{self.tomogram_tag}_{timestamp}"
