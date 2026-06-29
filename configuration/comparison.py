from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List, Optional


@dataclass
class TrialComparisonConfig:
    runs_dir : Path      = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/backbone")
    run_tags : List[str] = field(default_factory=list)

    compare_images : bool          = True
    compare_gifs   : bool          = True
    embed_images   : bool          = False
    output_dir     : Optional[Path] = None


@dataclass
class PreprocessingComparisonConfig:
    runs_dir : Path      = Path("/ste/rnd/User/vice_vi/Dataset")
    run_tags : List[str] = field(default_factory=list)

    pixel_sample : int = 200000
    block_size   : int = 8
    range_chunk  : int = 512

    make_plots : bool          = True
    output_dir : Optional[Path] = None


@dataclass
class ParamExtractionComparisonConfig:
    params_dir : Path      = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/preprocessing/params")
    run_tags   : List[str] = field(default_factory=list)

    pixel_sample : int = 200000
    block_size   : int = 8
    range_chunk  : int = 512

    make_plots : bool          = True
    output_dir : Optional[Path] = None
