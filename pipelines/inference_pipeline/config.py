from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List, Optional


@dataclass
class InferenceConfig:
    run_directory      : Path
    output_subdir      : Optional[str]  = None
    device             : str            = "cuda"

    use_ema            : bool          = True
    checkpoint_name    : str           = "best_model.pt"

    split              : str           = "test"
    batch_size         : Optional[int] = None
    num_workers        : int           = 4

    stitch_window      : str           = "hann"
    cube_dtype         : str           = "float32"
    save_cubes         : bool          = True

    n_best_profiles    : int           = 12
    n_worst_profiles   : int           = 12
    n_random_profiles  : int           = 12
    profile_seed       : int           = 0

    n_range_slices     : int           = 5
    n_azimuth_slices   : int           = 5
    n_elevation_slices : int           = 5

    gif_axes           : List[str]     = field(default_factory=lambda: ["elevation"])
    gif_fps            : int           = 12
    gif_max_frames     : int           = 150
    gif_dpi            : int           = 110

    cmap_intensity     : str           = "viridis"
    cmap_error         : str           = "magma"
    fig_dpi            : int           = 150
    save_dpi           : int           = 300

    seed               : int           = 0
    log_level          : str           = "INFO"

    def __post_init__(self) -> None:
        self.run_directory = Path(self.run_directory)
        if not self.run_directory.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_directory}")
        if self.split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of train/val/test, got {self.split!r}")
        if self.stitch_window not in ("hann", "triangular", "uniform"):
            raise ValueError(f"stitch_window must be hann|triangular|uniform, got {self.stitch_window!r}")
        for axis in self.gif_axes:
            if axis not in ("elevation", "range", "azimuth"):
                raise ValueError(f"Each gif_axis must be elevation|range|azimuth, got {axis!r}")
