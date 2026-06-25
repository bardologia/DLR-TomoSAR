from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Optional


@dataclass
class ImageAeInferencePaths:
    figures_subdir   : str = "figures"
    logs_subdir      : str = "logs"
    metrics_filename : str = "metrics.json"
    report_filename  : str = "report.md"


@dataclass
class ImageAeInferenceConfig:
    run_directory   : Path
    output_subdir   : Optional[str] = None
    device          : str           = "cuda"
    seed            : int           = 0
    log_level       : str           = "INFO"

    split           : str           = "test"
    checkpoint_name : str           = "best_model.pt"
    batch_size      : Optional[int] = None
    num_workers     : int           = 4

    save_plots       : bool = True
    n_best_patches   : int  = 12
    n_worst_patches  : int  = 12
    n_random_patches : int  = 12
    n_scatter_points : int  = 20000
    patch_seed       : int  = 0

    fig_dpi : int = 150
    save_dpi: int = 300

    paths : ImageAeInferencePaths = field(default_factory=ImageAeInferencePaths)
