from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List, Optional

from configuration.benchmark.general  import BenchmarkPathsConfig, TrainingQueueConfig
from configuration.sar.geometry_config import GeometryConfig


@dataclass
class ProfileAeInferencePaths:
    figures_subdir   : str = "figures"
    logs_subdir      : str = "logs"
    metrics_filename : str = "metrics.json"
    report_filename  : str = "report.md"


@dataclass
class ProfileAeInferenceConfig:
    run_directory   : Path
    output_subdir   : Optional[str] = None
    device          : str           = "cuda"
    seed            : int           = 0
    log_level       : str           = "INFO"

    split           : str           = "test"
    checkpoint_name : str           = "best_model.pt"
    batch_size      : Optional[int] = 4096
    num_workers     : int           = 4

    pixel_subsample : float = 1.0
    keep_empty_frac : float = 0.05

    save_plots        : bool = True
    n_best_curves     : int  = 12
    n_worst_curves    : int  = 12
    n_random_curves   : int  = 12
    n_scatter_points  : int  = 20000
    curve_seed        : int  = 0

    fig_dpi : int = 150
    save_dpi: int = 300

    paths : ProfileAeInferencePaths = field(default_factory=ProfileAeInferencePaths)


@dataclass
class ProfileAeInferenceEntryConfig:
    logs_dir        : Path      = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/profile_autoencoder")
    run_filter      : List[str] = field(default_factory=list)
    gpus            : List[int] = field(default_factory=lambda: [0])
    poll_interval_s : float     = 5.0

    n_gaussians : int = 5

    paths    : BenchmarkPathsConfig = field(default_factory=BenchmarkPathsConfig)
    geometry : GeometryConfig       = field(default_factory=GeometryConfig)
    training : TrainingQueueConfig  = field(default_factory=TrainingQueueConfig)

    inference : ProfileAeInferenceConfig = field(default_factory=lambda: ProfileAeInferenceConfig(run_directory=Path(".")))
