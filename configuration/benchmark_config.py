from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchmarkPathsConfig:
    dataset_path    : Path = Path("/ste/rnd/User/vice_vi/Dataset/clean_dataset")
    parameters_path : Path = Path("/ste/rnd/User/vice_vi/Dataset/clean_dataset/params/params_sig_k5/parameters_sig_k5.npy")
    log_base_dir    : Path = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/logs/benchmark")


@dataclass
class OverfitGateConfig:
    max_steps           : int   = 5000
    stop_threshold      : float = 1e-3
    batch_size          : int   = 9
    azimuth_start       : int   = 1000
    azimuth_lines       : int   = 128
    range_lines         : int   = 128
    seed                : int   = 42
    require_convergence : bool  = True
    abort_on_fail       : bool  = True


@dataclass
class SizeMatchConfig:
    reference_model : str   = "unet"
    tolerance       : float = 0.005
    max_iterations  : int   = 40
    scale_low       : float = 0.05
    scale_high      : float = 8.0
    in_channels     : int   = 11


@dataclass
class TrainingQueueConfig:
    epochs               : int             = 200
    scheduler_epochs     : int | None      = None
    validation_frequency : int             = 1
    batch_size           : int             = 256
    num_workers          : int             = 4
    warmup_steps         : int             = 200
    eta_min              : float           = 1e-6
    early_stop_patience  : int             = 30
    early_stop_min_delta : float           = 1e-4
    patch_size           : tuple[int, int] = (64, 64)
    patch_stride         : int             = 32
    train_azimuth        : tuple[int, int] = (1000, 9120)
    val_azimuth          : tuple[int, int] = (9120, 12400)
    test_azimuth         : tuple[int, int] = (12400, 16000)
    log_all_losses       : bool            = False


@dataclass
class InferenceQueueConfig:
    split              : str        = "test"
    use_ema            : bool       = True
    checkpoint_name    : str        = "best_model.pt"
    batch_size         : int | None = None
    num_workers        : int        = 4
    cpu_workers        : int        = 16

    save_cubes         : bool       = False
    stitch_window      : str        = "hann"

    n_best_profiles    : int        = 12
    n_worst_profiles   : int        = 12
    n_random_profiles  : int        = 12

    n_range_slices     : int        = 5
    n_azimuth_slices   : int        = 5
    n_elevation_slices : int        = 5

    gif_axes           : list[str]  = field(default_factory=lambda: ["elevation"])
    gif_fps            : int        = 12
    gif_max_frames     : int        = 150


@dataclass
class ComparisonReportConfig:
    embed_images : bool = False


@dataclass
class BenchmarkConfig:
    paths      : BenchmarkPathsConfig   = field(default_factory=BenchmarkPathsConfig)
    overfit    : OverfitGateConfig      = field(default_factory=OverfitGateConfig)
    size_match : SizeMatchConfig        = field(default_factory=SizeMatchConfig)
    training   : TrainingQueueConfig    = field(default_factory=TrainingQueueConfig)
    inference  : InferenceQueueConfig   = field(default_factory=InferenceQueueConfig)
    comparison : ComparisonReportConfig = field(default_factory=ComparisonReportConfig)

    gpus            : list[int]  = field(default_factory=lambda: [0, 1, 2, 3])
    skip_models     : list[str]  = field(default_factory=list)
    run_tag         : str | None = None
    resume          : bool       = True
    seed            : int        = 0
    n_gaussians     : int        = 5
    poll_interval_s : float      = 5.0


@dataclass
class OverfitTestConfig:
    gpu         : int        = 0
    run_tag     : str | None = None
    n_gaussians : int        = 5
    skip_models : list[str]  = field(default_factory=list)

    paths   : BenchmarkPathsConfig = field(default_factory=BenchmarkPathsConfig)
    overfit : OverfitGateConfig    = field(default_factory=OverfitGateConfig)
