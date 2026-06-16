from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.benchmark.jepa import JepaBenchConfig


@dataclass
class BenchmarkPathsConfig:
    dataset_path     : Path  = Path("/ste/rnd/User/vice_vi/Dataset/base_dataset_w20_10")
    parameters_path  : Path  = Path("/ste/rnd/User/vice_vi/Dataset/base_dataset_w20_10/params/params_Ng3_sigonly_k5/parameters_Ng3_sigonly_k5.npy")
    log_base_dir     : Path  = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/benchmark_extended")
    secondary_labels : tuple = ("FL01_PS04", "FL01_PS06", "FL01_PS08", "FL01_PS26")


@dataclass
class OverfitGateConfig:
    max_steps           : int   = 30000
    stop_threshold      : float = 1e-3
    batch_size          : int   = 9
    azimuth_start       : int   = 1000
    azimuth_lines       : int   = 128
    range_lines         : int   = 128
    seed                : int   = 42
    require_convergence : bool  = True
    abort_on_fail       : bool  = True


@dataclass
class MaxBatchConfig:
    vram_budget_gb : float = 40.0
    max_batch      : int   = 512
    measure_steps  : int   = 3
    seed           : int   = 42


@dataclass
class SizeMatchConfig:
    reference_model : str         = "unet"
    tolerance       : float       = 0.05
    max_iterations  : int         = 100
    scale_low       : float       = 0.05
    scale_high      : float       = 8.0
    in_channels     : int         = 9
    locked_params   : tuple[str, ...] = ("embedding_dim", "embedding_dims")


@dataclass
class TrainingQueueConfig:
    epochs               : int             = 200
    scheduler_epochs     : int | None      = 200
    validation_frequency : int             = 1
    batch_size           : int             = 256
    num_workers          : int             = 4
    prefetch_factor      : int             = 2
    warmup_steps         : int             = 200
    eta_min              : float           = 1e-6
    early_stop_patience  : int             = 30
    patch_size           : tuple[int, int] = (64, 64)
    patch_stride         : int             = 32
    train_azimuth        : tuple[int, int] = (1000, 13000)
    val_azimuth          : tuple[int, int] = (13000, 14500)
    test_azimuth         : tuple[int, int] = (14500, 16000)
    log_all_losses       : bool            = False

    use_amp                     : bool = False
    gradient_accumulation_steps : int  = 1

    scale_lr_with_batch     : bool = True
    lr_reference_batch_size : int  = 256


@dataclass
class InferenceQueueConfig:
    split           : str        = "test"
    checkpoint_name : str        = "best_model.pt"
    batch_size      : int | None = None
    num_workers     : int        = 4
    cpu_workers     : int        = 16

    save_plots      : bool = False
    save_animations : bool = False
    save_cubes      : bool = False
    stitch_window   : str  = "hann"

    n_best_profiles   : int = 12
    n_worst_profiles  : int = 12
    n_random_profiles : int = 12

    n_range_slices     : int = 5
    n_azimuth_slices   : int = 5
    n_elevation_slices : int = 5

    gif_axes       : list[str] = field(default_factory=lambda: ["elevation"])
    gif_fps        : int       = 12
    gif_max_frames : int       = 150


@dataclass
class ComparisonReportConfig:
    embed_images : bool = False


def _default_ae_loss():
    from configuration.training.profile_autoencoder import ProfileAeLossConfig

    return ProfileAeLossConfig()


@dataclass
class BenchmarkConfig:
    training_type : str = "backbone"

    paths      : BenchmarkPathsConfig   = field(default_factory=BenchmarkPathsConfig)
    overfit    : OverfitGateConfig      = field(default_factory=OverfitGateConfig)
    max_batch  : MaxBatchConfig         = field(default_factory=MaxBatchConfig)
    size_match : SizeMatchConfig        = field(default_factory=SizeMatchConfig)
    training   : TrainingQueueConfig    = field(default_factory=TrainingQueueConfig)
    inference  : InferenceQueueConfig   = field(default_factory=InferenceQueueConfig)
    comparison : ComparisonReportConfig = field(default_factory=ComparisonReportConfig)

    ae_loss         : object          = field(default_factory=_default_ae_loss)
    jepa            : JepaBenchConfig  = field(default_factory=JepaBenchConfig)
    pixel_subsample : float           = 1.0
    keep_empty_frac : float           = 0.05

    gpus            : list[int]  = field(default_factory=lambda: [2, 3])
    skip_models     : list[str]  = field(default_factory=list)
    run_tag         : str | None = None
    resume          : bool       = True
    seed            : int        = 0
    n_gaussians     : int        = 5
    poll_interval_s : float      = 5.0

    def runs_size_match(self) -> bool:
        return self.training_type == "backbone"

    def runs_max_batch(self) -> bool:
        return self.training_type == "backbone"

    def runs_inference(self) -> bool:
        return self.training_type in ("backbone", "jepa")
