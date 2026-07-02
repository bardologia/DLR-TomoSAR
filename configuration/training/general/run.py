from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path

from configuration.training.general.optimization import ClipMode, SchedulerType, WarmupMode


@dataclass
class RunPathsConfig:
    dataset_path     : Path  = Path("/ste/rnd/User/vice_vi/Dataset/base_dataset_w20_10")
    parameters_path  : Path  = Path("/ste/rnd/User/vice_vi/Dataset/base_dataset_w20_10/params/params_Ng3_sigonly_k5/parameters_Ng3_sigonly_k5.npy")
    log_base_dir     : Path  = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/benchmark_extended")
    secondary_labels : tuple = ("FL01_PS04", "FL01_PS06", "FL01_PS08", "FL01_PS26")


@dataclass
class TrainingQueueConfig:
    epochs               : int             = 60
    scheduler_epochs     : int | None      = None
    validation_frequency : int             = 1
    batch_size           : int             = 256
    num_workers          : int             = 4
    prefetch_factor      : int             = 2
    eta_min              : float           = 1e-6
    early_stop_patience  : int             = 30
    log_all_losses       : bool            = False

    use_amp                     : bool = False
    gradient_accumulation_steps : int  = 1
    abort_on_nonfinite_loss     : bool = True

    scale_lr_with_batch     : bool = True
    lr_reference_batch_size : int  = 256

    scheduler_type      : SchedulerType = SchedulerType.COSINE_ANNEALING
    scheduler_step_size : int           = 30
    scheduler_gamma     : float         = 0.1
    scheduler_power     : float         = 1.0

    warmup_enabled    : bool       = True
    warmup_steps      : int        = 200
    warmup_mode       : WarmupMode = WarmupMode.LINEAR
    warmup_poly_power : float      = 2.0

    clip_mode                : ClipMode = ClipMode.FIXED
    max_grad_norm            : float    = 1.0
    clip_adaptive_window     : int      = 200
    clip_adaptive_percentile : float    = 95.0
    clip_adaptive_mean_std_k : float    = 2.0

    patch_size           : tuple[int, int] = (64, 64)
    patch_stride         : int             = 32
    train_azimuth        : tuple[int, int] = (1000, 13000)
    val_azimuth          : tuple[int, int] = (13000, 14500)
    test_azimuth         : tuple[int, int] = (14500, 16000)
