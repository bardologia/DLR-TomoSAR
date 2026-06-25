from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path


@dataclass
class RunPathsConfig:
    dataset_path     : Path  = Path("/ste/rnd/User/vice_vi/Dataset/base_dataset_w20_10")
    parameters_path  : Path  = Path("/ste/rnd/User/vice_vi/Dataset/base_dataset_w20_10/params/params_Ng3_sigonly_k5/parameters_Ng3_sigonly_k5.npy")
    log_base_dir     : Path  = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/benchmark_extended")
    secondary_labels : tuple = ("FL01_PS04", "FL01_PS06", "FL01_PS08", "FL01_PS26")


@dataclass
class TrainingQueueConfig:
    epochs               : int             = 100
    scheduler_epochs     : int | None      = 100
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
