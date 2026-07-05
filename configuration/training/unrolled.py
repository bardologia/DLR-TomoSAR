from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.dataset                import AugmentationConfig
from configuration.normalization.general  import NormalizationConfig
from configuration.training.general.run   import TrainingPathsConfig
from configuration.sar.geometry_config    import GeometryConfig


@dataclass
class UnrolledTrainingConfig:
    epochs               : int        = 60
    scheduler_epochs     : int | None = None
    eta_min              : float      = 1e-6
    early_stop_patience  : int        = 30
    early_stop_min_delta : float      = 0.0
    max_grad_norm        : float      = 1.0

    warmup_enabled : bool = True
    warmup_steps   : int  = 200

    use_ema   : bool  = False
    ema_decay : float = 0.999

    reserve_vram      : bool  = False
    vram_keep_free_gb : float = 1.0

    batch_size      : int = 256
    num_workers     : int = 4
    prefetch_factor : int = 2

    patch_size    : tuple[int, int] = (64, 64)
    patch_stride  : int             = 32
    train_azimuth : tuple[int, int] = (1000, 13000)
    val_azimuth   : tuple[int, int] = (13064, 14500)
    test_azimuth  : tuple[int, int] = (14564, 16000)


@dataclass
class UnrolledEntryConfig:
    run_name        : str | None = None
    model_name      : str        = "gamma_net"
    gpu             : int        = 0
    seed            : int        = 0
    logdir          : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/unrolled")
    model_overrides : dict       = field(default_factory=dict)

    paths         : TrainingPathsConfig    = field(default_factory=TrainingPathsConfig)
    training      : UnrolledTrainingConfig = field(default_factory=UnrolledTrainingConfig)
    geometry      : GeometryConfig         = field(default_factory=GeometryConfig)
    normalization : NormalizationConfig    = field(default_factory=NormalizationConfig)
    augmentation  : AugmentationConfig     = field(default_factory=AugmentationConfig)

    curve_loss            : str   = "l1"
    measurement_noise_std : float = 0.0
    power_floor           : float = 1e-6
