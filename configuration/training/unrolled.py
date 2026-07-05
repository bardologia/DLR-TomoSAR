from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.dataset                import AugmentationConfig
from configuration.normalization.general  import NormalizationConfig
from configuration.training.general.run   import TrainingPathsConfig, TrainingQueueConfig
from configuration.sar.geometry_config    import GeometryConfig


@dataclass
class UnrolledEntryConfig:
    run_name        : str | None = None
    model_name      : str        = "gamma_net"
    gpu             : int        = 0
    seed            : int        = 0
    logdir          : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/unrolled")
    model_overrides : dict       = field(default_factory=dict)

    paths         : TrainingPathsConfig = field(default_factory=TrainingPathsConfig)
    training      : TrainingQueueConfig = field(default_factory=TrainingQueueConfig)
    geometry      : GeometryConfig      = field(default_factory=GeometryConfig)
    normalization : NormalizationConfig = field(default_factory=NormalizationConfig)
    augmentation  : AugmentationConfig  = field(default_factory=AugmentationConfig)

    curve_loss            : str   = "l1"
    measurement_noise_std : float = 0.0
    power_floor           : float = 1e-6
