from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.benchmark_config import BenchmarkPathsConfig, TrainingQueueConfig
from configuration.models_config    import AutoencoderConfig
from configuration.geometry_config     import GeometryConfig
from configuration.optimization_config import EarlyStoppingConfig, GradientClipperConfig, OptimizerConfig, SchedulerConfig, WarmupConfig
from configuration.runtime_config      import IOConfig, MemoryConfig, OverfitConfig, ResourceConfig, TrainingLoopConfig
from configuration.trainer_config       import SharedSubConfigInheritance


@dataclass
class AutoencoderLossConfig:
    curve_kind      : str   = "mse"
    huber_delta     : float = 1.0
    charbonnier_eps : float = 1e-3


@dataclass
class ProfileAeTrainerConfig(SharedSubConfigInheritance):
    gaussian            : object
    autoencoder         : AutoencoderConfig = field(default_factory=AutoencoderConfig)
    ae_loss             : AutoencoderLossConfig     = field(default_factory=AutoencoderLossConfig)
    geometry            : GeometryConfig            = field(default_factory=GeometryConfig)
    early_stopping      : EarlyStoppingConfig       = field(default_factory=EarlyStoppingConfig)
    warmup              : WarmupConfig              = field(default_factory=WarmupConfig)
    scheduler           : SchedulerConfig           = field(default_factory=SchedulerConfig)
    io                  : IOConfig                  = field(default_factory=IOConfig)
    optimizer           : OptimizerConfig           = field(default_factory=OptimizerConfig)
    training            : TrainingLoopConfig       = field(default_factory=TrainingLoopConfig)
    overfit             : OverfitConfig             = field(default_factory=OverfitConfig)
    resources           : ResourceConfig            = field(default_factory=ResourceConfig)
    memory              : MemoryConfig              = field(default_factory=MemoryConfig)
    gradient_clipper    : GradientClipperConfig     = field(default_factory=GradientClipperConfig)


@dataclass
class ProfileAeEntryConfig:
    run_name        : str | None = None
    gpu             : int        = 0
    seed            : int        = 0
    n_gaussians     : int        = 5
    logdir          : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/logs/jepa_stage_a")

    pixel_subsample : float      = 1.0
    keep_empty_frac : float      = 0.05

    autoencoder     : AutoencoderConfig = field(default_factory=AutoencoderConfig)
    ae_loss         : AutoencoderLossConfig     = field(default_factory=AutoencoderLossConfig)
    overfit         : OverfitConfig             = field(default_factory=OverfitConfig)
    geometry        : GeometryConfig            = field(default_factory=GeometryConfig)

    paths           : BenchmarkPathsConfig = field(default_factory=BenchmarkPathsConfig)
    training        : TrainingQueueConfig  = field(default_factory=TrainingQueueConfig)
