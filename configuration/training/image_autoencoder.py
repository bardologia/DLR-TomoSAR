from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.benchmark.general                import BenchmarkPathsConfig, TrainingQueueConfig
from configuration.architectures.image_autoencoder  import Conv2dImageAutoencoderConfig, ImageAutoencoderBaseConfig
from configuration.sar.geometry_config              import GeometryConfig
from configuration.training.general.optimization    import EarlyStoppingConfig, GradientClipperConfig, OptimizerConfig, SchedulerConfig, WarmupConfig
from configuration.training.general.runtime         import IOConfig, MemoryConfig, OverfitConfig, ResourceConfig, TrainingLoopConfig
from configuration.training.general.trainer         import SharedSubConfigInheritance


@dataclass
class ImageAeLossConfig:
    recon_kind      : str   = "mse"
    huber_delta     : float = 1.0
    charbonnier_eps : float = 1e-3


@dataclass
class ImageAeTrainerConfig(SharedSubConfigInheritance):
    gaussian          : object
    image_autoencoder : ImageAutoencoderBaseConfig = field(default_factory=Conv2dImageAutoencoderConfig)
    ae_loss           : ImageAeLossConfig          = field(default_factory=ImageAeLossConfig)
    geometry          : GeometryConfig             = field(default_factory=GeometryConfig)
    early_stopping    : EarlyStoppingConfig        = field(default_factory=EarlyStoppingConfig)
    warmup            : WarmupConfig               = field(default_factory=WarmupConfig)
    scheduler         : SchedulerConfig            = field(default_factory=SchedulerConfig)
    io                : IOConfig                   = field(default_factory=IOConfig)
    optimizer         : OptimizerConfig            = field(default_factory=OptimizerConfig)
    training          : TrainingLoopConfig         = field(default_factory=TrainingLoopConfig)
    overfit           : OverfitConfig              = field(default_factory=OverfitConfig)
    resources         : ResourceConfig             = field(default_factory=ResourceConfig)
    memory            : MemoryConfig               = field(default_factory=MemoryConfig)
    gradient_clipper  : GradientClipperConfig      = field(default_factory=GradientClipperConfig)


@dataclass
class ImageAeEntryConfig:
    run_name    : str | None = None
    gpu         : int        = 0
    seed        : int        = 0
    n_gaussians : int        = 5
    logdir      : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/image_autoencoder")

    ae_model_name     : str                        = "conv2d_ae"
    image_autoencoder : ImageAutoencoderBaseConfig = field(default_factory=Conv2dImageAutoencoderConfig)
    ae_loss           : ImageAeLossConfig          = field(default_factory=ImageAeLossConfig)
    overfit           : OverfitConfig              = field(default_factory=OverfitConfig)
    geometry          : GeometryConfig             = field(default_factory=GeometryConfig)

    paths    : BenchmarkPathsConfig = field(default_factory=BenchmarkPathsConfig)
    training : TrainingQueueConfig  = field(default_factory=TrainingQueueConfig)
