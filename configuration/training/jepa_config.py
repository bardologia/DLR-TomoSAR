from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.experiments.benchmark_config    import BenchmarkPathsConfig, TrainingQueueConfig
from configuration.model.autoencoder_models_config import AutoencoderBaseConfig, MlpAutoencoderConfig
from configuration.inference.inference_config      import InferenceConfig
from configuration.sar.geometry_config             import GeometryConfig
from configuration.training.optimization_config    import EarlyStoppingConfig, GradientClipperConfig, OptimizerConfig, SchedulerConfig, WarmupConfig
from configuration.training.runtime_config         import IOConfig, MemoryConfig, OverfitConfig, ResourceConfig, TrainingLoopConfig
from configuration.training.trainer_config         import SharedSubConfigInheritance


@dataclass
class EmbeddingLossConfig:
    use_embedding_mse    : bool  = True
    weight_embedding_mse : float = 1.0

    use_embedding_cosine    : bool  = False
    weight_embedding_cosine : float = 0.0

    use_embedding_smoothl1    : bool  = False
    weight_embedding_smoothl1 : float = 0.0
    smoothl1_beta             : float = 1.0

    use_curve_recon    : bool  = True
    weight_curve_recon : float = 1.0
    curve_kind         : str   = "mse"
    huber_delta        : float = 1.0
    charbonnier_eps    : float = 1e-3


@dataclass
class JepaTrainerConfig(SharedSubConfigInheritance):
    gaussian            : object
    autoencoder    : AutoencoderBaseConfig = field(default_factory=MlpAutoencoderConfig)
    embedding_loss : EmbeddingLossConfig   = field(default_factory=EmbeddingLossConfig)

    profile_autoencoder_mode       : str        = "frozen"
    target_provider                : str        = "stopgrad"
    ema_decay                      : float      = 0.996
    profile_autoencoder_checkpoint : str | None = None

    ae_finetune_lr : float = 3e-5
    ae_finetune_wd : float = 1e-4

    geometry         : GeometryConfig        = field(default_factory=GeometryConfig)
    early_stopping   : EarlyStoppingConfig   = field(default_factory=EarlyStoppingConfig)
    warmup           : WarmupConfig          = field(default_factory=WarmupConfig)
    scheduler        : SchedulerConfig       = field(default_factory=SchedulerConfig)
    io               : IOConfig              = field(default_factory=IOConfig)
    optimizer        : OptimizerConfig       = field(default_factory=OptimizerConfig)
    training         : TrainingLoopConfig    = field(default_factory=TrainingLoopConfig)
    overfit          : OverfitConfig         = field(default_factory=OverfitConfig)
    resources        : ResourceConfig        = field(default_factory=ResourceConfig)
    memory           : MemoryConfig          = field(default_factory=MemoryConfig)
    gradient_clipper : GradientClipperConfig = field(default_factory=GradientClipperConfig)


class JepaDefaults:
    @staticmethod
    def inference() -> InferenceConfig:
        return InferenceConfig(
            run_directory = Path("."),
            save_cubes    = True,
            cpu_workers   = 16,
            gif_axes      = ["elevation", "range", "azimuth"],
        )


@dataclass
class JepaEntryConfig:
    run_name        : str | None = None
    model_name      : str        = "resunet"
    gpu             : int        = 0
    seed            : int        = 0
    n_gaussians     : int        = 5
    logdir          : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/jepa")
    model_overrides : dict       = field(default_factory=dict)

    profile_autoencoder_logdir : Path        = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/profile_autoencoder")
    profile_autoencoder_run    : str | None  = None
    profile_autoencoder_mode   : str         = "frozen"
    target_provider            : str         = "stopgrad"

    embedding_loss : EmbeddingLossConfig = field(default_factory=EmbeddingLossConfig)
    overfit        : OverfitConfig       = field(default_factory=OverfitConfig)
    geometry       : GeometryConfig      = field(default_factory=GeometryConfig)

    paths    : BenchmarkPathsConfig = field(default_factory=BenchmarkPathsConfig)
    training : TrainingQueueConfig  = field(default_factory=TrainingQueueConfig)

    infer_after : bool            = False
    inference   : InferenceConfig = field(default_factory=JepaDefaults.inference)
