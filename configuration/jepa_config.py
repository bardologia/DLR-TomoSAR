from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.autoencoder_config import ProfileAutoencoderConfig
from configuration.benchmark_config   import BenchmarkPathsConfig, TrainingQueueConfig
from configuration.inference_config   import InferenceConfig
from configuration.training_config    import (
    EarlyStoppingConfig,
    GeometryConfig,
    GradientClipperConfig,
    IOConfig,
    OptimizerConfig,
    OverfitConfig,
    ResourceConfig,
    SchedulerConfig,
    TrainingConfigInner,
    WarmupConfig,
)


@dataclass
class EmbeddingLossConfig:
    use_embedding_mse        : bool  = True
    weight_embedding_mse     : float = 1.0

    use_embedding_cosine     : bool  = False
    weight_embedding_cosine  : float = 0.0

    use_embedding_smoothl1   : bool  = False
    weight_embedding_smoothl1: float = 0.0
    smoothl1_beta            : float = 1.0

    use_curve_recon          : bool  = True
    weight_curve_recon       : float = 1.0
    curve_kind               : str   = "mse"
    huber_delta              : float = 1.0
    charbonnier_eps          : float = 1e-3


@dataclass
class JepaTrainerConfig:
    gaussian            : object
    autoencoder         : ProfileAutoencoderConfig = field(default_factory=ProfileAutoencoderConfig)
    embedding_loss      : EmbeddingLossConfig       = field(default_factory=EmbeddingLossConfig)

    stage_a_mode        : str                       = "frozen"
    target_provider     : str                       = "stopgrad"
    ema_decay           : float                     = 0.996
    stage_a_checkpoint  : str | None                = None

    ae_finetune_lr      : float                     = 3e-5
    ae_finetune_wd      : float                     = 1e-4

    geometry            : GeometryConfig            = field(default_factory=GeometryConfig)
    early_stopping      : EarlyStoppingConfig       = field(default_factory=EarlyStoppingConfig)
    warmup              : WarmupConfig              = field(default_factory=WarmupConfig)
    scheduler           : SchedulerConfig           = field(default_factory=SchedulerConfig)
    io                  : IOConfig                  = field(default_factory=IOConfig)
    optimizer           : OptimizerConfig           = field(default_factory=OptimizerConfig)
    training            : TrainingConfigInner       = field(default_factory=TrainingConfigInner)
    overfit             : OverfitConfig             = field(default_factory=OverfitConfig)
    resources           : ResourceConfig            = field(default_factory=ResourceConfig)
    gradient_clipper    : GradientClipperConfig     = field(default_factory=GradientClipperConfig)


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
    logdir          : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/logs/jepa_stage_b")
    model_overrides : dict       = field(default_factory=dict)

    stage_a_run     : Path | None = None
    stage_a_mode    : str         = "frozen"
    target_provider : str         = "stopgrad"

    autoencoder     : ProfileAutoencoderConfig = field(default_factory=ProfileAutoencoderConfig)
    embedding_loss  : EmbeddingLossConfig       = field(default_factory=EmbeddingLossConfig)
    overfit         : OverfitConfig             = field(default_factory=OverfitConfig)
    geometry        : GeometryConfig            = field(default_factory=GeometryConfig)

    paths           : BenchmarkPathsConfig = field(default_factory=BenchmarkPathsConfig)
    training        : TrainingQueueConfig  = field(default_factory=TrainingQueueConfig)

    infer_after     : bool            = False
    inference       : InferenceConfig = field(default_factory=JepaDefaults.inference)
