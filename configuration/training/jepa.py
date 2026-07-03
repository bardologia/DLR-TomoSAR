from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.training.general.run               import RunPathsConfig, TrainingQueueConfig
from configuration.architectures.profile_autoencoder  import ProfileAutoencoderBaseConfig, MlpAutoencoderConfig
from configuration.architectures.image_autoencoder    import ImageAutoencoderBaseConfig
from configuration.inference.general                  import InferenceConfig
from configuration.sar.geometry_config                import GeometryConfig
from configuration.training.general.loss              import LossConfig, ParamMatching
from configuration.training.general.optimization      import EarlyStoppingConfig, GradientClipperConfig, OptimizerConfig, SchedulerConfig, WarmupConfig
from configuration.training.general.runtime           import IOConfig, MemoryConfig, OverfitConfig, ResourceConfig, TrainingLoopConfig
from configuration.training.general.pretraining       import PretrainConfig
from configuration.training.general.trainer           import SharedSubConfigInheritance


def default_param_loss() -> LossConfig:
    return LossConfig(
        use_param_l1             = True,
        weight_param_l1          = 1.0,
        param_matching           = ParamMatching.SORTED_GT,
        use_active_normalization = True,
        presence_balance         = False,
    )


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
    gaussian       : object
    autoencoder    : ProfileAutoencoderBaseConfig | None = None
    embedding_loss : EmbeddingLossConfig          = field(default_factory=EmbeddingLossConfig)

    profile_autoencoder_mode       : str        = "frozen"
    target_provider                : str        = "stopgrad"
    ema_decay                      : float      = 0.996
    profile_autoencoder_checkpoint : str | None = None

    ae_finetune_lr : float = 3e-5
    ae_finetune_wd : float = 1e-4

    image_autoencoder            : ImageAutoencoderBaseConfig | None = None
    image_autoencoder_mode       : str                              = "frozen"
    image_autoencoder_checkpoint : str | None                       = None

    image_ae_finetune_lr : float = 3e-5
    image_ae_finetune_wd : float = 1e-4

    param_loss : LossConfig = field(default_factory=default_param_loss)

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
    backbone_name   : str        = "resunet"
    gpu             : int        = 0
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=list)
    logdir          : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/jepa")
    model_overrides : dict       = field(default_factory=dict)

    profile_autoencoder_logdir : Path        = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/profile_autoencoder")
    profile_autoencoder_run    : str | None  = None
    profile_autoencoder_mode   : str         = "frozen"
    target_provider            : str         = "stopgrad"

    image_autoencoder_logdir : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/image_autoencoder")
    image_autoencoder_run    : str | None  = None
    image_autoencoder_mode   : str         = "frozen"

    embedding_loss : EmbeddingLossConfig = field(default_factory=EmbeddingLossConfig)
    param_loss     : LossConfig          = field(default_factory=default_param_loss)
    geometry       : GeometryConfig      = field(default_factory=GeometryConfig)

    paths    : RunPathsConfig      = field(default_factory=RunPathsConfig)
    training : TrainingQueueConfig = field(default_factory=TrainingQueueConfig)
    pretrain : PretrainConfig      = field(default_factory=PretrainConfig)

    infer_after : bool            = False
    inference   : InferenceConfig = field(default_factory=JepaDefaults.inference)
