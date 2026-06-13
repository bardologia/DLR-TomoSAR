from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

import torch.nn as nn

from configuration.benchmark_config import BenchmarkPathsConfig, TrainingQueueConfig
from configuration.inference_config import InferenceConfig
from configuration.training_config  import (
    EarlyStoppingConfig,
    GeometryConfig,
    GradientClipperConfig,
    IOConfig,
    LossConfig,
    LossCurriculumConfig,
    MemoryConfig,
    OptimizerConfig,
    OverfitConfig,
    PermutationMetricsConfig,
    ResourceConfig,
    SchedulerConfig,
    TrainingConfigInner,
    WarmupConfig,
)


@dataclass
class ProfileAutoencoderConfig:
    profile_length     : int   = 256
    n_gaussians        : int   = 5
    params_per_gaussian: int   = 3

    embedding_dim      : int   = 24
    embedding_structure: str   = "plain"

    encoder_kind       : str   = "mlp"
    decoder_kind       : str   = "mlp"
    count_strategy     : str   = "amplitude_threshold"

    hidden_dim         : int   = 128
    depth              : int   = 2
    activation         : str   = "gelu"
    dropout            : float = 0.0
    init_mode          : str   = "default"

    seq_channels       : int   = 32
    seq_kernel_size    : int   = 5
    num_heads          : int   = 4

    encoder_lr         : float = 3e-4
    decoder_lr         : float = 3e-4
    heads_lr           : float = 3e-4

    encoder_wd         : float = 1e-4
    decoder_wd         : float = 1e-4
    heads_wd           : float = 1e-4

    @property
    def out_channels(self) -> int:
        return self.n_gaussians * self.params_per_gaussian

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        groups = [
            {"params": list(model.encoder.parameters()),     "lr": self.encoder_lr, "weight_decay": self.encoder_wd, "name": "ae_encoder"},
            {"params": list(model.decoder.parameters()),     "lr": self.decoder_lr, "weight_decay": self.decoder_wd, "name": "ae_decoder"},
            {"params": list(model.param_heads.parameters()), "lr": self.heads_lr,   "weight_decay": self.heads_wd,   "name": "ae_heads"},
        ]
        return [g for g in groups if len(g["params"]) > 0]


@dataclass
class AutoencoderLossConfig:
    use_ae_curve       : bool  = True
    weight_ae_curve    : float = 1.0
    ae_curve_kind      : str   = "mse"
    ae_huber_delta     : float = 1.0
    ae_charbonnier_eps : float = 1e-3


@dataclass
class EmbeddingLossConfig:
    use_embedding_mse        : bool  = True
    weight_embedding_mse     : float = 1.0

    use_embedding_cosine     : bool  = False
    weight_embedding_cosine  : float = 0.0

    use_embedding_smoothl1   : bool  = False
    weight_embedding_smoothl1: float = 0.0
    smoothl1_beta            : float = 1.0

    standardize_target       : bool  = True
    standardize_momentum     : float = 0.01


@dataclass
class ProfileAeTrainerConfig:
    gaussian            : object
    autoencoder         : ProfileAutoencoderConfig = field(default_factory=ProfileAutoencoderConfig)
    ae_loss             : AutoencoderLossConfig     = field(default_factory=AutoencoderLossConfig)
    geometry            : GeometryConfig            = field(default_factory=GeometryConfig)
    early_stopping      : EarlyStoppingConfig       = field(default_factory=EarlyStoppingConfig)
    warmup              : WarmupConfig              = field(default_factory=WarmupConfig)
    scheduler           : SchedulerConfig           = field(default_factory=SchedulerConfig)
    io                  : IOConfig                  = field(default_factory=IOConfig)
    optimizer           : OptimizerConfig           = field(default_factory=OptimizerConfig)
    training            : TrainingConfigInner       = field(default_factory=TrainingConfigInner)
    overfit             : OverfitConfig             = field(default_factory=OverfitConfig)
    curriculum          : LossCurriculumConfig      = field(default_factory=LossCurriculumConfig)
    resources           : ResourceConfig            = field(default_factory=ResourceConfig)
    memory              : MemoryConfig              = field(default_factory=MemoryConfig)
    gradient_clipper    : GradientClipperConfig     = field(default_factory=GradientClipperConfig)
    permutation_metrics : PermutationMetricsConfig  = field(default_factory=PermutationMetricsConfig)


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
    curriculum          : LossCurriculumConfig      = field(default_factory=LossCurriculumConfig)
    resources           : ResourceConfig            = field(default_factory=ResourceConfig)
    memory              : MemoryConfig              = field(default_factory=MemoryConfig)
    gradient_clipper    : GradientClipperConfig     = field(default_factory=GradientClipperConfig)
    permutation_metrics : PermutationMetricsConfig  = field(default_factory=PermutationMetricsConfig)


class JepaDefaults:
    @staticmethod
    def stage_a_curriculum() -> LossCurriculumConfig:
        warmup = LossConfig(use_mse_curve=True, weight_mse_curve=1.0, use_param_huber=True, weight_param_huber=1.0)
        return LossCurriculumConfig(enabled=False, warmup=warmup, complete=warmup)

    @staticmethod
    def stage_b_curriculum() -> LossCurriculumConfig:
        warmup = LossConfig(use_mse_curve=True, weight_mse_curve=1.0, use_param_huber=True, weight_param_huber=1.0)
        return LossCurriculumConfig(enabled=False, warmup=warmup, complete=warmup)

    @staticmethod
    def inference() -> InferenceConfig:
        return InferenceConfig(
            run_directory = Path("."),
            save_cubes    = True,
            cpu_workers   = 16,
            gif_axes      = ["elevation", "range", "azimuth"],
        )


@dataclass
class ProfileAeEntryConfig:
    run_name        : str | None = None
    gpu             : int        = 0
    seed            : int        = 0
    n_gaussians     : int        = 5
    logdir          : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/logs/jepa_stage_a")

    pixel_subsample : float      = 1.0
    keep_empty_frac : float      = 0.05

    autoencoder     : ProfileAutoencoderConfig = field(default_factory=ProfileAutoencoderConfig)
    ae_loss         : AutoencoderLossConfig     = field(default_factory=AutoencoderLossConfig)
    curriculum      : LossCurriculumConfig      = field(default_factory=JepaDefaults.stage_a_curriculum)
    overfit         : OverfitConfig             = field(default_factory=OverfitConfig)
    geometry        : GeometryConfig            = field(default_factory=GeometryConfig)

    paths           : BenchmarkPathsConfig = field(default_factory=BenchmarkPathsConfig)
    training        : TrainingQueueConfig  = field(default_factory=TrainingQueueConfig)


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
    curriculum      : LossCurriculumConfig      = field(default_factory=JepaDefaults.stage_b_curriculum)
    overfit         : OverfitConfig             = field(default_factory=OverfitConfig)
    geometry        : GeometryConfig            = field(default_factory=GeometryConfig)

    paths           : BenchmarkPathsConfig = field(default_factory=BenchmarkPathsConfig)
    training        : TrainingQueueConfig  = field(default_factory=TrainingQueueConfig)

    infer_after     : bool            = False
    inference       : InferenceConfig = field(default_factory=JepaDefaults.inference)
