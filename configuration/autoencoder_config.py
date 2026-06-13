from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

import torch.nn as nn

from configuration.benchmark_config import BenchmarkPathsConfig, TrainingQueueConfig
from configuration.training_config  import (
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
class ProfileAutoencoderConfig:
    profile_length     : int   = 256

    embedding_dim      : int   = 24
    embedding_norm     : str   = "l2"
    curve_norm         : str   = "log1p"

    encoder_kind       : str   = "mlp"
    decoder_kind       : str   = "mlp"

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

    encoder_wd         : float = 1e-4
    decoder_wd         : float = 1e-4

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        groups = [
            {"params": list(model.encoder.parameters()), "lr": self.encoder_lr, "weight_decay": self.encoder_wd, "name": "ae_encoder"},
            {"params": list(model.decoder.parameters()), "lr": self.decoder_lr, "weight_decay": self.decoder_wd, "name": "ae_decoder"},
        ]
        return [g for g in groups if len(g["params"]) > 0]


@dataclass
class AutoencoderLossConfig:
    curve_kind      : str   = "mse"
    huber_delta     : float = 1.0
    charbonnier_eps : float = 1e-3


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
    resources           : ResourceConfig            = field(default_factory=ResourceConfig)
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

    autoencoder     : ProfileAutoencoderConfig = field(default_factory=ProfileAutoencoderConfig)
    ae_loss         : AutoencoderLossConfig     = field(default_factory=AutoencoderLossConfig)
    overfit         : OverfitConfig             = field(default_factory=OverfitConfig)
    geometry        : GeometryConfig            = field(default_factory=GeometryConfig)

    paths           : BenchmarkPathsConfig = field(default_factory=BenchmarkPathsConfig)
    training        : TrainingQueueConfig  = field(default_factory=TrainingQueueConfig)
