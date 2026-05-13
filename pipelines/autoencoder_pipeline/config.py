from __future__ import annotations

from dataclasses import dataclass, field
from enum        import Enum
from typing      import Any


class BackboneType(Enum):
    conv1d = "conv1d"
    mlp    = "mlp"


class ContrastiveView(Enum):
    augmentation = "augmentation"
    neighbor     = "neighbor"
    both         = "both"


class ReconLossName(Enum):
    mse         = "mse"
    l1          = "l1"
    smooth_l1   = "smooth_l1"
    charbonnier = "charbonnier"


class NormalizationMode(Enum):
    none               = "none"
    per_profile_max    = "per_profile_max"
    per_profile_zscore = "per_profile_zscore"
    global_stat        = "global"


@dataclass
class EncoderConfig:
    backbone            : BackboneType = BackboneType.conv1d
    channels            : list[int]    = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_size         : int          = 5
    stride              : int          = 2
    activation          : str          = "gelu"
    normalization       : str          = "batch"
    dropout             : float        = 0.0
    mlp_hidden          : list[int]    = field(default_factory=lambda: [256, 256, 128])
    proj_hidden         : list[int]    = field(default_factory=lambda: [256])
    proj_dim            : int          = 64
    use_projection_head : bool         = True


@dataclass
class DecoderConfig:
    backbone          : BackboneType = BackboneType.conv1d
    channels          : list[int]    = field(default_factory=lambda: [256, 128, 64, 32])
    kernel_size       : int          = 5
    stride            : int          = 2
    activation        : str          = "gelu"
    normalization     : str          = "batch"
    dropout           : float        = 0.0
    mlp_hidden        : list[int]    = field(default_factory=lambda: [128, 256, 256])
    output_activation : str | None   = None


@dataclass
class LossConfig:
    reconstruction_weight   : float           = 1.0
    variance_weight         : float           = 1.0
    covariance_weight       : float           = 0.04
    contrastive_weight      : float           = 0.5

    reconstruction_loss     : ReconLossName   = ReconLossName.mse
    charbonnier_eps         : float           = 1e-3

    variance_target_std     : float           = 1.0
    contrastive_temperature : float           = 0.1
    contrastive_view        : ContrastiveView = ContrastiveView.augmentation

    use_reconstruction      : bool            = True
    use_variance            : bool            = True
    use_covariance          : bool            = True
    use_contrastive         : bool            = True


@dataclass
class AugmentationConfig:
    jitter_std     : float                = 0.02
    scale_range    : tuple[float, float]  = (0.9, 1.1)
    shift_max      : int                  = 2
    mask_prob      : float                = 0.1
    mask_max_width : int                  = 4
    seed           : int                  = 0


@dataclass
class DataConfig:
    profile_length     : int                 = 80
    normalize          : str                 = "per_profile_max"
    log_compress       : bool                = True
    log_eps            : float               = 1e-6
    drop_zero_profiles : bool                = True
    contrastive_view   : ContrastiveView     = ContrastiveView.augmentation
    augmentation       : AugmentationConfig  = field(default_factory=AugmentationConfig)
    max_profiles       : int | None          = None
    sampling_seed      : int                 = 42


@dataclass
class TrainerConfig:
    epochs              : int            = 100
    learning_rate       : float          = 1e-3
    weight_decay        : float          = 1e-5
    optimizer           : str            = "adamw"
    scheduler           : str            = "cosine"
    scheduler_kwargs    : dict[str, Any] = field(default_factory=dict)
    warmup_steps        : int            = 0
    grad_clip           : float          = 1.0
    use_amp             : bool           = True
    save_every          : int            = 10
    val_every           : int            = 1
    early_stop_patience : int            = 20
    device              : str            = "cuda"
    log_every_n_steps   : int            = 50


@dataclass
class IOConfig:
    logdir         : str        = "logs"
    run_name       : str | None = None
    tb_dir         : str | None = None
    docs_dir       : str | None = None
    logs_dir       : str | None = None
    images_dir     : str | None = None
    embed_dir      : str | None = None
    recon_dir      : str | None = None
    checkpoint_dir : str | None = None
    report_path    : str | None = None


@dataclass
class AutoencoderConfig:
    profile_length : int           = 80
    latent_dim     : int           = 32

    encoder        : EncoderConfig = field(default_factory=EncoderConfig)
    decoder        : DecoderConfig = field(default_factory=DecoderConfig)
    loss           : LossConfig    = field(default_factory=LossConfig)
    data           : DataConfig    = field(default_factory=DataConfig)
    trainer        : TrainerConfig = field(default_factory=TrainerConfig)
    io             : IOConfig      = field(default_factory=IOConfig)

    init_mode      : str           = "default"
