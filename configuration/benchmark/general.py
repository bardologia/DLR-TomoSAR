from __future__ import annotations

from dataclasses import dataclass, field

from configuration.benchmark.jepa       import JepaBenchConfig
from configuration.dataset              import AugmentationConfig, InputConfig
from configuration.normalization.general import NormalizationConfig
from configuration.sar.geometry_config  import GeometryConfig
from configuration.training.backbone     import default_curriculum
from configuration.training.general.loss import LossCurriculumConfig
from configuration.training.general.run import RunPathsConfig, TrainingQueueConfig


@dataclass
class MaxBatchConfig:
    vram_budget_gb : float = 40.0
    max_batch      : int   = 512
    measure_steps  : int   = 3
    seed           : int   = 42


@dataclass
class SizeMatchConfig:
    reference_model : str         = "unet"
    tolerance       : float       = 0.05
    max_iterations  : int         = 100
    scale_low       : float       = 0.05
    scale_high      : float       = 8.0
    in_channels     : int         = 9
    locked_params   : tuple[str, ...] = ("embedding_dim", "embedding_dims")


@dataclass
class InferenceQueueConfig:
    split           : str        = "test"
    checkpoint_name : str        = "best_model.pt"
    batch_size      : int | None = None
    num_workers     : int        = 4
    cpu_workers     : int        = 16

    save_plots      : bool = False
    save_animations : bool = False
    save_cubes      : bool = False
    stitch_window   : str  = "hann"

    n_best_profiles   : int = 12
    n_worst_profiles  : int = 12
    n_random_profiles : int = 12

    n_range_slices     : int = 5
    n_azimuth_slices   : int = 5
    n_elevation_slices : int = 5

    gif_axes       : list[str] = field(default_factory=lambda: ["elevation"])
    gif_fps        : int       = 12
    gif_max_frames : int       = 150


@dataclass
class ComparisonReportConfig:
    embed_images : bool = False


def _default_ae_loss():
    from configuration.training.profile_autoencoder import ProfileAeLossConfig

    return ProfileAeLossConfig()


@dataclass
class BenchmarkConfig:
    training_type : str = "backbone"

    paths      : RunPathsConfig         = field(default_factory=RunPathsConfig)
    max_batch  : MaxBatchConfig         = field(default_factory=MaxBatchConfig)
    size_match : SizeMatchConfig        = field(default_factory=SizeMatchConfig)
    training   : TrainingQueueConfig    = field(default_factory=TrainingQueueConfig)
    inference  : InferenceQueueConfig   = field(default_factory=InferenceQueueConfig)
    comparison : ComparisonReportConfig = field(default_factory=ComparisonReportConfig)

    input         : InputConfig         = field(default_factory=InputConfig.full_stack)
    geometry      : GeometryConfig      = field(default_factory=GeometryConfig)
    normalization : NormalizationConfig = field(default_factory=NormalizationConfig)
    augmentation  : AugmentationConfig  = field(default_factory=AugmentationConfig)
    curriculum    : LossCurriculumConfig = field(default_factory=default_curriculum)
    predict_presence : bool             = False

    ae_loss         : object          = field(default_factory=_default_ae_loss)
    jepa            : JepaBenchConfig  = field(default_factory=JepaBenchConfig)
    pixel_subsample : float           = 1.0
    keep_empty_frac : float           = 0.05

    gpus            : list[int]  = field(default_factory=lambda: [2, 3])
    skip_models     : list[str]  = field(default_factory=list)
    run_tag         : str | None = None
    resume          : bool       = True
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=list)
    n_gaussians     : int        = 5
    poll_interval_s : float      = 5.0

    sweep_loss_components : list[str] = field(default_factory=lambda: ["param_l1"])

    def runs_size_match(self) -> bool:
        return self.training_type == "backbone"

    def runs_max_batch(self) -> bool:
        return self.training_type == "backbone"

    def runs_inference(self) -> bool:
        return self.training_type in ("backbone", "jepa")
