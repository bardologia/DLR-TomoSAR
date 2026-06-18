from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.benchmark.general            import BenchmarkPathsConfig, TrainingQueueConfig
from configuration.inference.general            import InferenceConfig
from configuration.sar.gaussian_config          import GaussianConfig
from configuration.sar.geometry_config          import GeometryConfig
from configuration.training.general.loss        import LossConfig, LossCurriculumConfig
from configuration.training.general.optimization import EarlyStoppingConfig, GradientClipperConfig, OptimizerConfig, SchedulerConfig, WarmupConfig
from configuration.training.general.runtime     import IOConfig, MemoryConfig, OverfitConfig, PermutationMetricsConfig, ResourceConfig, TrainingLoopConfig


def _default_curriculum() -> LossCurriculumConfig:
    return LossCurriculumConfig(
        enabled  = False,
        warmup   = LossConfig(use_param_l1=True, weight_param_l1=1.0),
        complete = LossConfig(use_param_l1=True, weight_param_l1=1.0),
    )


def _default_inference() -> InferenceConfig:
    return InferenceConfig(
        run_directory = Path("."),
        save_cubes    = True,
        cpu_workers   = 16,
        gif_axes      = ["elevation", "range", "azimuth"],
    )


def _default_warmup_losses() -> dict:
    return {
        "pL11" : {"use_param_l1": True, "weight_param_l1": 1.0},
    }


def _default_complete_losses() -> dict:
    curve_terms = {
        "mse"   : ("use_mse_curve",          "weight_mse_curve"),
        "l1"    : ("use_l1_curve",           "weight_l1_curve"),
        "huber" : ("use_huber_curve",        "weight_huber_curve"),
        "charb" : ("use_charbonnier_curve",  "weight_charbonnier_curve"),
        "cos"   : ("use_cosine_curve",       "weight_cosine_curve"),
        "spec"  : ("use_spectral_coherence", "weight_spectral_coh"),
        "ssim"  : ("use_ssim_curve",         "weight_ssim_curve"),
    }
    weights = [0.01, 0.05, 0.02]

    losses = {}
    for label, (use_key, weight_key) in curve_terms.items():
        for weight in weights:
            losses[f"pL11-{label}{weight:g}"] = {
                "use_param_l1"    : True,
                "weight_param_l1" : 1.0,
                use_key           : True,
                weight_key        : weight,
            }

    return losses


@dataclass
class PatchTrialsConfig:
    sizes        : list[int] = field(default_factory=lambda: [32, 48, 64, 96, 128])
    stride_ratio : float     = 0.5


@dataclass
class SecondaryTrialsConfig:
    strategy      : str          = "consecutive"
    n_secondaries : int          = 4
    n_trials      : int          = 8
    mean          : float | None = None
    sigma         : float | None = None
    block_step    : int          = 1
    spacing       : int          = 2
    seed          : int          = 0


@dataclass
class BackboneTrainerConfig:
    gaussian            : GaussianConfig
    geometry            : GeometryConfig           = field(default_factory=GeometryConfig)
    early_stopping      : EarlyStoppingConfig      = field(default_factory=EarlyStoppingConfig)
    warmup              : WarmupConfig             = field(default_factory=WarmupConfig)
    scheduler           : SchedulerConfig          = field(default_factory=SchedulerConfig)
    io                  : IOConfig                 = field(default_factory=IOConfig)
    optimizer           : OptimizerConfig          = field(default_factory=OptimizerConfig)
    training            : TrainingLoopConfig       = field(default_factory=TrainingLoopConfig)
    overfit             : OverfitConfig            = field(default_factory=OverfitConfig)
    curriculum          : LossCurriculumConfig     = field(default_factory=LossCurriculumConfig)
    resources           : ResourceConfig           = field(default_factory=ResourceConfig)
    memory              : MemoryConfig             = field(default_factory=MemoryConfig)
    gradient_clipper    : GradientClipperConfig    = field(default_factory=GradientClipperConfig)
    permutation_metrics : PermutationMetricsConfig = field(default_factory=PermutationMetricsConfig)


@dataclass
class BackboneEntryConfig:
    run_name        : str | None = None
    backbone_name   : str        = "resunet"
    gpu             : int        = 0
    seed             : int        = 0
    n_gaussians      : int        = 5
    predict_presence : bool       = False
    logdir          : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/backbone")
    model_overrides : dict       = field(default_factory=dict)

    paths      : BenchmarkPathsConfig = field(default_factory=BenchmarkPathsConfig)
    training   : TrainingQueueConfig  = field(default_factory=TrainingQueueConfig)
    curriculum : LossCurriculumConfig = field(default_factory=_default_curriculum)
    overfit    : OverfitConfig        = field(default_factory=OverfitConfig)
    geometry   : GeometryConfig       = field(default_factory=GeometryConfig)

    probe_enabled    : bool = False
    probe_n_batches  : int  = 1000
    probe_reference  : str  = "param_l1"
    probe_exit_after : bool = True

    infer_after : bool            = False
    inference   : InferenceConfig = field(default_factory=_default_inference)

    trials_enabled   : bool                  = False
    trials_mode      : str                   = "curriculum"
    warmup_losses    : dict                  = field(default_factory=_default_warmup_losses)
    complete_losses  : dict                  = field(default_factory=_default_complete_losses)
    secondary_trials : SecondaryTrialsConfig = field(default_factory=SecondaryTrialsConfig)
    patch_trials     : PatchTrialsConfig     = field(default_factory=PatchTrialsConfig)
    gpus             : list[int]             = field(default_factory=lambda: [0, 1, 3])
    poll_interval_s  : float                 = 5.0
