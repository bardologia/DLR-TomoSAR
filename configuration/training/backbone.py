from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.training.general.run         import RunPathsConfig, TrainingQueueConfig
from configuration.dataset                       import AugmentationConfig, InputConfig
from configuration.inference.general            import InferenceConfig
from configuration.sar.gaussian_config          import GaussianConfig
from configuration.sar.geometry_config          import GeometryConfig
from configuration.normalization.general        import NormalizationConfig
from configuration.training.general.ablation    import AblationCatalog
from configuration.training.general.loss        import LossConfig, LossCurriculumConfig
from configuration.training.general.optimization import EarlyStoppingConfig, GradientClipperConfig, OptimizerConfig, SchedulerConfig, WarmupConfig
from configuration.training.general.runtime     import IOConfig, MemoryConfig, OverfitConfig, ResourceConfig, TrainingLoopConfig
from configuration.training.general.pretraining import PretrainConfig


def _default_curriculum() -> LossCurriculumConfig:
    warmup = LossConfig(
        use_l1_curve    = True,
        weight_l1_curve = 1.0,
        param_matching  = AblationCatalog.PARAM_MATCH_FULL,
    )

    complete = LossConfig(
        use_l1_curve             = True,
        weight_l1_curve          = 1.0,
        param_matching           = AblationCatalog.PARAM_MATCH_FULL,
        use_active_normalization = True,
        presence_balance         = True,
        use_coherence_resyn      = True,
        weight_coherence_resyn   = 0.05,
        use_covariance_match     = True,
        weight_covariance_match  = 0.05,
    )

    return LossCurriculumConfig(
        enabled    = True,
        swap_epoch = AblationCatalog.CURRICULUM_SWAP_EPOCH,
        warmup     = warmup,
        complete   = complete,
    )


def _default_inference() -> InferenceConfig:
    return InferenceConfig(
        run_directory   = Path("."),
        save_cubes      = True,
        cpu_workers     = 16,
        gif_axes        = ["elevation", "range", "azimuth"],
        compute_reduced = False,
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


def _default_presence_trials() -> dict:
    active_norm = {"use_active_normalization": True}
    balance     = {"presence_balance": True}
    presence    = {"predict_presence": True, "use_presence_bce": True, "weight_presence_bce": 1.0}
    focal       = {"amp_focal_gamma": 2.0}

    return {
        "none"        : {},
        "A"           : {**active_norm},
        "B"           : {**balance},
        "P"           : {**presence},
        "F"           : {**focal},
        "AB"          : {**active_norm, **balance},
        "AP"          : {**active_norm, **presence},
        "AF"          : {**active_norm, **focal},
        "BP"          : {**balance, **presence},
        "BF"          : {**balance, **focal},
        "PF"          : {**presence, **focal},
        "ABP"         : {**active_norm, **balance, **presence},
        "ABF"         : {**active_norm, **balance, **focal},
        "APF"         : {**active_norm, **presence, **focal},
        "BPF"         : {**balance, **presence, **focal},
        "ABPF"        : {**active_norm, **balance, **presence, **focal},
        "ABPF-bce0.5" : {**active_norm, **balance, **presence, **focal, "weight_presence_bce": 0.5},
        "ABPF-bce2"   : {**active_norm, **balance, **presence, **focal, "weight_presence_bce": 2.0},
        "ABPF-fg1"    : {**active_norm, **balance, **presence, **focal, "amp_focal_gamma": 1.0},
        "ABPF-fg3"    : {**active_norm, **balance, **presence, **focal, "amp_focal_gamma": 3.0},
    }


def _default_input_trials() -> dict:
    return {
        "amp-allsec-noifg" : {"use_primary": True, "use_secondaries": True, "use_interferograms": False},
    }


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


@dataclass
class BackboneEntryConfig:
    run_name        : str | None = None
    backbone_name   : str        = "resunet"
    gpu             : int        = 0
    seed             : int        = 0
    seeds            : list[int]  = field(default_factory=list)
    n_gaussians      : int        = 5
    predict_presence : bool       = False
    logdir          : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/backbone")
    model_overrides : dict       = field(default_factory=dict)

    paths      : RunPathsConfig       = field(default_factory=RunPathsConfig)
    training   : TrainingQueueConfig  = field(default_factory=TrainingQueueConfig)
    pretrain   : PretrainConfig       = field(default_factory=PretrainConfig)
    curriculum : LossCurriculumConfig = field(default_factory=_default_curriculum)
    geometry   : GeometryConfig       = field(default_factory=GeometryConfig)
    input      : InputConfig          = field(default_factory=InputConfig.full_stack)
    normalization : NormalizationConfig = field(default_factory=NormalizationConfig)
    augmentation  : AugmentationConfig  = field(default_factory=AugmentationConfig)

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
    presence_trials  : dict                  = field(default_factory=_default_presence_trials)
    secondary_trials : SecondaryTrialsConfig = field(default_factory=SecondaryTrialsConfig)
    patch_trials     : PatchTrialsConfig     = field(default_factory=PatchTrialsConfig)
    input_trials     : dict                  = field(default_factory=_default_input_trials)

    ablation_features     : list = field(default_factory=AblationCatalog.default_features)
    ablation_catalog      : dict = field(default_factory=AblationCatalog.as_dict)
    ablation_include_full : bool = True

    gpus             : list[int]             = field(default_factory=lambda: [0, 1, 3])
    poll_interval_s  : float                 = 5.0
