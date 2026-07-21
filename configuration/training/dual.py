from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.training.general.run         import TrainingPathsConfig, TrainingQueueConfig, standard_seeds
from configuration.dataset                      import AugmentationConfig, InputConfig
from configuration.inference.general            import InferenceConfig
from configuration.sar.geometry_config          import GeometryConfig
from configuration.normalization.general        import NormalizationConfig
from configuration.training.backbone            import _default_inference, default_curriculum
from configuration.training.general.loss        import LossCurriculumConfig, ParamMatching
from configuration.training.general.runtime     import OverfitCheckConfig
from configuration.training.general.pretraining import PretrainConfig


def dual_curriculum() -> LossCurriculumConfig:
    curriculum = default_curriculum()

    curriculum.warmup.param_matching   = ParamMatching.HUNGARIAN
    curriculum.complete.param_matching = ParamMatching.HUNGARIAN

    return curriculum


def _parity_resunet_features() -> list[int]:
    return [48, 96, 184, 352]


def _default_dual_input_trials() -> dict:
    full   = ["pass", "ifg"]
    passes = ["pass"]
    ifg    = ["ifg"]

    return {
        "full-full" : {"params": full,   "existence": full},
        "pass-full" : {"params": passes, "existence": full},
        "full-pass" : {"params": full,   "existence": passes},
        "ifg-full"  : {"params": ifg,    "existence": full},
        "full-ifg"  : {"params": full,   "existence": ifg},
        "pass-ifg"  : {"params": passes, "existence": ifg},
        "ifg-pass"  : {"params": ifg,    "existence": passes},
    }


@dataclass
class DualInputTrialsConfig:
    params_features    : list[int] = field(default_factory=_parity_resunet_features)
    existence_features : list[int] = field(default_factory=_parity_resunet_features)
    trials             : dict      = field(default_factory=_default_dual_input_trials)


@dataclass
class DualEntryConfig:
    run_name        : str | None = None
    resume          : bool       = True
    model_name      : str        = "dual_resunet"
    gpu             : int        = 0
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=standard_seeds)
    logdir          : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/dual")
    model_overrides : dict       = field(default_factory=dict)

    params_backbone    : str       = "resunet"
    existence_backbone : str       = "resunet"
    params_input       : list[str] = field(default_factory=lambda: ["pass", "ifg"])
    existence_input    : list[str] = field(default_factory=lambda: ["ifg"])

    paths         : TrainingPathsConfig  = field(default_factory=TrainingPathsConfig)
    training      : TrainingQueueConfig  = field(default_factory=TrainingQueueConfig)
    pretrain      : PretrainConfig       = field(default_factory=PretrainConfig)
    curriculum    : LossCurriculumConfig = field(default_factory=dual_curriculum)
    geometry      : GeometryConfig       = field(default_factory=GeometryConfig)
    input         : InputConfig          = field(default_factory=InputConfig.full_stack)
    normalization : NormalizationConfig  = field(default_factory=NormalizationConfig)
    augmentation  : AugmentationConfig   = field(default_factory=AugmentationConfig)

    probe_enabled    : bool = False
    probe_n_batches  : int  = 1000
    probe_reference  : str  = "param_l1"
    probe_exit_after : bool = True

    overfit_check : OverfitCheckConfig = field(default_factory=OverfitCheckConfig)

    infer_after : bool            = False
    inference   : InferenceConfig = field(default_factory=_default_inference)

    trials_enabled : bool                  = False
    trials_mode    : str                   = "input"
    input_trials   : DualInputTrialsConfig = field(default_factory=DualInputTrialsConfig)

    gpus             : list[int] = field(default_factory=lambda: [0, 1, 3])
    gpus_file        : str       = ""
    poll_interval_s  : float     = 5.0
