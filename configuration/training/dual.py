from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.training.general.run         import TrainingPathsConfig, TrainingQueueConfig
from configuration.dataset                      import AugmentationConfig, InputConfig
from configuration.inference.general            import InferenceConfig
from configuration.sar.geometry_config          import GeometryConfig
from configuration.normalization.general        import NormalizationConfig
from configuration.training.backbone            import _default_inference, default_curriculum
from configuration.training.general.loss        import LossCurriculumConfig
from configuration.training.general.runtime     import OverfitCheckConfig
from configuration.training.general.pretraining import PretrainConfig


@dataclass
class DualEntryConfig:
    run_name        : str | None = None
    model_name      : str        = "dual_resunet"
    gpu             : int        = 0
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=list)
    logdir          : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/dual")
    model_overrides : dict       = field(default_factory=dict)

    paths         : TrainingPathsConfig  = field(default_factory=TrainingPathsConfig)
    training      : TrainingQueueConfig  = field(default_factory=TrainingQueueConfig)
    pretrain      : PretrainConfig       = field(default_factory=PretrainConfig)
    curriculum    : LossCurriculumConfig = field(default_factory=default_curriculum)
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
