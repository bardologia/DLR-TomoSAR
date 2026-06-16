from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.benchmark.general import (
    BenchmarkPathsConfig,
    ComparisonReportConfig,
    InferenceQueueConfig,
    TrainingQueueConfig,
)
from configuration.cross_validation.jepa                import JepaCvConfig
from configuration.cross_validation.profile_autoencoder import AeCvConfig
from configuration.sar.geometry_config                  import GeometryConfig
from configuration.training.general.runtime             import OverfitConfig


@dataclass
class FoldConfig:
    n_folds       : int = 10
    azimuth_start : int = 1000
    azimuth_end   : int = 16000


@dataclass
class CrossValidationConfig:
    training_type   : str  = "backbone"

    backbone_name   : str  = "resunet"
    model_overrides : dict = field(default_factory=dict)

    paths      : BenchmarkPathsConfig   = field(default_factory=lambda: BenchmarkPathsConfig(log_base_dir=Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/cross_validation")))
    folds      : FoldConfig             = field(default_factory=FoldConfig)
    training   : TrainingQueueConfig    = field(default_factory=TrainingQueueConfig)
    inference  : InferenceQueueConfig   = field(default_factory=InferenceQueueConfig)
    comparison : ComparisonReportConfig = field(default_factory=ComparisonReportConfig)

    geometry    : GeometryConfig = field(default_factory=GeometryConfig)
    overfit     : OverfitConfig  = field(default_factory=OverfitConfig)
    jepa        : JepaCvConfig   = field(default_factory=JepaCvConfig)
    autoencoder : AeCvConfig     = field(default_factory=AeCvConfig)

    inference_splits : list[str] = field(default_factory=lambda: ["val", "test"])

    gpus            : list[int]  = field(default_factory=lambda: [2, 3])
    run_tag         : str | None = None
    resume          : bool       = True
    seed            : int        = 0
    n_gaussians     : int        = 5
    poll_interval_s : float      = 5.0

    def runs_inference(self) -> bool:
        return self.training_type != "profile_autoencoder"
