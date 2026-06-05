from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from configuration.benchmark_config import (
    BenchmarkPathsConfig,
    ComparisonReportConfig,
    InferenceQueueConfig,
    TrainingQueueConfig,
)


@dataclass
class FoldConfig:
    n_folds       : int = 10
    azimuth_start : int = 1000
    azimuth_end   : int = 16000


@dataclass
class CrossValidationConfig:
    model_name      : str  = "resunet"
    model_overrides : dict = field(default_factory=dict)

    paths      : BenchmarkPathsConfig   = field(default_factory=lambda: BenchmarkPathsConfig(log_base_dir=Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/logs/cross_validation")))
    folds      : FoldConfig             = field(default_factory=FoldConfig)
    training   : TrainingQueueConfig    = field(default_factory=TrainingQueueConfig)
    inference  : InferenceQueueConfig   = field(default_factory=InferenceQueueConfig)
    comparison : ComparisonReportConfig = field(default_factory=ComparisonReportConfig)

    inference_splits : list[str] = field(default_factory=lambda: ["val", "test"])

    gpus            : list[int]  = field(default_factory=lambda: [2, 3])
    run_tag         : str | None = None
    resume          : bool       = True
    seed            : int        = 0
    n_gaussians     : int        = 5
    poll_interval_s : float      = 5.0
