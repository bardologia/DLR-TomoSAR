from __future__ import annotations

from dataclasses import dataclass, field

from configuration.experiments.benchmark_config        import BenchmarkConfig
from configuration.experiments.cross_validation_config import CrossValidationConfig
from configuration.experiments.tuning_config           import TuningEntryConfig


@dataclass
class ExperimentEntryConfig:
    mode : str = "benchmark"

    benchmark : BenchmarkConfig       = field(default_factory=BenchmarkConfig)
    cv        : CrossValidationConfig = field(default_factory=CrossValidationConfig)
    tune      : TuningEntryConfig     = field(default_factory=TuningEntryConfig)
