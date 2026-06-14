import importlib

_EXPORTS = {
    "BenchmarkPathsConfig": "benchmark_config",
    "OverfitGateConfig": "benchmark_config",
    "SizeMatchConfig": "benchmark_config",
    "TrainingQueueConfig": "benchmark_config",
    "InferenceQueueConfig": "benchmark_config",
    "ComparisonReportConfig": "benchmark_config",
    "BenchmarkConfig": "benchmark_config",
    "FoldConfig": "cross_validation_config",
    "JepaCvConfig": "cross_validation_config",
    "AeCvConfig": "cross_validation_config",
    "CrossValidationConfig": "cross_validation_config",
    "TuningConfig": "tuning_config",
    "TuningEntryConfig": "tuning_config",
}

__all__ = [
    "BenchmarkPathsConfig",
    "OverfitGateConfig",
    "SizeMatchConfig",
    "TrainingQueueConfig",
    "InferenceQueueConfig",
    "ComparisonReportConfig",
    "BenchmarkConfig",
    "FoldConfig",
    "JepaCvConfig",
    "AeCvConfig",
    "CrossValidationConfig",
    "TuningConfig",
    "TuningEntryConfig",
]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
