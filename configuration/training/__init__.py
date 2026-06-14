import importlib

_EXPORTS = {
    "AutoencoderLossConfig": "autoencoder_config",
    "ProfileAeTrainerConfig": "autoencoder_config",
    "ProfileAeEntryConfig": "autoencoder_config",
    "SecondaryTrialsConfig": "backbone_config",
    "BackboneEntryConfig": "backbone_config",
    "EmbeddingLossConfig": "jepa_config",
    "JepaTrainerConfig": "jepa_config",
    "JepaDefaults": "jepa_config",
    "JepaEntryConfig": "jepa_config",
    "LossNormalizationConfig": "loss_config",
    "LossConfig": "loss_config",
    "LossCurriculumConfig": "loss_config",
    "OptimizerConfig": "optimization_config",
    "SchedulerConfig": "optimization_config",
    "WarmupConfig": "optimization_config",
    "EarlyStoppingConfig": "optimization_config",
    "GradientClipperConfig": "optimization_config",
    "IOConfig": "runtime_config",
    "OverfitConfig": "runtime_config",
    "TrainingLoopConfig": "runtime_config",
    "MemoryConfig": "runtime_config",
    "ResourceConfig": "runtime_config",
    "PermutationMetricsConfig": "runtime_config",
    "SharedSubConfigInheritance": "trainer_config",
    "TrainerConfig": "training_config",
}

__all__ = [
    "AutoencoderLossConfig",
    "ProfileAeTrainerConfig",
    "ProfileAeEntryConfig",
    "SecondaryTrialsConfig",
    "BackboneEntryConfig",
    "EmbeddingLossConfig",
    "JepaTrainerConfig",
    "JepaDefaults",
    "JepaEntryConfig",
    "LossNormalizationConfig",
    "LossConfig",
    "LossCurriculumConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "WarmupConfig",
    "EarlyStoppingConfig",
    "GradientClipperConfig",
    "IOConfig",
    "OverfitConfig",
    "TrainingLoopConfig",
    "MemoryConfig",
    "ResourceConfig",
    "PermutationMetricsConfig",
    "SharedSubConfigInheritance",
    "TrainerConfig",
]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
