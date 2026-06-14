from __future__ import annotations

from dataclasses import dataclass, field

from configuration.sar.gaussian_config          import GaussianConfig
from configuration.sar.geometry_config          import GeometryConfig
from configuration.training.loss_config         import LossCurriculumConfig
from configuration.training.optimization_config import EarlyStoppingConfig, GradientClipperConfig, OptimizerConfig, SchedulerConfig, WarmupConfig
from configuration.training.runtime_config      import IOConfig, MemoryConfig, OverfitConfig, PermutationMetricsConfig, ResourceConfig, TrainingLoopConfig


@dataclass
class TrainerConfig:
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
