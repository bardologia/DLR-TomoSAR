from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from configuration.benchmark_config import BenchmarkPathsConfig, TrainingQueueConfig
from configuration.training_config import GeometryConfig, LossConfig, LossCurriculumConfig, OverfitConfig


def _default_curriculum() -> LossCurriculumConfig:
    return LossCurriculumConfig(
        enabled  = False,
        warmup   = LossConfig(use_param_l1=True, weight_param_l1=1.0),
        complete = LossConfig(use_param_l1=True, weight_param_l1=1.0),
    )


@dataclass
class SingleTrainConfig:
    run_name        : str | None = None
    model_name      : str        = "resunet"
    gpu             : int        = 0
    seed            : int        = 0
    n_gaussians     : int        = 5
    logdir          : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/logs/test")
    model_overrides : dict       = field(default_factory=dict)

    paths      : BenchmarkPathsConfig  = field(default_factory=BenchmarkPathsConfig)
    training   : TrainingQueueConfig   = field(default_factory=TrainingQueueConfig)
    curriculum : LossCurriculumConfig  = field(default_factory=_default_curriculum)
    overfit    : OverfitConfig         = field(default_factory=OverfitConfig)
    geometry   : GeometryConfig        = field(default_factory=GeometryConfig)

    probe_enabled    : bool = False
    probe_n_batches  : int  = 1000
    probe_reference  : str  = "param_l1"
    probe_exit_after : bool = True


@dataclass
class BatchTrainConfig:
    gpus            : list[int]         = field(default_factory=lambda: [0, 1, 3])
    poll_interval_s : float             = 5.0
    base            : SingleTrainConfig = field(default_factory=SingleTrainConfig)
