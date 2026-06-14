from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.experiments.benchmark_config import BenchmarkPathsConfig, TrainingQueueConfig
from configuration.training.runtime_config      import OverfitConfig


def _default_ae_loss():
    from configuration.training.autoencoder_config import AutoencoderLossConfig

    return AutoencoderLossConfig()


def _default_embedding_loss():
    from configuration.training.jepa_config import EmbeddingLossConfig

    return EmbeddingLossConfig()


@dataclass
class JepaTuneConfig:
    stage_a_logdir  : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/jepa_stage_a")
    stage_a_run     : str | None = None
    stage_a_mode    : str        = "frozen"
    target_provider : str        = "stopgrad"
    embedding_loss  : object     = field(default_factory=_default_embedding_loss)


@dataclass
class TuningConfig:
    n_trials : int = 100
    n_epochs : int = 30

    base_seed               : int  = 42

    early_stop_patience     : int  = 8

    pruner_n_startup_trials : int = 8
    pruner_n_warmup_steps   : int = 8

    emit_trial_docs  : bool = False
    emit_study_plots : bool = False


@dataclass
class TuningEntryConfig:
    training_type : str = "backbone"

    paths  : BenchmarkPathsConfig = field(default_factory=lambda: BenchmarkPathsConfig(log_base_dir=Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/tuning")))
    tuning : TuningConfig         = field(default_factory=TuningConfig)

    gpus         : list[int]  = field(default_factory=lambda: [0, 1, 2, 3])
    skip_models  : list[str]  = field(default_factory=list)
    run_tag      : str | None = None
    batch_size   : int        = 256
    num_workers  : int        = 4
    warmup_steps : int        = 200
    eta_min      : float      = 1e-6

    n_gaussians     : int                 = 5
    training        : TrainingQueueConfig = field(default_factory=TrainingQueueConfig)
    overfit         : OverfitConfig       = field(default_factory=OverfitConfig)
    ae_loss         : object              = field(default_factory=_default_ae_loss)
    jepa            : JepaTuneConfig       = field(default_factory=JepaTuneConfig)
    pixel_subsample : float               = 1.0
    keep_empty_frac : float               = 0.05
