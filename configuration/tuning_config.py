from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from configuration.benchmark_config import BenchmarkPathsConfig


@dataclass
class Phase1TuneConfig:
    n_trials                : int   = 100
    n_epochs                : int   = 30
    
    early_stop_patience     : int   = 8
    
    lr_low                  : float = 1e-5
    lr_high                 : float = 1e-2
    
    wd_low                  : float = 1e-6
    wd_high                 : float = 1e-1
    
    pruner_n_startup_trials : int   = 8
    pruner_n_warmup_steps   : int   = 8


@dataclass
class Phase2TuneConfig:
    n_trials                : int   = 100
    n_epochs                : int   = 30
   
    early_stop_patience     : int   = 10
   
    pruner_n_startup_trials : int   = 8
    pruner_n_warmup_steps   : int   = 8


@dataclass
class TuningConfig:
    phase1            : Phase1TuneConfig       = field(default_factory=Phase1TuneConfig)
    phase2            : Phase2TuneConfig       = field(default_factory=Phase2TuneConfig)
    study_storage_dir : str                    = "/ste/rnd/User/vice_vi/DLR-TomoSAR/logs/tuning"
    n_gpus            : int                    = 4
    emit_trial_docs   : bool                   = False


@dataclass
class TuningEntryConfig:
    paths  : BenchmarkPathsConfig = field(default_factory=lambda: BenchmarkPathsConfig(log_base_dir=Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/logs/tuning")))
    tuning : TuningConfig         = field(default_factory=TuningConfig)

    gpus         : list[int]  = field(default_factory=lambda: [0, 1, 2, 3])
    skip_models  : list[str]  = field(default_factory=list)
    run_tag      : str | None = None
    batch_size   : int        = 256
    num_workers  : int        = 4
    warmup_steps : int        = 200
    eta_min      : float      = 1e-6
