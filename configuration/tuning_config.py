from __future__ import annotations

from dataclasses import dataclass, field


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
class SinglePhaseTuneConfig:
    n_trials                : int   = 500
    n_epochs                : int   = 60

    early_stop_patience     : int   = 10

    pruner_n_startup_trials : int   = 20
    pruner_n_warmup_steps   : int   = 20


@dataclass
class TuningConfig:
    phase1            : Phase1TuneConfig       = field(default_factory=Phase1TuneConfig)
    phase2            : Phase2TuneConfig       = field(default_factory=Phase2TuneConfig)
    single_phase      : SinglePhaseTuneConfig  = field(default_factory=SinglePhaseTuneConfig)
    study_storage_dir : str                    = "/ste/rnd/User/vice_vi/DLR-TomoSAR/logs/tuning"
    n_gpus            : int                    = 4
    emit_trial_docs   : bool                   = False
