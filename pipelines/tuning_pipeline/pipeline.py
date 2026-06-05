from __future__ import annotations

from pathlib import Path

import optuna

from configuration.tuning_config      import TuningConfig
from pipelines.tuning_pipeline.tuners import Phase1Tuner, Phase2Tuner


class TuningPipeline:
    def __init__(
        self,
        model_name          : str,
        model_config_cls,
        base_trainer_config,
        base_dataset_config,
        tune_cfg            : TuningConfig,
        log_dir             : str,
        logger,
    ) -> None:
    
        self.model_name          = model_name
        self.model_config_cls    = model_config_cls
        self.base_trainer_config = base_trainer_config
        self.base_dataset_config = base_dataset_config
        self.tune_cfg            = tune_cfg
        self.log_dir             = log_dir
        self.logger              = logger

    def run_phase1(self, study: optuna.Study, n_trials: int) -> None:
        tuner = Phase1Tuner(
            model_name          = self.model_name,
            model_config_cls    = self.model_config_cls,
            base_trainer_config = self.base_trainer_config,
            base_dataset_config = self.base_dataset_config,
            tune_cfg            = self.tune_cfg.phase1,
            log_dir             = str(Path(self.log_dir) / "phase1"),
            logger              = self.logger,
            emit_trial_docs     = self.tune_cfg.emit_trial_docs,
        )
        tuner.run(study, n_trials)

    def run_phase2(self, study: optuna.Study, n_trials: int, best_phase1_params: dict) -> None:
        tuner = Phase2Tuner(
            model_name          = self.model_name,
            model_config_cls    = self.model_config_cls,
            base_trainer_config = self.base_trainer_config,
            base_dataset_config = self.base_dataset_config,
            tune_cfg            = self.tune_cfg.phase2,
            best_phase1_params  = best_phase1_params,
            log_dir             = str(Path(self.log_dir) / "phase2"),
            logger              = self.logger,
            emit_trial_docs     = self.tune_cfg.emit_trial_docs,
        )
        tuner.run(study, n_trials)
