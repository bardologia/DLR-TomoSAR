from __future__ import annotations

import copy
from pathlib import Path

import optuna

from configuration.tuning_config        import Phase2TuneConfig
from pipelines.tuning_pipeline.phase1_tuner import _sample_params


class Phase2Tuner:
    def __init__(
        self,
        model_name          : str,
        model_config_cls,
        base_trainer_config,
        base_dataset_config,
        tune_cfg            : Phase2TuneConfig,
        best_phase1_params  : dict,
        log_dir             : str,
        logger,
    ) -> None:
        self.model_name          = model_name
        self.model_config_cls    = model_config_cls
        self.base_trainer_config = base_trainer_config
        self.base_dataset_config = base_dataset_config
        self.tune_cfg            = tune_cfg
        self.best_phase1_params  = best_phase1_params
        self.log_dir             = log_dir
        self.logger              = logger

    def _objective(self, trial: optuna.Trial) -> float:
        from pipelines.tuning_pipeline.trial_pipeline import TrialPipeline

        arch_sampled = _sample_params(trial, self.model_config_cls.tunable_arch_params())

        model_config = self.model_config_cls()
        for k, v in self.best_phase1_params.items():
            if hasattr(model_config, k):
                setattr(model_config, k, v)
        for k, v in arch_sampled.items():
            setattr(model_config, k, v)

        trainer_cfg = copy.deepcopy(self.base_trainer_config)
        dataset_cfg = copy.deepcopy(self.base_dataset_config)

        trainer_cfg.training.epochs         = self.tune_cfg.n_epochs
        trainer_cfg.scheduler.epochs        = self.tune_cfg.n_epochs
        trainer_cfg.early_stopping.patience = self.tune_cfg.early_stop_patience
        trainer_cfg.io.logdir               = str(Path(self.log_dir) / f"trial_{trial.number:04d}")

        pipeline = TrialPipeline(
            trainer_config = trainer_cfg,
            dataset_config = dataset_cfg,
            model_name     = self.model_name,
            model_config   = model_config,
            seed           = trial.number,
            run_name       = f"phase2_trial_{trial.number:04d}",
            trial          = trial,
        )

        try:
            _, _, best_val_loss = pipeline.run()
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as exc:
            self.logger.error(f"Phase-2 trial {trial.number} raised: {exc}")
            raise optuna.exceptions.TrialPruned()

        return best_val_loss

    def run(self, study: optuna.Study, n_trials: int) -> None:
        self.logger.section(f"[Phase 2 Tuner — {self.model_name}]")
        self.logger.kv_table({
            "Trials (this worker)" : n_trials,
            "Epochs / trial"       : self.tune_cfg.n_epochs,
            "Early-stop patience"  : self.tune_cfg.early_stop_patience,
            "Phase-1 best params"  : str(self.best_phase1_params),
            "Log dir"              : self.log_dir,
        }, title="Phase 2 config")

        study.optimize(self._objective, n_trials=n_trials, gc_after_trial=True)
