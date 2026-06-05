from __future__ import annotations

import copy
from pathlib import Path

import optuna


class ParamSampler:
    def sample(self, trial: optuna.Trial, space: dict) -> dict:
        sampled = {}

        for name, spec in space.items():
            kind = spec["type"]

            if kind == "float":
                sampled[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))

            elif kind == "categorical":
                sampled[name] = trial.suggest_categorical(name, spec["choices"])

            elif kind == "indexed_categorical":
                idx           = trial.suggest_categorical(name + "__idx", list(range(len(spec["choices"]))))
                sampled[name] = spec["choices"][idx]

        return sampled


class BaseTuner:
    run_name_prefix : str = ""
    section_title   : str = ""
    config_title    : str = ""
    error_label     : str = ""

    def __init__(
        self,
        model_name          : str,
        model_config_cls,
        base_trainer_config,
        base_dataset_config,
        tune_cfg,
        log_dir             : str,
        logger,
        emit_trial_docs     : bool = False,
    ) -> None:
        self.model_name          = model_name
        self.model_config_cls    = model_config_cls
        self.base_trainer_config = base_trainer_config
        self.base_dataset_config = base_dataset_config
        self.tune_cfg            = tune_cfg
        self.log_dir             = log_dir
        self.logger              = logger
        self.emit_trial_docs     = emit_trial_docs

        self.sampler = ParamSampler()

    def _apply_params(self, trial: optuna.Trial, model_config) -> None:
        raise NotImplementedError

    def _extra_config_rows(self) -> dict:
        return {}

    def _objective(self, trial: optuna.Trial) -> float:
        from pipelines.tuning_pipeline.trial import TrialPipeline

        model_config = self.model_config_cls()
        self._apply_params(trial, model_config)

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
            run_name       = f"{self.run_name_prefix}{trial.number:04d}",
            trial          = trial,
            emit_docs      = self.emit_trial_docs,
        )

        try:
            _, _, best_val_loss = pipeline.run()
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as exc:
            self.logger.error(f"{self.error_label} trial {trial.number} raised: {exc}")
            raise optuna.exceptions.TrialPruned()

        return best_val_loss

    def run(self, study: optuna.Study, n_trials: int) -> None:
        self.logger.section(f"[{self.section_title} — {self.model_name}]")

        rows = {
            "Trials (this worker)" : n_trials,
            "Epochs / trial"       : self.tune_cfg.n_epochs,
            "Early-stop patience"  : self.tune_cfg.early_stop_patience,
        }
        rows.update(self._extra_config_rows())
        rows["Log dir"] = self.log_dir

        self.logger.kv_table(rows, title=self.config_title)

        study.optimize(self._objective, n_trials=n_trials, gc_after_trial=True)


class Phase1Tuner(BaseTuner):
    run_name_prefix = "phase1_trial_"
    section_title   = "Phase 1 Tuner"
    config_title    = "Phase 1 config"
    error_label     = "Phase-1"

    def _apply_params(self, trial: optuna.Trial, model_config) -> None:
        sampled = self.sampler.sample(trial, self.model_config_cls.tunable_lr_params())
        for k, v in sampled.items():
            setattr(model_config, k, v)


class Phase2Tuner(BaseTuner):
    run_name_prefix = "phase2_trial_"
    section_title   = "Phase 2 Tuner"
    config_title    = "Phase 2 config"
    error_label     = "Phase-2"

    def __init__(
        self,
        model_name          : str,
        model_config_cls,
        base_trainer_config,
        base_dataset_config,
        tune_cfg,
        best_phase1_params  : dict,
        log_dir             : str,
        logger,
        emit_trial_docs     : bool = False,
    ) -> None:
        super().__init__(
            model_name          = model_name,
            model_config_cls    = model_config_cls,
            base_trainer_config = base_trainer_config,
            base_dataset_config = base_dataset_config,
            tune_cfg            = tune_cfg,
            log_dir             = log_dir,
            logger              = logger,
            emit_trial_docs     = emit_trial_docs,
        )
        self.best_phase1_params = best_phase1_params

    def _apply_params(self, trial: optuna.Trial, model_config) -> None:
        arch_sampled = self.sampler.sample(trial, self.model_config_cls.tunable_arch_params())

        for k, v in self.best_phase1_params.items():
            if hasattr(model_config, k):
                setattr(model_config, k, v)
        for k, v in arch_sampled.items():
            setattr(model_config, k, v)

    def _extra_config_rows(self) -> dict:
        return {"Phase-1 best params": str(self.best_phase1_params)}
