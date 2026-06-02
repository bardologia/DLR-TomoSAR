from __future__ import annotations

import optuna

from pipelines.tuning_pipeline.base_tuner import BaseTuner


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
