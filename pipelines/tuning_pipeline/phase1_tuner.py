from __future__ import annotations

import optuna

from pipelines.tuning_pipeline.base_tuner import BaseTuner


class Phase1Tuner(BaseTuner):
    run_name_prefix = "phase1_trial_"
    section_title   = "Phase 1 Tuner"
    config_title    = "Phase 1 config"
    error_label     = "Phase-1"

    def _apply_params(self, trial: optuna.Trial, model_config) -> None:
        sampled = self.sampler.sample(trial, self.model_config_cls.tunable_lr_params())
        for k, v in sampled.items():
            setattr(model_config, k, v)
