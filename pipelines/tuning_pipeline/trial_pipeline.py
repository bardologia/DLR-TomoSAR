from __future__ import annotations

import optuna

from pipelines.training_pipeline.pipeline  import TrainingPipeline
from pipelines.tuning_pipeline.trial_trainer import TrialTrainer


class TrialPipeline(TrainingPipeline):
    def __init__(self, *args, trial: optuna.Trial, **kwargs) -> None:
        self._trial = trial
        super().__init__(*args, **kwargs)

    def _make_trainer(self, model, model_cfg, x_axis, norm_stats):
        return TrialTrainer(
            model      = model,
            model_cfg  = model_cfg,
            x_axis     = x_axis,
            config     = self.trainer_config,
            run_dir    = self.run_metadata.run_directory,
            logger     = self.logger,
            norm_stats = norm_stats,
            trial      = self._trial,
        )
