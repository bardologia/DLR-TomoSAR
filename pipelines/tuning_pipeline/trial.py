from __future__ import annotations

import optuna

from pipelines.backbone_pipeline.pipeline import TrainingPipeline
from pipelines.backbone_pipeline.trainer  import Trainer


class TrialTrainer(Trainer):
    def __init__(self, *args, trial: optuna.Trial, emit_docs: bool = False, **kwargs) -> None:
        self._trial = trial
        super().__init__(*args, emit_docs=emit_docs, **kwargs)

    def _trial_callback(self, val_loss: float, epoch: int) -> None:
        self._trial.report(val_loss, epoch)
        if self._trial.should_prune():
            raise optuna.exceptions.TrialPruned()


class TrialPipeline(TrainingPipeline):
    def __init__(self, *args, trial: optuna.Trial, emit_docs: bool = False, **kwargs) -> None:
        self._trial     = trial
        self._emit_docs = emit_docs
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
            emit_docs  = self._emit_docs,
        )
