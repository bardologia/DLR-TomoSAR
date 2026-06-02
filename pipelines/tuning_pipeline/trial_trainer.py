from __future__ import annotations

import optuna

from pipelines.training_pipeline.trainer import Trainer


class TrialTrainer(Trainer):
    def __init__(self, *args, trial: optuna.Trial, emit_docs: bool = False, **kwargs) -> None:
        self._trial = trial
        super().__init__(*args, emit_docs=emit_docs, **kwargs)

    def _trial_callback(self, val_loss: float, epoch: int) -> None:
        self._trial.report(val_loss, epoch)
        if self._trial.should_prune():
            raise optuna.exceptions.TrialPruned()
