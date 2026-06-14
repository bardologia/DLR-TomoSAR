from __future__ import annotations

import optuna

from pipelines.training.autoencoder.autoencoder_trainer import ProfileAeTrainer
from pipelines.training.autoencoder.pipeline            import ProfileAePipeline
from pipelines.training.backbone.pipeline import TrainingPipeline
from pipelines.training.backbone.trainer  import Trainer


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


class TrialProfileAeTrainer(ProfileAeTrainer):
    def __init__(self, *args, trial: optuna.Trial, **kwargs) -> None:
        self._trial = trial
        super().__init__(*args, **kwargs)

    def _after_eval(self, val_loss: float, epoch: int) -> None:
        super()._after_eval(val_loss, epoch)
        self._trial.report(val_loss, epoch)
        if self._trial.should_prune():
            raise optuna.exceptions.TrialPruned()


class TrialProfileAePipeline(ProfileAePipeline):
    def __init__(self, *args, trial: optuna.Trial, **kwargs) -> None:
        self._trial = trial
        super().__init__(*args, **kwargs)

    def _make_trainer(self, run_meta, logger, model, x_axis):
        return TrialProfileAeTrainer(model, self.autoencoder_cfg, x_axis, self.trainer_config, run_meta.run_directory, logger, trial=self._trial)
