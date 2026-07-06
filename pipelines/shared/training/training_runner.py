from __future__ import annotations

from pathlib import Path

import torch

from pipelines.shared.config.run_metadata         import TrainingRunMetadata
from pipelines.shared.training.pretrain_preflight import PretrainPreflight
from tools.training.pretraining          import PretrainContext, TrainStepMemoryProbe, TrainerFeed


class SingleTrainRunner:
    def __init__(self, config) -> None:
        self.config        = config
        self.run_directory = None

    @property
    def label(self) -> str:
        raise NotImplementedError

    def _resolve_run_name(self) -> str:
        raise NotImplementedError

    def _resolve_run_directory(self) -> None:
        self.config.run_name = self._resolve_run_name()
        self.run_directory   = Path(self.config.logdir) / self.config.run_name

    def _pretrain_preflight(self) -> None:
        PretrainPreflight(
            pretrain_config = self.config.pretrain,
            training_config = self.config.training,
            build_context   = self._build_pretrain_context,
            run_directory   = self.run_directory,
            label           = self.label,
        ).run()

    def _build_pretrain_context(self, logger, device) -> PretrainContext:
        trainer, dataset, model = self._build_pretrain_trainer(logger)
        trainer.model.train()

        feed       = TrainerFeed(trainer)
        context_gb = TrainStepMemoryProbe.measure_context(device)

        return PretrainContext(
            dataset        = dataset,
            model          = model,
            to_model_input = feed.to_model_input,
            forward_loss   = feed.forward_loss,
            trial_step     = TrainStepMemoryProbe(trainer, dataset, self.config.pretrain.measure_steps, device, context_gb),
            device         = device,
            use_amp        = trainer.use_amp,
            context_gb     = context_gb,
            on_oom         = lambda: self._release(trainer),
        )

    def _build_pretrain_trainer(self, logger):
        raise NotImplementedError

    def _release(self, trainer) -> None:
        trainer.optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self):
        raise NotImplementedError


class EntryConfigTrainRunner(SingleTrainRunner):
    pipeline_class = None

    def _resolve_run_name(self) -> str:
        return TrainingRunMetadata.resolve_name(self.pipeline_class.run_label, self.config.run_name)

    def _build_pretrain_trainer(self, logger):
        work_dir = self.run_directory / "pretrain" / "context"

        return self.pipeline_class(self.config).build_pretrain_trainer(work_dir, logger)

    def run(self):
        self._resolve_run_directory()
        self._pretrain_preflight()

        return self.pipeline_class(self.config).run()
