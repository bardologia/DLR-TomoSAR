from __future__ import annotations

from pathlib import Path

import torch

from pipelines.shared.training.pretrain_preflight import PretrainPreflight
from tools.training.pretraining          import PretrainContext, TrainStepMemoryProbe, TrainerFeed


class SingleTrainRunner:
    def __init__(self, config) -> None:
        self.config = config

    @property
    def label(self) -> str:
        raise NotImplementedError

    def _pretrain_preflight(self) -> None:
        PretrainPreflight(
            pretrain_config = self.config.pretrain,
            training_config = self.config.training,
            build_context   = self._build_pretrain_context,
            logdir          = Path(self.config.logdir),
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

    def _build_pretrain_trainer(self, logger):
        work_dir = Path(self.config.logdir) / "pretrain" / "context"

        return self.pipeline_class(self.config).build_pretrain_trainer(work_dir, logger)

    def run(self):
        self._pretrain_preflight()

        return self.pipeline_class(self.config).run()
