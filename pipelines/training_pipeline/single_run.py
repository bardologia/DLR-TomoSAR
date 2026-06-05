from __future__ import annotations

from configuration.single_train_config import SingleTrainConfig
from pipelines.benchmark_pipeline.config_factory import ConfigFactory
from tools.loss_scale_probe import LossScaleProbeConfig


class SingleTrainRunner:
    def __init__(self, config: SingleTrainConfig) -> None:
        self.config  = config
        self.factory = ConfigFactory(config)

    def run(self):
        from models import CONFIG_REGISTRY
        from pipelines.training_pipeline.pipeline import TrainingPipeline

        trainer_config            = self.factory.training_trainer_config(logdir=self.config.logdir)
        trainer_config.curriculum = self.config.curriculum
        trainer_config.overfit    = self.config.overfit
        trainer_config.geometry   = self.config.geometry

        model_config = CONFIG_REGISTRY[self.config.model_name]()
        for attribute, value in self.config.model_overrides.items():
            setattr(model_config, attribute, value)

        pipeline = TrainingPipeline(
            trainer_config = trainer_config,
            dataset_config = self.factory.training_dataset_config(),
            model_name     = self.config.model_name,
            model_config   = model_config,
            seed           = self.config.seed,
            run_name       = self.config.run_name,
        )

        return pipeline.run(probe_config=self._probe_config())

    def _probe_config(self) -> LossScaleProbeConfig:
        return LossScaleProbeConfig(
            enabled        = self.config.probe_enabled,
            n_batches      = self.config.probe_n_batches,
            reference      = self.config.probe_reference,
            exit_after     = self.config.probe_exit_after,
            enabled_losses = {},
        )
