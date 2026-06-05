from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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

        results = pipeline.run(probe_config=self._probe_config())

        if self.config.infer_after:
            self._run_inference(pipeline.run_metadata.run_directory)

        return results

    def _run_inference(self, run_directory: Path):
        import gc

        import torch

        from pipelines.inference_pipeline.pipeline import InferencePipeline

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inference_config = replace(self.config.inference, run_directory=Path(run_directory), output_subdir=None)

        return InferencePipeline(inference_config).run()

    def _probe_config(self) -> LossScaleProbeConfig:
        return LossScaleProbeConfig(
            enabled        = self.config.probe_enabled,
            n_batches      = self.config.probe_n_batches,
            reference      = self.config.probe_reference,
            exit_after     = self.config.probe_exit_after,
            enabled_losses = {},
        )
