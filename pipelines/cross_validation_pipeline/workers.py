from __future__ import annotations

from configuration.cross_validation_config import CrossValidationConfig
from pipelines.benchmark_pipeline.workers import BenchmarkWorker
from pipelines.cross_validation_pipeline.config_factory import FoldConfigFactory


class CrossValidationWorker(BenchmarkWorker):
    def __init__(self, config: CrossValidationConfig, run_tag: str) -> None:
        super().__init__(config=config, run_tag=run_tag)
        self.factory = FoldConfigFactory(config)

    def fold_name(self, fold_index: int) -> str:
        return f"fold_{fold_index}"


class FoldTrainingWorker(CrossValidationWorker):
    def run(self, fold_index: int) -> None:
        from models import CONFIG_REGISTRY
        from pipelines.training_pipeline.pipeline import TrainingPipeline

        model_config = CONFIG_REGISTRY[self.config.model_name]()

        for attribute, value in self.config.model_overrides.items():
            setattr(model_config, attribute, value)

        pipeline = TrainingPipeline(
            trainer_config = self.factory.training_trainer_config(logdir=self.run_dir / "folds"),
            dataset_config = self.factory.fold_dataset_config(fold_index),
            model_name     = self.config.model_name,
            model_config   = model_config,
            seed           = self.config.seed,
            run_name       = self.fold_name(fold_index),
        )

        pipeline.run(probe_config=self._probe_config())


class FoldInferenceWorker(CrossValidationWorker):
    def run(self, fold_index: int, split: str) -> None:
        from pipelines.inference_pipeline.pipeline import InferencePipeline

        run_directory = self.run_dir / "folds" / self.fold_name(fold_index)

        pipeline = InferencePipeline(self.factory.fold_inference_config(run_directory, split))
        pipeline.run()
