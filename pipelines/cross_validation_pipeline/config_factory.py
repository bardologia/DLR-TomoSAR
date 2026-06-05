from __future__ import annotations

from pathlib import Path

from configuration.cross_validation_config import CrossValidationConfig
from configuration.dataset_config import DatasetConfiguration
from configuration.inference_config import InferenceConfig
from pipelines.benchmark_pipeline.config_factory import ConfigFactory
from pipelines.cross_validation_pipeline.fold_planner import FoldPlanner


class FoldConfigFactory(ConfigFactory):
    def __init__(self, config: CrossValidationConfig) -> None:
        super().__init__(config)
        self._planner: FoldPlanner | None = None

    def planner(self) -> FoldPlanner:
        if self._planner is None:
            crop          = self.global_crop()
            self._planner = FoldPlanner(self.config, range_start=crop.range_start, range_end=crop.range_end)
        return self._planner

    def fold_dataset_config(self, fold_index: int) -> DatasetConfiguration:
        dataset_config               = self.training_dataset_config()
        dataset_config.split_regions = self.planner().plan(fold_index).split_regions

        return dataset_config

    def fold_inference_config(self, run_directory: Path, split: str) -> InferenceConfig:
        inference_config               = self.inference_config(run_directory)
        inference_config.split         = split
        inference_config.output_subdir = split

        return inference_config
