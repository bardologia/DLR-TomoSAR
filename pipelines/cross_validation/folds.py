from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from configuration.experiments.cross_validation_config       import CrossValidationConfig
from configuration.inference.inference_config              import InferenceConfig
from pipelines.shared.config_factory import ConfigFactory
from pipelines.benchmark.results        import TrialCollector, TrialRecord
from pipelines.benchmark.workers        import BenchmarkWorker
from tools.data.regions                               import CropRegion, SplitRegions
from tools.monitoring.logger                                import Logger


@dataclass
class FoldPlan:
    fold_index    : int
    test_block    : int
    val_block     : int
    train_blocks  : list[int]
    split_regions : SplitRegions


class FoldPlanner:
    def __init__(self, config: CrossValidationConfig, range_start: int, range_end: int) -> None:
        folds = config.folds

        if folds.n_folds < 3:
            raise ValueError(f"n_folds must be >= 3 so that train, val, and test stay disjoint; got {folds.n_folds}")

        self.n_folds     = folds.n_folds
        self.range_start = range_start
        self.range_end   = range_end
        self.blocks      = self._partition(folds.azimuth_start, folds.azimuth_end, folds.n_folds)

    def _partition(self, azimuth_start: int, azimuth_end: int, n_folds: int) -> list[tuple[int, int]]:
        total = azimuth_end - azimuth_start
        if total < n_folds:
            raise ValueError(f"Azimuth extent {total} is smaller than n_folds {n_folds}")

        size   = total // n_folds
        bounds = [azimuth_start + index * size for index in range(n_folds)] + [azimuth_end]

        return [(bounds[index], bounds[index + 1]) for index in range(n_folds)]

    def _merge_adjacent(self, block_indices: list[int]) -> list[tuple[int, int]]:
        runs  = []
        start = block_indices[0]
        prev  = block_indices[0]

        for index in block_indices[1:]:
            if index == prev + 1:
                prev = index
                continue
            runs.append((start, prev))
            start = index
            prev  = index

        runs.append((start, prev))
        return runs

    def _run_region(self, run: tuple[int, int]) -> CropRegion:
        first_block, last_block = run
        return CropRegion(self.blocks[first_block][0], self.blocks[last_block][1], self.range_start, self.range_end)

    def _block_region(self, block_index: int) -> CropRegion:
        azimuth_start, azimuth_end = self.blocks[block_index]
        return CropRegion(azimuth_start, azimuth_end, self.range_start, self.range_end)

    def plan(self, fold_index: int) -> FoldPlan:
        if not 0 <= fold_index < self.n_folds:
            raise ValueError(f"fold_index must be in [0, {self.n_folds}); got {fold_index}")

        test_block   = fold_index
        val_block    = (fold_index + 1) % self.n_folds
        train_blocks = [index for index in range(self.n_folds) if index not in (test_block, val_block)]

        train_regions = [self._run_region(run) for run in self._merge_adjacent(train_blocks)]

        split_regions = SplitRegions(
            train = train_regions if len(train_regions) > 1 else train_regions[0],
            val   = self._block_region(val_block),
            test  = self._block_region(test_block),
        )

        return FoldPlan(
            fold_index    = fold_index,
            test_block    = test_block,
            val_block     = val_block,
            train_blocks  = train_blocks,
            split_regions = split_regions,
        )

    def plans(self) -> list[FoldPlan]:
        return [self.plan(fold_index) for fold_index in range(self.n_folds)]


class FoldNaming:
    @staticmethod
    def name(index: int) -> str:
        return f"fold_{index}"

    @staticmethod
    def index(name: str) -> int:
        return int(name.split("_")[-1])


class FoldConfigFactory(ConfigFactory):
    def __init__(self, config: CrossValidationConfig) -> None:
        super().__init__(config)
        self._planner: FoldPlanner | None = None

    def planner(self) -> FoldPlanner:
        if self._planner is None:
            crop          = self.global_crop()
            self._planner = FoldPlanner(self.config, range_start=crop.range_start, range_end=crop.range_end)
        return self._planner

    def fold_inference_config(self, run_directory: Path, split: str) -> InferenceConfig:
        inference_config               = self.inference_config(run_directory)
        inference_config.split         = split
        inference_config.output_subdir = split

        return inference_config


class FoldCollector(TrialCollector):
    def __init__(self, run_dir: Path, splits: list[str], logger: Logger) -> None:
        super().__init__(run_dir=run_dir, logger=logger)
        self.training_dir = run_dir / "folds"
        self.splits       = splits

    def collect_by_split(self) -> tuple[list[TrialRecord], dict[str, list[TrialRecord]]]:
        base_records = self.collect()

        records_by_split = {
            split: [self._split_view(record, split) for record in base_records]
            for split in self.splits
        }

        return base_records, records_by_split

    def _split_view(self, record: TrialRecord, split: str) -> TrialRecord:
        inference_dir = record.run_dir / "inference" / split

        if not (inference_dir / "metrics.json").exists():
            return replace(record, inference_dir=None, metrics={}, figures=[], animations=[], report_path=None)

        return replace(record, **self._inference_fields(inference_dir))

    def _inference_fields(self, inference_dir: Path) -> dict:
        report_path = inference_dir / "report.md"

        return {
            "inference_dir" : inference_dir,
            "metrics"       : self._load_json(inference_dir / "metrics.json"),
            "figures"       : sorted((inference_dir / "figures").glob("*.png")) if (inference_dir / "figures").is_dir() else [],
            "animations"    : sorted((inference_dir / "animations").glob("*.gif")) if (inference_dir / "animations").is_dir() else [],
            "report_path"   : report_path if report_path.exists() else None,
        }


class CrossValidationWorker(BenchmarkWorker):
    def __init__(self, config: CrossValidationConfig, run_tag: str) -> None:
        super().__init__(config=config, run_tag=run_tag)
        self.factory = FoldConfigFactory(config)

    def fold_name(self, fold_index: int) -> str:
        return FoldNaming.name(fold_index)


class FoldTrainingWorker(CrossValidationWorker):
    def _run_backbone(self, fold_index: int, split_regions: SplitRegions) -> None:
        from models import CONFIG_REGISTRY
        from pipelines.backbone.training.pipeline import TrainingPipeline

        model_config = CONFIG_REGISTRY[self.config.model_name]()

        for attribute, value in self.config.model_overrides.items():
            setattr(model_config, attribute, value)

        dataset_config               = self.factory.training_dataset_config()
        dataset_config.split_regions = split_regions

        pipeline = TrainingPipeline(
            trainer_config = self.factory.training_trainer_config(logdir=self.run_dir / "folds"),
            dataset_config = dataset_config,
            model_name     = self.config.model_name,
            model_config   = model_config,
            seed           = self.config.seed,
            run_name       = self.fold_name(fold_index),
        )

        pipeline.run(probe_config=self._probe_config())

    def _run_jepa(self, fold_index: int, split_regions: SplitRegions) -> None:
        from pipelines.jepa.training.pipeline import TrainingPipeline

        TrainingPipeline(self._jepa_entry_config(self.fold_name(fold_index)), split_regions=split_regions).run()

    def _jepa_entry_config(self, run_name: str):
        from configuration.training.jepa_config import JepaEntryConfig

        cv   = self.config
        jepa = cv.jepa

        return JepaEntryConfig(
            run_name        = run_name,
            model_name      = cv.model_name,
            seed            = cv.seed,
            n_gaussians     = cv.n_gaussians,
            logdir          = self.run_dir / "folds",
            model_overrides = cv.model_overrides,
            stage_a_run     = jepa.stage_a_run,
            stage_a_mode    = jepa.stage_a_mode,
            target_provider = jepa.target_provider,
            embedding_loss  = jepa.embedding_loss,
            overfit         = cv.overfit,
            geometry        = cv.geometry,
            paths           = cv.paths,
            training        = cv.training,
        )

    def _run_autoencoder(self, fold_index: int, split_regions: SplitRegions) -> None:
        from pipelines.profile_autoencoder.training.pipeline import TrainingPipeline

        TrainingPipeline(self._ae_entry_config(self.fold_name(fold_index)), split_regions=split_regions).run()

    def _ae_entry_config(self, run_name: str):
        from configuration.training.autoencoder_config import ProfileAeEntryConfig

        cv = self.config
        ae = cv.autoencoder

        return ProfileAeEntryConfig(
            run_name        = run_name,
            seed            = cv.seed,
            n_gaussians     = cv.n_gaussians,
            logdir          = self.run_dir / "folds",
            pixel_subsample = ae.pixel_subsample,
            keep_empty_frac = ae.keep_empty_frac,
            ae_model_name   = ae.ae_model_name,
            autoencoder     = ae.autoencoder,
            ae_loss         = ae.ae_loss,
            overfit         = cv.overfit,
            geometry        = cv.geometry,
            paths           = cv.paths,
            training        = cv.training,
        )

    def run(self, fold_index: int) -> None:
        dispatch = {
            "backbone"    : self._run_backbone,
            "jepa"        : self._run_jepa,
            "autoencoder" : self._run_autoencoder,
        }

        training_type = self.config.training_type
        if training_type not in dispatch:
            raise ValueError(f"Unknown training_type '{training_type}', expected one of {sorted(dispatch)}")

        split_regions = self.factory.planner().plan(fold_index).split_regions
        dispatch[training_type](fold_index, split_regions)


class FoldInferenceWorker(CrossValidationWorker):
    def run(self, fold_index: int, split: str) -> None:
        from pipelines.backbone.inference.pipeline import InferencePipeline

        run_directory = self.run_dir / "folds" / self.fold_name(fold_index)

        pipeline = InferencePipeline(self.factory.fold_inference_config(run_directory, split))
        pipeline.run()
