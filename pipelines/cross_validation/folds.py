from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from configuration.experiments.cross_validation_config       import CrossValidationConfig
from configuration.inference.inference_config              import InferenceConfig
from pipelines.shared.config_factory import ConfigFactory
from tools.data.regions                               import CropRegion, SplitRegions


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
