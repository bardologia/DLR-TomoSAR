from __future__ import annotations

import math
from dataclasses import dataclass

from configuration.patch_sweep             import PatchSweepConfig
from pipelines.shared.model.model_builder  import ModelBuilder
from tools.baselines                       import TrackBaselines


@dataclass(frozen=True)
class SweepUnit:
    track_count             : int
    patch_size              : int
    patch_stride            : int
    batch_size              : int
    lr_reference_batch_size : int
    secondary_labels        : tuple

    @property
    def name(self) -> str:
        return f"n{self.track_count:02d}-p{self.patch_size:03d}"


class ArchitecturePatchStep:

    PYRAMID_BACKBONES = frozenset({"unet", "unet_skip", "resunet", "attention_unet"})

    def __init__(self, backbone_name: str, backbone_head: str, model_overrides: dict) -> None:
        self.backbone_name   = backbone_name
        self.backbone_head   = backbone_head
        self.model_overrides = model_overrides

    def resolve(self) -> int:
        if self.backbone_name not in self.PYRAMID_BACKBONES:
            raise ValueError(f"Backbone '{self.backbone_name}' is not a verified one-halving-per-stage U-Net ({sorted(self.PYRAMID_BACKBONES)}); the admissible patch step cannot be derived from the architecture, set patch.step explicitly")

        model_config = ModelBuilder.config_from_registry(self.backbone_name, self.model_overrides, head=self.backbone_head)
        features     = model_config.features

        if not features:
            raise ValueError(f"Backbone '{self.backbone_name}' has an empty encoder feature pyramid; set patch.step explicitly")

        return 2 ** len(features)


class SecondarySpread:
    @staticmethod
    def even(candidates: list[str], n_secondaries: int) -> tuple:
        total = len(candidates)

        if n_secondaries == total:
            return tuple(candidates)
        if n_secondaries == 1:
            return (candidates[(total - 1) // 2],)

        indices = [round(position * (total - 1) / (n_secondaries - 1)) for position in range(n_secondaries)]
        return tuple(candidates[index] for index in indices)


class PatchSweepPlanner:
    def __init__(self, config: PatchSweepConfig, candidates: list[str]) -> None:
        self.config     = config
        self.candidates = list(candidates)

        self._validate()

    @classmethod
    def from_dataset(cls, config: PatchSweepConfig) -> "PatchSweepPlanner":
        path = config.geometry.baselines_file(config.paths.dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"The patch sweep needs the baselines table to enumerate all tracks, but {path} does not exist")

        table = TrackBaselines.load(path)
        return cls(config, list(table.labels[1:]))

    @property
    def total_tracks(self) -> int:
        return 1 + len(self.candidates)

    def _validate(self) -> None:
        counts = self.config.track_counts

        if not counts:
            raise ValueError("track_counts must list at least one track count to sweep")

        if len(set(counts)) != len(counts):
            raise ValueError(f"track_counts must be unique, got {counts}")

        outside = [n for n in counts if not 2 <= n <= self.total_tracks]
        if outside:
            raise ValueError(f"track_counts must lie in [2, {self.total_tracks}] (primary + available secondaries), got {outside}")

        if not 0 < self.config.patch.stride_ratio <= 1:
            raise ValueError(f"patch.stride_ratio={self.config.patch.stride_ratio} must be in (0, 1]")

    def patch_step(self) -> int:
        explicit = self.config.patch.step
        if explicit > 0:
            return explicit

        return ArchitecturePatchStep(self.config.backbone_name, self.config.backbone_head, self.config.model_overrides).resolve()

    def patch_sizes(self) -> list[int]:
        grid  = self.config.patch
        step  = self.patch_step()
        start = grid.minimum if grid.minimum > 0 else step

        if start % step != 0:
            raise ValueError(f"patch.minimum={start} is not a multiple of the admissible step {step}")
        if grid.maximum < start:
            raise ValueError(f"patch.maximum={grid.maximum} is below the first admissible size {start}")

        sizes = list(range(start, grid.maximum + 1, step))
        if len(sizes) < 2:
            raise ValueError(f"The grid [{start}, {grid.maximum}] with step {step} yields {len(sizes)} patch size; a sweep needs at least 2")

        return sizes

    def selections(self) -> dict[int, tuple]:
        return {n: SecondarySpread.even(self.candidates, n - 1) for n in sorted(self.config.track_counts)}

    def _pixel_rescaled(self, base: int, patch_size: int) -> int:
        if not self.config.patch.constant_pixel_budget:
            return base

        reference    = self.config.training.patch_size
        pixel_budget = base * reference[0] * reference[1]

        return max(1, pixel_budget // (patch_size * patch_size))

    def predicted_optimum(self, track_count: int) -> float:
        return self.config.boxcar_window * math.sqrt(self.total_tracks / track_count)

    def unit(self, name: str) -> SweepUnit:
        by_name = {unit.name: unit for unit in self.units()}
        if name not in by_name:
            raise KeyError(f"Unknown sweep unit '{name}', expected one of {sorted(by_name)}")

        return by_name[name]

    def summary(self) -> dict:
        sizes = self.patch_sizes()

        return {
            "Track counts"  : sorted(self.config.track_counts),
            "Total tracks"  : self.total_tracks,
            "Patch step"    : self.patch_step(),
            "Patch sizes"   : sizes,
            "Units"         : len(self.config.track_counts) * len(sizes),
            "Boxcar window" : self.config.boxcar_window,
        }

    def units(self) -> list[SweepUnit]:
        stride_ratio = self.config.patch.stride_ratio
        sizes        = self.patch_sizes()

        plans = []
        for track_count, labels in self.selections().items():
            for size in sizes:
                plans.append(SweepUnit(
                    track_count             = track_count,
                    patch_size              = size,
                    patch_stride            = max(1, int(round(size * stride_ratio))),
                    batch_size              = self._pixel_rescaled(self.config.training.batch_size, size),
                    lr_reference_batch_size = self._pixel_rescaled(self.config.training.lr_reference_batch_size, size),
                    secondary_labels        = labels,
                ))

        return plans
