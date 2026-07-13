from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path

from configuration.patch_sweep             import PatchSweepConfig
from pipelines.shared.model.model_builder  import ModelBuilder


@dataclass(frozen=True)
class SweepUnit:
    dataset_path            : Path
    parameters_path         : Path
    patch_size              : int
    patch_stride            : int
    batch_size              : int
    lr_reference_batch_size : int

    @property
    def dataset(self) -> str:
        return self.dataset_path.name

    @property
    def name(self) -> str:
        return f"{self.dataset}-p{self.patch_size:03d}"


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


class PatchSweepPlanner:
    def __init__(self, config: PatchSweepConfig) -> None:
        self.config = config

        self._validate()

    def _validate(self) -> None:
        datasets = self.config.dataset_paths

        if not datasets:
            raise ValueError("dataset_paths must list at least one dataset to sweep")

        names      = [Path(dataset).name for dataset in datasets]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"dataset_paths must have unique directory names (they key the sweep units), duplicated: {duplicates}")

        if not 0 < self.config.patch.stride_ratio <= 1:
            raise ValueError(f"patch.stride_ratio={self.config.patch.stride_ratio} must be in (0, 1]")

        self.parameters_template()

    def parameters_template(self) -> Path:
        paths = self.config.paths

        try:
            return Path(paths.parameters_path).relative_to(paths.dataset_path)
        except ValueError:
            raise ValueError(f"paths.parameters_path={paths.parameters_path} must live inside paths.dataset_path={paths.dataset_path}; the sweep re-roots this relative layout onto every swept dataset")

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

    def _pixel_rescaled(self, base: int, patch_size: int) -> int:
        if not self.config.patch.constant_pixel_budget:
            return base

        reference    = self.config.training.patch_size
        pixel_budget = base * reference[0] * reference[1]

        return max(1, pixel_budget // (patch_size * patch_size))

    def unit(self, name: str) -> SweepUnit:
        by_name = {unit.name: unit for unit in self.units()}
        if name not in by_name:
            raise KeyError(f"Unknown sweep unit '{name}', expected one of {sorted(by_name)}")

        return by_name[name]

    def summary(self) -> dict:
        sizes = self.patch_sizes()

        return {
            "Datasets"    : [Path(dataset).name for dataset in self.config.dataset_paths],
            "Patch step"  : self.patch_step(),
            "Patch sizes" : sizes,
            "Units"       : len(self.config.dataset_paths) * len(sizes),
            "Seeds"       : list(self.config.seeds) or [self.config.seed],
        }

    def units(self) -> list[SweepUnit]:
        stride_ratio = self.config.patch.stride_ratio
        sizes        = self.patch_sizes()
        template     = self.parameters_template()

        plans = []
        for dataset in self.config.dataset_paths:
            dataset = Path(dataset)
            for size in sizes:
                plans.append(SweepUnit(
                    dataset_path            = dataset,
                    parameters_path         = dataset / template,
                    patch_size              = size,
                    patch_stride            = max(1, int(round(size * stride_ratio))),
                    batch_size              = self._pixel_rescaled(self.config.training.batch_size, size),
                    lr_reference_batch_size = self._pixel_rescaled(self.config.training.lr_reference_batch_size, size),
                ))

        return plans
