from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.baselines import TrackBaselines


class TrialPlanner:
    def __init__(self, warmup_losses: dict) -> None:
        self.warmup_losses = warmup_losses

    @staticmethod
    def _stage_overrides(stage: str, loss: dict) -> dict:
        return {f"curriculum.{stage}.{key}": value for key, value in loss.items()}


class CurriculumTrialPlanner(TrialPlanner):
    def __init__(self, warmup_losses: dict, complete_losses: dict) -> None:
        super().__init__(warmup_losses)
        self.complete_losses = complete_losses

    def summary(self) -> dict:
        return {"Warmup losses": len(self.warmup_losses), "Complete losses": len(self.complete_losses)}

    def plan(self) -> list[tuple[str, dict]]:
        plans = []

        for warmup_label, warmup_loss in self.warmup_losses.items():
            for complete_label, complete_loss in self.complete_losses.items():
                run_name  = f"w-{warmup_label}_c-{complete_label}"
                overrides = {"curriculum.enabled": True, "curriculum.inherit": False}
                overrides.update(self._stage_overrides("warmup",   warmup_loss))
                overrides.update(self._stage_overrides("complete", complete_loss))
                plans.append((run_name, overrides))

        return plans


class WarmupTrialPlanner(TrialPlanner):
    def summary(self) -> dict:
        return {"Warmup losses": len(self.warmup_losses)}

    def plan(self) -> list[tuple[str, dict]]:
        plans = []

        for label, loss in self.warmup_losses.items():
            run_name  = f"nc-{label}"
            overrides = {"curriculum.enabled": False}
            overrides.update(self._stage_overrides("complete", loss))
            plans.append((run_name, overrides))

        return plans


class SlotPresenceTrialPlanner:

    def __init__(self, presence_trials: dict) -> None:
        self.presence_trials = presence_trials

    def summary(self) -> dict:
        return {
            "Presence trials" : len(self.presence_trials),
            "Total runs"      : len(self.presence_trials),
        }

    def _overrides(self, spec: dict) -> dict:
        overrides = {"curriculum.enabled": False}

        for key, value in spec.items():
            overrides[f"curriculum.complete.{key}"] = value

        return overrides

    def plan(self) -> list[tuple[str, dict]]:
        plans = []

        for trial_label, spec in self.presence_trials.items():
            run_name = f"pr-{trial_label}"
            plans.append((run_name, self._overrides(spec)))

        return plans


class AblationTrialPlanner:

    def __init__(self, features: list, include_full: bool) -> None:
        self.features     = list(features)
        self.include_full = include_full

        self._validate()

    def _validate(self) -> None:
        if not self.features:
            raise ValueError("ablation_features must list at least one feature to ablate")

        for index, feature in enumerate(self.features):
            missing = [key for key in ("label", "enable", "degrade") if key not in feature]
            if missing:
                raise ValueError(f"Ablation feature #{index} must define {missing}, got {feature}")

        labels     = [feature["label"] for feature in self.features]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(f"Ablation feature labels must be unique, duplicated: {duplicates}; duplicate labels collapse distinct trials onto one run directory")

    def summary(self) -> dict:
        return {
            "Ablation features" : len(self.features),
            "Order"             : " -> ".join(feature["label"] for feature in self.features),
            "Total runs"        : len(self.features) + (1 if self.include_full else 0),
        }

    def _enabled_overrides(self) -> dict:
        merged = {}
        for feature in self.features:
            merged.update(feature["enable"])
        return merged

    def _degraded_prefix(self, count: int) -> dict:
        merged = {}
        for feature in self.features[:count]:
            merged.update(feature["degrade"])
        return merged

    def _run_name(self, step: int) -> str:
        if step == 0:
            return "abl-0-full"
        if step == len(self.features):
            return f"abl-{step}-baseline"
        return f"abl-{step}-no_{self.features[step - 1]['label']}"

    def plan(self) -> list[tuple[str, dict]]:
        enabled = self._enabled_overrides()
        plans   = []

        if self.include_full:
            plans.append((self._run_name(0), dict(enabled)))

        for step in range(1, len(self.features) + 1):
            overrides = {**enabled, **self._degraded_prefix(step)}
            plans.append((self._run_name(step), overrides))

        return plans


class InputTrialPlanner:

    INPUT_KEYS = ("use_primary", "use_secondaries", "use_interferograms", "use_dem")

    def __init__(self, input_trials: dict, candidates: list[str]) -> None:
        self.input_trials = input_trials
        self.candidates   = list(candidates)

        self._validate()

    @classmethod
    def from_dataset(cls, input_trials: dict, geometry, dataset_path: str | Path) -> "InputTrialPlanner":
        path = geometry.baselines_file(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Input trials need the baselines table to enumerate all tracks, but {path} does not exist")

        table = TrackBaselines.load(path)
        return cls(input_trials, list(table.labels[1:]))

    def _validate(self) -> None:
        if not self.input_trials:
            raise ValueError("input_trials must list at least one input variant")

        for label, spec in self.input_trials.items():
            unknown = [key for key in spec if key not in self.INPUT_KEYS]
            if unknown:
                raise ValueError(f"Input trial '{label}' has unknown keys {unknown}; allowed keys are {self.INPUT_KEYS}")

    def summary(self) -> dict:
        return {
            "Input variants" : len(self.input_trials),
            "Tracks"         : f"all ({len(self.candidates)} secondaries)",
        }

    def _overrides(self, spec: dict) -> dict:
        overrides = {f"input.{key}": value for key, value in spec.items()}
        overrides["paths.secondary_labels"] = tuple(self.candidates)
        return overrides

    def plan(self) -> list[tuple[str, dict]]:
        plans = []

        for label, spec in self.input_trials.items():
            run_name = f"in-{label}"
            plans.append((run_name, self._overrides(spec)))

        return plans


class PatchSizeTrialPlanner:
    def __init__(self, trials) -> None:
        self.trials = trials

        self._validate()

    def _validate(self) -> None:
        if not self.trials.sizes:
            raise ValueError("patch_trials.sizes must list at least one patch size")

        if any(size < 1 for size in self.trials.sizes):
            raise ValueError(f"patch_trials.sizes must all be positive integers, got {self.trials.sizes}")

        if not 0 < self.trials.stride_ratio <= 1:
            raise ValueError(f"patch_trials.stride_ratio={self.trials.stride_ratio} must be in (0, 1]")

    def summary(self) -> dict:
        return {
            "Patch sizes"     : list(self.trials.sizes),
            "Stride ratio"    : self.trials.stride_ratio,
            "Max-batch probe" : self.trials.find_max_batch,
            "Scale LR"        : self.trials.scale_lr,
        }

    def plan(self) -> list[tuple[str, dict]]:
        plans = []

        for size in self.trials.sizes:
            stride    = max(1, int(round(size * self.trials.stride_ratio)))
            run_name  = f"p-{size}"
            overrides = {
                "training.patch_size"          : (size, size),
                "training.patch_stride"        : stride,
                "pretrain.find_batch_size"     : self.trials.find_max_batch,
                "training.scale_lr_with_batch" : self.trials.scale_lr,
            }
            plans.append((run_name, overrides))

        return plans


class SecondaryTrialPlanner:

    STRATEGIES   = ("uniform", "gaussian", "consecutive", "spaced")
    MAX_ATTEMPTS = 1000

    def __init__(self, trials, candidates: list[str]) -> None:
        self.trials     = trials
        self.candidates = list(candidates)

        self._validate()

    @classmethod
    def from_dataset(cls, trials, geometry, dataset_path: str | Path) -> "SecondaryTrialPlanner":
        path = geometry.baselines_file(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Secondary trials need the baselines table, but {path} does not exist")

        table = TrackBaselines.load(path)
        return cls(trials, list(table.labels[1:]))

    def _validate(self) -> None:
        t = self.trials

        if t.strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown secondary_trials.strategy '{t.strategy}', expected one of {self.STRATEGIES}")

        if not 1 <= t.n_secondaries <= len(self.candidates):
            raise ValueError(f"secondary_trials.n_secondaries={t.n_secondaries} must be in [1, {len(self.candidates)}], candidates are {self.candidates}")

        if t.strategy in ("uniform", "gaussian") and t.n_trials < 1:
            raise ValueError(f"secondary_trials.n_trials={t.n_trials} must be >= 1 for strategy '{t.strategy}'")

        if t.strategy == "gaussian" and (t.mean is None or t.sigma is None):
            raise ValueError("secondary_trials strategy 'gaussian' requires explicit mean and sigma (index units over the secondary list)")

        if t.strategy in ("consecutive", "spaced") and t.block_step < 1:
            raise ValueError(f"secondary_trials.block_step={t.block_step} must be >= 1")

        if t.strategy == "spaced" and t.spacing < 1:
            raise ValueError(f"secondary_trials.spacing={t.spacing} must be >= 1")

    def summary(self) -> dict:
        return {
            "Strategy"       : self.trials.strategy,
            "Secondaries"    : self.trials.n_secondaries,
            "Candidates"     : f"{len(self.candidates)} -> {self.candidates}",
        }

    def _pick(self, indices) -> list[str]:
        return [self.candidates[index] for index in sorted(indices)]

    def _uniform(self) -> list[list[str]]:
        rng = np.random.default_rng(self.trials.seed)

        seen, selections = set(), []
        for _ in range(self.MAX_ATTEMPTS):
            if len(selections) == self.trials.n_trials:
                break

            indices = tuple(sorted(rng.choice(len(self.candidates), size=self.trials.n_secondaries, replace=False).tolist()))
            if indices in seen:
                continue

            seen.add(indices)
            selections.append(self._pick(indices))

        self._ensure_complete(selections)
        return selections

    def _gaussian(self) -> list[list[str]]:
        rng = np.random.default_rng(self.trials.seed)

        seen, selections = set(), []
        for _ in range(self.MAX_ATTEMPTS):
            if len(selections) == self.trials.n_trials:
                break

            indices = self._gaussian_draw(rng)
            if indices is None or indices in seen:
                continue

            seen.add(indices)
            selections.append(self._pick(indices))

        self._ensure_complete(selections)
        return selections

    def _gaussian_draw(self, rng) -> tuple | None:
        indices = set()

        for _ in range(self.MAX_ATTEMPTS):
            if len(indices) == self.trials.n_secondaries:
                return tuple(sorted(indices))

            index = int(round(rng.normal(self.trials.mean, self.trials.sigma)))
            if 0 <= index < len(self.candidates):
                indices.add(index)

        return None

    def _consecutive(self) -> list[list[str]]:
        n, step, total = self.trials.n_secondaries, self.trials.block_step, len(self.candidates)

        return [self._pick(range(start, start + n)) for start in range(0, total - n + 1, step)]

    def _spaced(self) -> list[list[str]]:
        n, step, spacing, total = self.trials.n_secondaries, self.trials.block_step, self.trials.spacing, len(self.candidates)

        span = (n - 1) * spacing + 1
        if span > total:
            raise ValueError(f"Spaced selection spans {span} candidates but only {total} are available; reduce n_secondaries or spacing")

        return [self._pick(range(start, start + span, spacing)) for start in range(0, total - span + 1, step)]

    def _ensure_complete(self, selections: list) -> None:
        if len(selections) < self.trials.n_trials:
            raise RuntimeError(f"Found only {len(selections)} of {self.trials.n_trials} distinct secondary sets after {self.MAX_ATTEMPTS} attempts; lower n_trials, widen sigma or change strategy")

    def plan(self) -> list[tuple[str, dict]]:
        selections = getattr(self, f"_{self.trials.strategy}")()

        plans = []
        for index, labels in enumerate(selections):
            run_name = f"sec-{self.trials.strategy}-t{index:02d}_{'-'.join(labels)}"
            plans.append((run_name, {"paths.secondary_labels": tuple(labels)}))

        return plans
