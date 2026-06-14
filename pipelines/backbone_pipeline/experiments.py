from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.track_baselines import TrackBaselines


class CurriculumTrialPlanner:
    def __init__(self, model_name: str, warmup_losses: dict, complete_losses: dict) -> None:
        self.model_name      = model_name
        self.warmup_losses   = warmup_losses
        self.complete_losses = complete_losses

    def summary(self) -> dict:
        return {"Warmup losses": len(self.warmup_losses), "Complete losses": len(self.complete_losses)}

    def plan(self) -> list[tuple[str, dict]]:
        plans = []

        for warmup_label, warmup_loss in self.warmup_losses.items():
            for complete_label, complete_loss in self.complete_losses.items():
                run_name  = f"{self.model_name}_w-{warmup_label}_c-{complete_label}"
                overrides = {"curriculum.enabled": True}
                overrides.update({f"curriculum.warmup.{key}":   value for key, value in warmup_loss.items()})
                overrides.update({f"curriculum.complete.{key}": value for key, value in complete_loss.items()})
                plans.append((run_name, overrides))

        return plans


class WarmupTrialPlanner:
    def __init__(self, model_name: str, warmup_losses: dict) -> None:
        self.model_name    = model_name
        self.warmup_losses = warmup_losses

    def summary(self) -> dict:
        return {"Warmup losses": len(self.warmup_losses)}

    def plan(self) -> list[tuple[str, dict]]:
        plans = []

        for label, loss in self.warmup_losses.items():
            run_name  = f"{self.model_name}_nc-{label}"
            overrides = {"curriculum.enabled": False}
            overrides.update({f"curriculum.warmup.{key}": value for key, value in loss.items()})
            plans.append((run_name, overrides))

        return plans


class SecondaryTrialPlanner:

    STRATEGIES   = ("uniform", "gaussian", "consecutive", "spaced")
    MAX_ATTEMPTS = 1000

    def __init__(self, model_name: str, trials, candidates: list[str]) -> None:
        self.model_name = model_name
        self.trials     = trials
        self.candidates = list(candidates)

        self._validate()

    @classmethod
    def from_dataset(cls, model_name: str, trials, geometry, dataset_path: str | Path) -> "SecondaryTrialPlanner":
        path = geometry.baselines_file(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Secondary trials need the baselines table, but {path} does not exist")

        table = TrackBaselines.load(path)
        return cls(model_name, trials, list(table.labels[1:]))

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
            run_name = f"{self.model_name}_sec-{self.trials.strategy}-t{index:02d}_{'-'.join(labels)}"
            plans.append((run_name, {"paths.secondary_labels": tuple(labels)}))

        return plans
