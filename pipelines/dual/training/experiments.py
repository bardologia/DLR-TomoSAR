from __future__ import annotations


class DualInputTrialPlanner:

    TRUNK_KEYS   = ("params", "existence")
    FEATURE_KEYS = ("params_features", "existence_features")

    def __init__(self, trials_config, model_overrides: dict, groups: tuple) -> None:
        self.trials          = trials_config
        self.model_overrides = dict(model_overrides)
        self.groups          = tuple(groups)

        self._validate()

    def _validate(self) -> None:
        for key in self.FEATURE_KEYS:
            features = getattr(self.trials, key)
            if not features:
                raise ValueError(f"input_trials.{key} must list at least one feature width")

            invalid = [width for width in features if not isinstance(width, int) or width < 1]
            if invalid:
                raise ValueError(f"input_trials.{key} must be positive integers, got {features}")

        duplicated = [key for key in self.FEATURE_KEYS if key in self.model_overrides]
        if duplicated:
            raise ValueError(f"Set {duplicated} via the input_trials fields, not model_overrides")

        if not self.trials.trials:
            raise ValueError("input_trials.trials must list at least one trunk-input variant")

        for label, spec in self.trials.trials.items():
            missing = [key for key in self.TRUNK_KEYS if key not in spec]
            if missing:
                raise ValueError(f"Input trial '{label}' must set {missing}; every trial defines all of {self.TRUNK_KEYS}")

            unknown = [key for key in spec if key not in self.TRUNK_KEYS]
            if unknown:
                raise ValueError(f"Input trial '{label}' has unknown keys {unknown}; allowed keys are {self.TRUNK_KEYS}")

            for key in self.TRUNK_KEYS:
                self._validate_selection(label, key, spec[key])

    def _validate_selection(self, label: str, key: str, selection) -> None:
        if not selection:
            raise ValueError(f"Input trial '{label}' selects no channel groups for '{key}'")

        unknown = [group for group in selection if group not in self.groups]
        if unknown:
            raise ValueError(f"Input trial '{label}' has unknown '{key}' groups {unknown}; available groups are {list(self.groups)}")

        duplicates = sorted({group for group in selection if list(selection).count(group) > 1})
        if duplicates:
            raise ValueError(f"Input trial '{label}' repeats '{key}' groups {duplicates}")

    def summary(self) -> dict:
        return {
            "Trunk features" : f"params {self.trials.params_features}, existence {self.trials.existence_features}",
            "Input variants" : {label: f"{'+'.join(spec['params'])} | {'+'.join(spec['existence'])}" for label, spec in self.trials.trials.items()},
            "Total runs"     : len(self.trials.trials),
        }

    def _overrides(self, spec: dict) -> dict:
        features = {
            "params_features"    : list(self.trials.params_features),
            "existence_features" : list(self.trials.existence_features),
        }

        return {
            "params_input"    : list(spec["params"]),
            "existence_input" : list(spec["existence"]),
            "model_overrides" : {**self.model_overrides, **features},
        }

    def plan(self) -> list[tuple[str, dict]]:
        plans = []

        for label, spec in self.trials.trials.items():
            run_name = f"di-{label}"
            plans.append((run_name, self._overrides(spec)))

        return plans
