from __future__ import annotations

import optuna


class ParamSampler:
    def sample(self, trial: optuna.Trial, space: dict) -> dict:
        sampled = {}
        for name, spec in space.items():
            kind = spec["type"]
            if kind == "float":
                sampled[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
            elif kind == "categorical":
                sampled[name] = trial.suggest_categorical(name, spec["choices"])
            elif kind == "indexed_categorical":
                idx           = trial.suggest_categorical(name + "__idx", list(range(len(spec["choices"]))))
                sampled[name] = spec["choices"][idx]
        return sampled
