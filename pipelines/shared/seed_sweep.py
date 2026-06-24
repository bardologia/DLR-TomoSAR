from __future__ import annotations

from dataclasses import replace
from datetime    import datetime
from typing      import Callable


class SeedSweepRunner:
    def __init__(self, config, runner_factory: Callable[[object], object]) -> None:
        self.config         = config
        self.runner_factory = runner_factory

    def _seeds(self) -> list[int]:
        seeds = list(getattr(self.config, "seeds", None) or [])
        return seeds or [self.config.seed]

    def _base_run_name(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.config.run_name or f"run_{timestamp}"

    def _run_seed(self, base_name: str, seed: int):
        config = replace(self.config, seed=seed, run_name=f"{base_name}_seed{seed}")
        return self.runner_factory(config).run()

    def run(self):
        seeds = self._seeds()

        if len(seeds) == 1:
            config = replace(self.config, seed=seeds[0])
            return self.runner_factory(config).run()

        base    = self._base_run_name()
        results = {seed: self._run_seed(base, seed) for seed in seeds}

        return results
