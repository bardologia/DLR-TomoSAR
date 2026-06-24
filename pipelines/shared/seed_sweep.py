from __future__ import annotations

from dataclasses import replace
from datetime    import datetime
from typing      import Callable


class SeedSet:
    @staticmethod
    def resolve(seeds, default_seed: int) -> list[int]:
        return list(seeds or []) or [default_seed]

    @staticmethod
    def run_name(base: str, seed: int) -> str:
        return f"{base}_seed{seed}"

    @staticmethod
    def base(run_name: str) -> str:
        return run_name.split("_seed")[0]

    @staticmethod
    def cli_args(seed: int | None) -> list[str]:
        return ["--seed", str(seed)] if seed is not None else []

    @staticmethod
    def units(bases, seeds) -> list[tuple[str, int | None, str]]:
        seeds = list(seeds or [])

        if not seeds:
            return [(base, None, base) for base in bases]

        return [(base, seed, SeedSet.run_name(base, seed)) for base in bases for seed in seeds]


class SeedSweepRunner:
    def __init__(self, config, runner_factory: Callable[[object], object]) -> None:
        self.config         = config
        self.runner_factory = runner_factory

    def _seeds(self) -> list[int]:
        return SeedSet.resolve(getattr(self.config, "seeds", None), self.config.seed)

    def _base_run_name(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.config.run_name or f"run_{timestamp}"

    def _run_seed(self, base_name: str, seed: int):
        config = replace(self.config, seed=seed, run_name=SeedSet.run_name(base_name, seed))
        return self.runner_factory(config).run()

    def run(self):
        seeds = self._seeds()

        if len(seeds) == 1:
            config = replace(self.config, seed=seeds[0])
            return self.runner_factory(config).run()

        base    = self._base_run_name()
        results = {seed: self._run_seed(base, seed) for seed in seeds}

        return results
