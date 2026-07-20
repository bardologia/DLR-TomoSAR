from __future__ import annotations

import sys
from copy    import deepcopy
from pathlib import Path
from typing  import Callable

from tools.orchestration      import ExperimentStage, GpuJob
from tools.monitoring.logger  import Logger
from tools.runtime.config_cli import ConfigCli
from tools.runtime.run_tag    import RunTag


class SeedSet:
    @staticmethod
    def resolve(seeds, default_seed: int) -> list[int]:
        return list(seeds or []) or [default_seed]

    @staticmethod
    def run_name(base: str, seed: int) -> str:
        return f"{base}/seed{seed}"

    @staticmethod
    def base(run_name: str) -> str:
        return run_name.split("/seed")[0]

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

    @staticmethod
    def base_run_name(config, base_label: str | None = None) -> str:
        if config.run_name:
            return config.run_name

        timestamp = RunTag.now()
        return f"{base_label}_{timestamp}" if base_label else timestamp

    def run(self):
        seeds = SeedSet.resolve(self.config.seeds, self.config.seed)

        if len(seeds) > 1:
            raise ValueError(f"SeedSweepRunner trains exactly one seed, got {seeds}; multi-seed runs fan out across the GPU pool via SeedFanoutScheduler")

        config      = deepcopy(self.config)
        config.seed = seeds[0]
        return self.runner_factory(config).run()


class SeedFanoutScheduler:

    SCHEDULER_FIELDS = ("gpus", "gpus_file", "poll_interval_s", "gpu", "seed", "seeds", "run_name")

    def __init__(self, config, cli_overrides: dict, entry_script: Path, base_name: str, run_dir: Path) -> None:
        self.config       = config
        self.entry_script = Path(entry_script)
        self.base_name    = base_name
        self.run_dir      = Path(run_dir)
        self.log_dir      = self.run_dir / "batch_train_logs"
        self.results_path = self.log_dir / "seed_sweep_results.json"

        self.forward_overrides = {path: value for path, value in cli_overrides.items() if path.split(".")[0] not in self.SCHEDULER_FIELDS}

        self.logger = Logger(log_dir=str(self.log_dir), name="seed_sweep")
        self.stage  = ExperimentStage(config=config, run_tag="seed_sweep", logger=self.logger, entry_script=self.entry_script, run_dir=self.run_dir)

    @classmethod
    def for_runner(cls, config, cli_overrides: dict, entry_script: Path, runner_factory: Callable[[object], object], base_label: str | None = None) -> "SeedFanoutScheduler":
        base           = SeedSweepRunner.base_run_name(config, base_label)
        probe          = deepcopy(config)
        probe.run_name = base
        run_dir        = Path(config.logdir) / runner_factory(probe)._resolve_run_name()

        return cls(config, cli_overrides, entry_script, base_name=base, run_dir=run_dir)

    def _job(self, run_name: str, seed: int) -> GpuJob:
        overrides = {**self.forward_overrides, "logdir": str(self.config.logdir), "run_name": run_name, "seed": seed, "seeds": (seed,)}

        return GpuJob(
            name     = run_name,
            command  = [sys.executable, str(self.entry_script)] + ConfigCli.to_argv(overrides),
            log_path = self.log_dir / f"seed{seed}.log",
        )

    def run(self) -> None:
        seeds = SeedSet.resolve(self.config.seeds, self.config.seed)
        units = [(seed, SeedSet.run_name(self.base_name, seed)) for seed in seeds]

        self.logger.section("Seed fan-out")
        self.logger.kv_table({
            "Base run"      : self.base_name,
            "Run dir"       : str(self.run_dir),
            "Seeds"         : seeds,
            "GPUs"          : self.config.gpus,
            "GPU pool file" : str(self.stage.pool_file),
            "CLI overrides" : self.forward_overrides or "—",
            "Log dir"       : str(self.log_dir),
        }, title="Configuration")

        jobs    = [self._job(run_name, seed) for seed, run_name in units]
        names   = [run_name for _, run_name in units]
        results = self.stage._order_results(self.stage._run_queue(jobs), names)

        self.stage._write_results(results, self.results_path)

        self.logger.section("Summary")
        rows = [{"Seed": r["name"], "Status": r["status"], "Duration": f"{r['duration_s'] / 60:.1f} min"} for r in results]
        self.logger.metrics_table(rows, columns=["Seed", "Status", "Duration"])

        failed = [r for r in results if r["status"] != "DONE"]
        self.stage._log_failures(failed)

        self.logger.close()

        if failed:
            raise SystemExit(f"{len(failed)} of {len(results)} seed runs failed; see {self.results_path}")
