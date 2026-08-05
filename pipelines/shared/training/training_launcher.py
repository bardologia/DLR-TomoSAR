from __future__ import annotations

from pathlib import Path

from pipelines.shared.training.seed_sweep import SeedFanoutScheduler, SeedSet, SeedSweepRunner
from tools.runtime.config_cli             import ConfigCli


class SeedSweepLauncher:

    def __init__(self, entry_config, runner_class, description: str, entry_script: Path, base_attr: str | None = None, supports_infer_at_end: bool = False) -> None:
        self.entry_config          = entry_config
        self.runner_class          = runner_class
        self.description           = description
        self.entry_script          = Path(entry_script)
        self.base_attr             = base_attr
        self.supports_infer_at_end = supports_infer_at_end

    def run(self, argv: list[str] | None = None) -> None:
        cli    = ConfigCli(self.entry_config, description=self.description)
        config = cli.apply(argv)

        infer_at_end = self.supports_infer_at_end and config.infer_at_end
        if infer_at_end and config.infer_after:
            raise SystemExit("infer_after and infer_at_end are mutually exclusive; infer_after runs inference inside each training job, infer_at_end batches inference after every training finishes")

        seeds = SeedSet.resolve(config.seeds, config.seed)
        if len(seeds) == 1:
            SeedSweepRunner(config, self.runner_class).run()
            return

        base_label = getattr(config, self.base_attr) if self.base_attr else None
        SeedFanoutScheduler.for_runner(config, cli.overrides, self.entry_script, self.runner_class, base_label=base_label, infer_at_end=infer_at_end).run()
