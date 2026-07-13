from __future__ import annotations

from pipelines.shared.training.seed_sweep import SeedSweepRunner
from tools.runtime.config_cli    import ConfigCli


class SeedSweepLauncher:

    def __init__(self, entry_config, runner_class, description: str, base_attr: str | None = None) -> None:
        self.entry_config = entry_config
        self.runner_class = runner_class
        self.description  = description
        self.base_attr    = base_attr

    def run(self, argv: list[str] | None = None) -> None:
        config     = ConfigCli(self.entry_config, description=self.description).apply(argv)
        base_label = getattr(config, self.base_attr) if self.base_attr else None

        SeedSweepRunner(config, self.runner_class, base_label=base_label).run()
