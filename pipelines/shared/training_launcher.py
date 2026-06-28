from __future__ import annotations

from pipelines.shared.seed_sweep import SeedSweepRunner
from tools.runtime.config_cli    import ConfigCli


class SeedSweepLauncher:

    def __init__(self, entry_config, runner_class, description: str) -> None:
        self.entry_config = entry_config
        self.runner_class = runner_class
        self.description  = description

    def run(self, argv: list[str] | None = None) -> None:
        config = ConfigCli(self.entry_config, description=self.description).apply(argv)
        SeedSweepRunner(config, self.runner_class).run()
