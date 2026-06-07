from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import EnvironmentPinner


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trial", action="store_true")
    args, _ = parser.parse_known_args()

    EnvironmentPinner.gpu(expandable_segments=True)

    from configuration.train_config import TrainEntryConfig
    from pipelines.training_pipeline.pipeline import SingleTrainRunner, TrainScheduler
    from tools.config_cli import ConfigCli

    cli    = ConfigCli(TrainEntryConfig(), description="Training run, optionally fanned out into curriculum, warmup-loss or secondary-set trials")
    config = cli.apply()

    if args.trial or not config.trials_enabled:
        SingleTrainRunner(config).run()
        return

    scheduler = TrainScheduler(
        config        = config,
        cli_overrides = cli.overrides,
        entry_script  = Path(__file__).resolve(),
    )
    scheduler.run()


if __name__ == "__main__":
    main()
