from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.runtime.config_cli import ConfigCli


class TrainingLauncher:

    MODES = ("backbone", "jepa", "autoencoder")

    def __init__(self, entry_script: Path) -> None:
        self.entry_script = Path(entry_script)

    def _resolve_mode(self, default: str, argv: list[str]) -> str:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--mode", choices=self.MODES, default=None)
        args, _ = parser.parse_known_args(argv)
        return args.mode or default

    def _backbone(self, config, argv: list[str]) -> None:
        from pipelines.backbone.training.pipeline import SingleTrainRunner, TrainScheduler

        trial_parser = argparse.ArgumentParser(add_help=False)
        trial_parser.add_argument("--trial", action="store_true")
        trial, _ = trial_parser.parse_known_args(argv)

        cli    = ConfigCli(config, description="Train one model end to end, or fan out curriculum/warmup/secondary trials across GPUs")
        config = cli.apply(argv)

        if trial.trial or not config.trials_enabled:
            SingleTrainRunner(config).run()
            return

        TrainScheduler(config=config, cli_overrides=cli.overrides, entry_script=self.entry_script, stage="backbone").run()

    def _jepa(self, config, argv: list[str]) -> None:
        from pipelines.jepa.training.pipeline import SingleTrainRunner

        config = ConfigCli(config, description="JEPA predictor training").apply(argv)
        SingleTrainRunner(config).run()

    def _autoencoder(self, config, argv: list[str]) -> None:
        from pipelines.profile_autoencoder.training.pipeline import SingleTrainRunner

        config = ConfigCli(config, description="Profile autoencoder training").apply(argv)
        SingleTrainRunner(config).run()

    def run(self) -> None:
        from configuration.training.train_config import TrainEntryConfig

        argv  = sys.argv[1:]
        entry = TrainEntryConfig()
        mode  = self._resolve_mode(entry.mode, argv)
        sub   = getattr(entry, mode)

        dispatch = {
            "backbone"    : self._backbone,
            "jepa"        : self._jepa,
            "autoencoder" : self._autoencoder,
        }
        dispatch[mode](sub, argv)
