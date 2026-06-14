from __future__ import annotations

import argparse
from pathlib import Path

from tools.runtime.config_cli import ConfigCli


class TrainingLauncher:

    STAGES = ("backbone", "jepa", "autoencoder")

    def __init__(self, entry_script: Path) -> None:
        self.entry_script = Path(entry_script)

    def _parse_stage(self) -> tuple[str, list[str]]:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("stage", choices=self.STAGES)
        args, rest = parser.parse_known_args()
        return args.stage, rest

    def _backbone(self, argv: list[str]) -> None:
        from configuration.training.backbone_config       import BackboneEntryConfig
        from pipelines.backbone.training.pipeline import SingleTrainRunner, TrainScheduler

        trial_parser = argparse.ArgumentParser(add_help=False)
        trial_parser.add_argument("--trial", action="store_true")
        trial, _ = trial_parser.parse_known_args(argv)

        cli    = ConfigCli(BackboneEntryConfig(), description="Train one model end to end, or fan out curriculum/warmup/secondary trials across GPUs")
        config = cli.apply(argv)

        if trial.trial or not config.trials_enabled:
            SingleTrainRunner(config).run()
            return

        TrainScheduler(config=config, cli_overrides=cli.overrides, entry_script=self.entry_script, stage="backbone").run()

    def _jepa(self, argv: list[str]) -> None:
        from configuration.training.jepa_config       import JepaEntryConfig
        from pipelines.jepa.training.pipeline import SingleTrainRunner

        config = ConfigCli(JepaEntryConfig(), description="Stage-B JEPA predictor training").apply(argv)
        SingleTrainRunner(config).run()

    def _autoencoder(self, argv: list[str]) -> None:
        from configuration.training.autoencoder_config       import ProfileAeEntryConfig
        from pipelines.profile_autoencoder.training.pipeline import SingleTrainRunner

        config = ConfigCli(ProfileAeEntryConfig(), description="Stage-A profile autoencoder training").apply(argv)
        SingleTrainRunner(config).run()

    def run(self) -> None:
        stage, argv = self._parse_stage()

        dispatch = {
            "backbone"    : self._backbone,
            "jepa"        : self._jepa,
            "autoencoder" : self._autoencoder,
        }
        dispatch[stage](argv)
