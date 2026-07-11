from __future__ import annotations

import sys
from pathlib import Path

from configuration.training import CurriculumInheritance, DualEntryConfig, dual_curriculum
from models.dual                          import DUAL_CONFIG_REGISTRY
from pipelines.backbone.training.launcher import SingleTrainRunner
from pipelines.dual.inference.pipeline    import DUAL_INFERENCE_COMPONENTS
from pipelines.dual.training.pipeline     import DualTrainingPipeline
from pipelines.shared.model.model_builder import ModelBuilder
from pipelines.shared.training.seed_sweep import SeedSweepRunner
from tools.runtime.config_cli             import ConfigCli


class DualSingleTrainRunner(SingleTrainRunner):

    pipeline_class = DualTrainingPipeline

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def model_head(self) -> str:
        return "set_pred"

    def _model_config(self):
        trunk_overrides = {
            "params_backbone"    : self.config.params_backbone,
            "existence_backbone" : self.config.existence_backbone,
            "params_input"       : tuple(self.config.params_input),
            "existence_input"    : tuple(self.config.existence_input),
        }

        duplicated = [key for key in trunk_overrides if key in self.config.model_overrides]
        if duplicated:
            raise ValueError(f"Select {duplicated} via the dedicated entry fields, not model_overrides")

        return ModelBuilder.config_from_registry(self.config.model_name, {**trunk_overrides, **self.config.model_overrides}, registry=DUAL_CONFIG_REGISTRY)

    def _inference_components(self):
        return DUAL_INFERENCE_COMPONENTS


class DualTrainingLauncher:

    def __init__(self, entry_script: Path) -> None:
        self.entry_script = Path(entry_script)

    def run(self, argv: list[str] | None = None) -> None:
        argv = list(sys.argv[1:] if argv is None else argv)

        cli    = ConfigCli(DualEntryConfig(), description="Train the dual-input ResUNet set-prediction model: a full-stack trunk feeds the gaussian heads while an interferogram-only trunk feeds the existence gate")
        config = cli.apply(argv)

        CurriculumInheritance(config.curriculum, dual_curriculum(), cli.overrides).apply()

        SeedSweepRunner(config, DualSingleTrainRunner).run()
