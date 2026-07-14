from __future__ import annotations

import copy
import shutil

from datetime import datetime
from pathlib  import Path

import numpy as np

from configuration.sar.gaussian_config          import GaussianConfig
from configuration.training                     import UnrolledEntryConfig
from models.unrolled                            import UNROLLED_MODEL_REGISTRY, get_unrolled
from pipelines.backbone.dataset.pipeline        import DatasetPipeline
from pipelines.shared.config.config_factory     import ConfigFactory
from pipelines.shared.config.config_persistence import UnrolledModelConfigIO
from pipelines.shared.training.overfit_check    import OverfitCheck
from pipelines.shared.training.seed_sweep       import SeedSweepRunner
from pipelines.unrolled.training.trainer        import UnrolledTrainer
from tools.data.gaussians          import GaussianAxis
from tools.monitoring.logger       import Logger
from tools.runtime.config_cli      import ConfigCli
from tools.runtime.reproducibility import Reproducibility


class OverfitGateTrainer:
    def __init__(self, trainer: UnrolledTrainer) -> None:
        self.trainer = trainer

    def train(self, train_loader, val_loader, test_loader) -> tuple:
        results = self.trainer.train(train_loader, val_loader, test_loader)
        losses  = [entry["train"]["loss"] for entry in results["history"]]

        return losses, None, None


class UnrolledOverfitGate:
    def __init__(self, entry_config: UnrolledEntryConfig, run_directory: Path, logger) -> None:
        self.entry  = entry_config
        self.logger = logger
        self.check  = OverfitCheck(entry_config.overfit_check, run_directory, logger)

    def _sanitized_entry_config(self) -> UnrolledEntryConfig:
        gate = copy.deepcopy(self.entry)

        gate.training.epochs              = self.check.planned_epochs
        gate.training.scheduler_epochs    = 1_000_000
        gate.training.early_stop_patience = self.check.planned_epochs
        gate.training.warmup_enabled      = False
        gate.training.use_ema             = False
        gate.training.reserve_vram        = False
        gate.measurement_noise_std        = 0.0

        self.check.record("training.scheduler_epochs",    1_000_000)
        self.check.record("training.early_stop_patience", self.check.planned_epochs)
        self.check.record("training.warmup_enabled",      False)
        self.check.record("training.use_ema",             False)
        self.check.record("measurement_noise_std",        0.0)

        return gate

    def run(self, model_cfg, x_axis, ppg: int, norm_stats, train_dataset) -> None:
        if not self.check.enabled:
            return

        gate_entry     = self._sanitized_entry_config()
        gate_model_cfg = self.check.sanitized_model_config(model_cfg)
        gate_model     = UNROLLED_MODEL_REGISTRY[self.entry.model_name](gate_model_cfg)

        gate_trainer = UnrolledTrainer(gate_model, gate_model_cfg, x_axis, gate_entry, ppg, self.check.work_directory, self.logger, norm_stats)

        try:
            self.check.run(OverfitGateTrainer(gate_trainer), train_dataset)
        finally:
            shutil.rmtree(self.check.work_directory, ignore_errors=True)


class UnrolledTrainingPipeline:
    def __init__(self, config: UnrolledEntryConfig) -> None:
        self.config  = config
        self.factory = ConfigFactory(config)

        Reproducibility.seed_everything(config.seed)

    def _run_directory(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name  = self.config.run_name if self.config.run_name else f"{self.config.model_name}_{stamp}"

        run_directory = Path(self.config.logdir) / name
        run_directory.mkdir(parents=True, exist_ok=True)

        return run_directory

    def _build_dataset_pipeline(self, run_directory: Path, logger, gaussian_cfg) -> tuple:
        dataset_config             = self.factory.training_dataset_config()
        dataset_config.n_gaussians = gaussian_cfg.n_default_gaussians

        geometry = self.config.geometry.resolved(self.config.paths.dataset_path, secondary_labels=self.factory._secondary_labels())

        dataset_pipeline = DatasetPipeline(
            config                 = dataset_config,
            training_run_directory = run_directory,
            logger                 = logger,
            seed                   = self.config.seed,
            height_axis_convention = geometry.height_axis_convention,
            build_geometry_field   = True,
        )

        dataset_config.x_axis = GaussianAxis.build(gaussian_cfg.x_min, gaussian_cfg.x_max, dataset_pipeline.layout.profile_length)

        return dataset_pipeline, dataset_config

    def _build_model(self, logger):
        model, model_cfg = get_unrolled(self.config.model_name, **self.config.model_overrides)

        logger.section("[Model Built]")
        logger.subsection(f"Architecture : {self.config.model_name}")
        logger.subsection(f"Iterations   : {model_cfg.n_iterations}")
        logger.subsection(f"Parameters   : {sum(p.numel() for p in model.parameters()):,}")

        return model, model_cfg

    def run(self) -> dict:
        run_directory = self._run_directory()
        logger        = Logger(log_dir=str(run_directory / "logs"), name="unrolled_training")

        logger.section("[Unrolled Training Pipeline Execution]")

        ConfigCli.save_resolved(self.config, run_directory / "docs" / "resolved_entry_config.json")

        gaussian_cfg = GaussianConfig.from_dataset(self.config.paths.dataset_path, self.config.paths.parameters_path)

        dataset_pipeline, dataset_config = self._build_dataset_pipeline(run_directory, logger, gaussian_cfg)
        train_loader, val_loader, test_loader, datasets = dataset_pipeline.run()

        model, model_cfg = self._build_model(logger)

        meta_directory = run_directory / "meta"
        meta_directory.mkdir(parents=True, exist_ok=True)
        UnrolledModelConfigIO.save(model_cfg, self.config.model_name, meta_directory)

        x_axis = np.asarray(dataset_config.x_axis, dtype=np.float32)

        UnrolledOverfitGate(self.config, run_directory, logger).run(model_cfg, x_axis, gaussian_cfg.params_per_gaussian, datasets["train"].normalizer, datasets["train"])

        trainer = UnrolledTrainer(
            model        = model,
            model_cfg    = model_cfg,
            x_axis       = x_axis,
            entry_config = self.config,
            ppg          = gaussian_cfg.params_per_gaussian,
            run_dir      = run_directory,
            logger       = logger,
            norm_stats   = datasets["train"].normalizer,
        )

        try:
            results = trainer.train(train_loader, val_loader, test_loader)
        finally:
            logger.close()

        return results


class UnrolledTrainingLauncher:
    def __init__(self, entry_script: Path) -> None:
        self.entry_script = Path(entry_script)

    def run(self, argv: list[str] | None = None) -> None:
        import sys

        argv   = list(sys.argv[1:] if argv is None else argv)
        cli    = ConfigCli(UnrolledEntryConfig(), description="Train the unrolled physics network on synthesised per-pixel coherence measurements")
        config = cli.apply(argv)

        SeedSweepRunner(config, UnrolledTrainingPipeline, base_label=config.model_name).run()
