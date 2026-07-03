from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import replace
from pathlib     import Path

import torch

from configuration.training import BackboneEntryConfig
from models                 import BACKBONE_IMAGE_SIZE_MODELS, get_backbone
from pipelines.backbone.dataset.pipeline     import DatasetPipeline
from pipelines.backbone.inference.pipeline   import InferencePipeline
from pipelines.backbone.training.experiments import AblationTrialPlanner, CurriculumTrialPlanner, InputTrialPlanner, PatchSizeTrialPlanner, SecondaryTrialPlanner, SlotPresenceTrialPlanner, WarmupTrialPlanner
from pipelines.backbone.training.loss_probe  import LossScaleProbeConfig
from pipelines.backbone.training.pipeline    import TrainingPipeline
from pipelines.backbone.training.trainer     import Trainer
from pipelines.shared.config.config_factory  import ConfigFactory
from pipelines.shared.model.model_builder    import ModelBuilder
from pipelines.shared.training.seed_sweep      import SeedSweepRunner
from pipelines.shared.training.training_runner import SingleTrainRunner as BaseSingleTrainRunner
from tools.data.gaussians     import GaussianAxis, GaussianHead
from tools.orchestration      import ExperimentStage, GpuJob
from tools.monitoring.logger  import Logger
from tools.runtime.config_cli import ConfigCli


class SingleTrainRunner(BaseSingleTrainRunner):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.factory = ConfigFactory(config)

    @property
    def label(self) -> str:
        return self.config.backbone_name

    def _build_pretrain_trainer(self, logger):
        work_dir = Path(self.config.logdir) / "pretrain" / "context"

        trainer_config            = self.factory.training_trainer_config(logdir=work_dir)
        trainer_config.curriculum = self.config.curriculum
        trainer_config.geometry   = self.config.geometry.resolved(self.config.paths.dataset_path, secondary_labels=self.factory._secondary_labels())

        dataset_config              = self.factory.training_dataset_config()
        dataset_config.input_config = self.config.input

        gaussian_cfg               = trainer_config.gaussian
        dataset_config.n_gaussians = gaussian_cfg.n_default_gaussians

        dataset_pipeline      = DatasetPipeline(config=dataset_config, training_run_directory=work_dir, logger=logger, seed=self.config.seed, height_axis_convention=trainer_config.geometry.height_axis_convention, build_geometry_field=TrainingPipeline.physics_geometry_active(trainer_config))
        profile_length        = dataset_pipeline.layout.profile_length
        dataset_config.x_axis = GaussianAxis.build(gaussian_cfg.x_min, gaussian_cfg.x_max, profile_length)

        _train_loader, _val_loader, _test_loader, datasets = dataset_pipeline.run()
        dataset                                            = datasets["train"]

        model_config = ModelBuilder.config_from_registry(self.config.backbone_name, self.config.model_overrides)

        in_channels  = dataset.input_channels
        out_channels = GaussianHead.total_channels(gaussian_cfg.params_per_gaussian, gaussian_cfg.n_default_gaussians)

        overrides = {"in_channels": in_channels, "out_channels": out_channels}
        if self.config.backbone_name in BACKBONE_IMAGE_SIZE_MODELS:
            overrides["image_size"] = dataset_config.patch.size[0]

        model, model_cfg = get_backbone(self.config.backbone_name, config=model_config, **overrides)

        trainer = Trainer(model=model, model_cfg=model_cfg, x_axis=dataset_config.x_axis, config=trainer_config, run_dir=work_dir, logger=logger, norm_stats=dataset.normalizer, emit_docs=False)
        trainer.criterion.set_curriculum(trainer_config.curriculum.complete)

        return trainer, dataset, model

    def _probe_config(self) -> LossScaleProbeConfig:
        return LossScaleProbeConfig(
            enabled        = self.config.probe_enabled,
            n_batches      = self.config.probe_n_batches,
            reference      = self.config.probe_reference,
            exit_after     = self.config.probe_exit_after,
            enabled_losses = {},
        )

    def _run_inference(self, run_directory: Path):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inference_config = replace(self.config.inference, run_directory=Path(run_directory), output_subdir=None)

        return InferencePipeline(inference_config).run()

    def run(self):
        self._pretrain_preflight()

        trainer_config            = self.factory.training_trainer_config(logdir=self.config.logdir)
        trainer_config.curriculum = self.config.curriculum
        trainer_config.geometry   = self.config.geometry.resolved(self.config.paths.dataset_path, secondary_labels=self.factory._secondary_labels())
        trainer_config.memory.adopt_reservation(self.config.pretrain)

        model_config = ModelBuilder.config_from_registry(self.config.backbone_name, self.config.model_overrides)

        dataset_config              = self.factory.training_dataset_config()
        dataset_config.input_config = self.config.input

        pipeline = TrainingPipeline(
            trainer_config = trainer_config,
            dataset_config = dataset_config,
            backbone_name  = self.config.backbone_name,
            model_config   = model_config,
            seed           = self.config.seed,
            run_name       = self.config.run_name,
        )

        results = pipeline.run(probe_config=self._probe_config())

        if self.config.infer_after:
            self._run_inference(pipeline.run_metadata.run_directory)

        return results


class TrainScheduler:

    SCHEDULER_FIELDS = ("trials_enabled", "trials_mode", "warmup_losses", "complete_losses", "presence_trials", "secondary_trials", "patch_trials", "input_trials", "ablation_features", "ablation_include_full", "gpus", "poll_interval_s")

    MODE_SUBDIRS = {
        "curriculum" : "curriculum",
        "warmup"     : "warmup",
        "presence"   : "presence",
        "secondary"  : "secondary",
        "patch"      : "patch",
        "input"      : "input",
        "ablation"   : "ablation",
    }

    def __init__(self, config, cli_overrides: dict, entry_script: Path) -> None:
        self.config       = config
        self.entry_script = Path(entry_script)

        base_logdir       = Path(config.logdir)
        subdir            = self.MODE_SUBDIRS.get(config.trials_mode)
        self.runs_root    = base_logdir / subdir if subdir else base_logdir

        self.log_dir      = self.runs_root / "batch_train_logs"
        self.results_path = self.log_dir / "train_scheduler_results.json"

        self.forward_overrides = {path: value for path, value in cli_overrides.items() if path.split(".")[0] not in self.SCHEDULER_FIELDS}

        self.logger = Logger(log_dir=str(self.log_dir), name="train_scheduler")
        self.stage  = ExperimentStage(config=config, run_tag="batch_train", logger=self.logger, entry_script=self.entry_script)

    def planner(self):
        mode = self.config.trials_mode

        if mode == "curriculum":
            return CurriculumTrialPlanner(self.config.backbone_name, self.config.warmup_losses, self.config.complete_losses)
        if mode == "warmup":
            return WarmupTrialPlanner(self.config.backbone_name, self.config.warmup_losses)
        if mode == "presence":
            return SlotPresenceTrialPlanner(self.config.backbone_name, self.config.presence_trials)
        if mode == "secondary":
            return SecondaryTrialPlanner.from_dataset(self.config.backbone_name, self.config.secondary_trials, self.config.geometry, self.config.paths.dataset_path)
        if mode == "patch":
            return PatchSizeTrialPlanner(self.config.backbone_name, self.config.patch_trials)
        if mode == "input":
            return InputTrialPlanner.from_dataset(self.config.backbone_name, self.config.input_trials, self.config.geometry, self.config.paths.dataset_path)
        if mode == "ablation":
            return AblationTrialPlanner(self.config.backbone_name, self.config.ablation_features, self.config.ablation_include_full)

        raise ValueError(f"Unknown trials_mode '{mode}', expected 'curriculum', 'warmup', 'presence', 'secondary', 'patch', 'input' or 'ablation'")

    def _job(self, run_name: str, overrides: dict) -> GpuJob:
        argv = ConfigCli.to_argv({**self.forward_overrides, **overrides, "run_name": run_name, "logdir": str(self.runs_root)})

        return GpuJob(
            name     = run_name,
            command  = [sys.executable, str(self.entry_script), "--trial"] + argv,
            log_path = self.log_dir / f"{run_name}.log",
        )

    def run(self) -> None:
        planner     = self.planner()
        experiments = planner.plan()

        self.logger.section(f"Training trials: {self.config.trials_mode}")
        self.logger.kv_table({
            "Model"         : self.config.backbone_name,
            "Mode"          : self.config.trials_mode,
            **planner.summary(),
            "Trials"        : len(experiments),
            "GPUs"          : self.config.gpus,
            "Infer after"   : self.config.infer_after,
            "CLI overrides" : self.forward_overrides or "—",
            "Log dir"       : str(self.log_dir),
        }, title="Configuration")

        jobs    = [self._job(run_name, overrides) for run_name, overrides in experiments]
        names   = [run_name for run_name, overrides in experiments]
        results = self.stage._order_results(self.stage._run_queue(jobs), names)

        self.stage._write_results(results, self.results_path)

        self.logger.section("Summary")
        rows = [{"Trial": r["name"], "Status": r["status"], "Duration": f"{r['duration_s'] / 60:.1f} min"} for r in results]
        self.logger.metrics_table(rows, columns=["Trial", "Status", "Duration"])

        failed = [r for r in results if r["status"] != "DONE"]
        self.stage._log_failures(failed)

        self.logger.close()


class BackboneTrainingLauncher:

    def __init__(self, entry_script: Path) -> None:
        self.entry_script = Path(entry_script)

    def run(self, argv: list[str] | None = None) -> None:
        argv = list(sys.argv[1:] if argv is None else argv)

        cli    = ConfigCli(BackboneEntryConfig(), description="Train one backbone end to end, or fan out curriculum/warmup/secondary trials across GPUs")
        config = cli.apply(argv)

        trial_parser = argparse.ArgumentParser(add_help=False)
        trial_parser.add_argument("--trial", action="store_true")
        trial, _ = trial_parser.parse_known_args(argv)

        if trial.trial or not config.trials_enabled:
            SeedSweepRunner(config, SingleTrainRunner).run()
            return

        TrainScheduler(config=config, cli_overrides=cli.overrides, entry_script=self.entry_script).run()
