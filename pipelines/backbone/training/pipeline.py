from __future__ import annotations

import gc
import sys
from dataclasses import replace
from pathlib     import Path

import numpy as np
import torch

from models                                  import BACKBONE_IMAGE_SIZE_MODELS
from tools.data.gaussians                    import GaussianHead
from configuration.dataset import DatasetConfig
from configuration.training import BackboneTrainerConfig
from pipelines.backbone.dataset.pipeline     import DatasetPipeline
from pipelines.backbone.inference.pipeline   import InferencePipeline
from tools.orchestration                     import ExperimentStage, GpuJob
from pipelines.backbone.training.loss_probe  import LossScaleProbeConfig
from pipelines.backbone.training.experiments import CurriculumTrialPlanner, InputTrialPlanner, PatchSizeTrialPlanner, SecondaryTrialPlanner, SlotPresenceTrialPlanner, WarmupTrialPlanner
from pipelines.backbone.training.trainer     import Trainer
from pipelines.shared.run_metadata           import TrainingRunMetadata
from tools.runtime.config_cli                import ConfigCli
from tools.monitoring.logger                 import Logger
from tools.runtime.reproducibility           import Reproducibility


def physics_geometry_active(trainer_config) -> bool:
    curriculum = trainer_config.curriculum
    loss_cfgs  = [curriculum.warmup] + ([curriculum.complete] if curriculum.enabled else [])
    flags      = ("use_coherence_resyn", "use_covariance_match", "use_capon_cycle")

    return any(getattr(cfg, flag, False) for cfg in loss_cfgs for flag in flags)


class TrainingPipeline:
    def __init__(
        self,
        trainer_config : BackboneTrainerConfig,
        dataset_config : DatasetConfig,
        backbone_name  : str,
        model_config   = None,
        seed           : int = 0,
        run_name       : str | None = None,
    ) -> None:

        patch_height, patch_width = dataset_config.patch.size

        self.trainer_config = trainer_config
        self.dataset_config = dataset_config
        self.backbone_name     = backbone_name
        self.model_config   = model_config
        self.image_size     = patch_height
        self.seed           = seed

        Reproducibility.seed_everything(self.seed)

        self.run_metadata = TrainingRunMetadata(
            trainer_config = trainer_config,
            model_name     = backbone_name,
            base_logdir    = Path(trainer_config.io.logdir),
            run_name       = run_name,
        )
        self.logger = self.run_metadata.logger

        self.dataset_pipeline = DatasetPipeline(
            config                 = dataset_config,
            training_run_directory = self.run_metadata.run_directory,
            logger                 = self.logger,
            seed                   = self.seed,
            height_axis_convention = trainer_config.geometry.height_axis_convention,
            build_geometry_field   = physics_geometry_active(trainer_config),
        )

    def _build_model(self, in_channels: int, out_channels: int):
        from models import get_backbone

        overrides = {"in_channels": in_channels, "out_channels": out_channels}
        if self.backbone_name in BACKBONE_IMAGE_SIZE_MODELS:
            overrides["image_size"] = self.image_size

        model, model_cfg = get_backbone(self.backbone_name, config=self.model_config, **overrides)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.section("[Model Built]")
        self.logger.subsection(f"Architecture : {self.backbone_name}")
        self.logger.subsection(f"In Channels  : {in_channels}")
        self.logger.subsection(f"Out Channels : {out_channels}")
        self.logger.subsection(f"Parameters   : {n_params:,}")
        return model, model_cfg

    def _make_trainer(self, model, model_cfg, x_axis, norm_stats):
        return Trainer(
            model      = model,
            model_cfg  = model_cfg,
            x_axis     = x_axis,
            config     = self.trainer_config,
            run_dir    = self.run_metadata.run_directory,
            logger     = self.logger,
            norm_stats = norm_stats,
        )

    def run(self, probe_config=None):
        self.logger.section("[PyTorch Training Pipeline Execution]")

        gaussian_cfg                    = self.trainer_config.gaussian
        self.dataset_config.n_gaussians = gaussian_cfg.n_default_gaussians

        x_axis_length = self.dataset_pipeline.layout.profile_length

        self.dataset_config.x_axis = np.linspace(gaussian_cfg.x_min, gaussian_cfg.x_max, x_axis_length, dtype=np.float32)

        train_loader, val_loader, test_loader, datasets = self.dataset_pipeline.run()

        train_dataset = datasets["train"]
        in_channels   = train_dataset.input_channels
        n_gaussians   = gaussian_cfg.n_default_gaussians
        out_channels  = GaussianHead.total_channels(gaussian_cfg.params_per_gaussian, n_gaussians, gaussian_cfg.predict_presence)
        x_axis        = np.asarray(self.dataset_config.x_axis, dtype=np.float32)

        model, model_cfg = self._build_model(in_channels=in_channels, out_channels=out_channels)

        self.run_metadata.save_trainer_config()
        self.run_metadata.save_model_config(model_cfg, self.backbone_name)

        self.run_metadata.save_run_summary(
            model_name       = self.backbone_name,
            in_channels      = in_channels,
            out_channels     = out_channels,
            x_axis_length    = x_axis_length,
            n_gaussians      = n_gaussians,
            predict_presence = gaussian_cfg.predict_presence,
        )

        trainer = self._make_trainer(model, model_cfg, x_axis, train_dataset.normalizer)

        try:
            trainer.maybe_run_loss_probe(train_loader, probe_config)
            results = trainer.train(train_loader, val_loader, test_loader)
        finally:
            self.run_metadata.close()

        return results


class SingleTrainRunner:
    def __init__(self, config) -> None:
        from pipelines.shared.config_factory import ConfigFactory

        self.config  = config
        self.factory = ConfigFactory(config)

    def _pretrain_preflight(self) -> None:
        from pipelines.shared.pretrain_preflight import PretrainPreflight

        PretrainPreflight(
            pretrain_config = self.config.pretrain,
            training_config = self.config.training,
            build_context   = self._build_pretrain_context,
            logdir          = Path(self.config.logdir),
            label           = self.config.backbone_name,
        ).run()

    def _build_pretrain_context(self, logger, device):
        from models                     import BACKBONE_CONFIG_REGISTRY, get_backbone
        from tools.training.pretraining  import PretrainContext, TrainStepMemoryProbe, TrainerFeed

        work_dir = Path(self.config.logdir) / "pretrain" / "context"

        trainer_config            = self.factory.training_trainer_config(logdir=work_dir)
        trainer_config.curriculum = self.config.curriculum
        trainer_config.geometry   = self.config.geometry.resolved(self.config.paths.dataset_path, secondary_labels=self.factory._secondary_labels())

        dataset_config              = self.factory.training_dataset_config()
        dataset_config.input_config = self.config.input

        gaussian_cfg               = trainer_config.gaussian
        dataset_config.n_gaussians = gaussian_cfg.n_default_gaussians

        dataset_pipeline      = DatasetPipeline(config=dataset_config, training_run_directory=work_dir, logger=logger, seed=self.config.seed, height_axis_convention=trainer_config.geometry.height_axis_convention, build_geometry_field=physics_geometry_active(trainer_config))
        profile_length        = dataset_pipeline.layout.profile_length
        dataset_config.x_axis = np.linspace(gaussian_cfg.x_min, gaussian_cfg.x_max, profile_length, dtype=np.float32)

        _train_loader, _val_loader, _test_loader, datasets = dataset_pipeline.run()
        dataset                                            = datasets["train"]

        model_config = BACKBONE_CONFIG_REGISTRY[self.config.backbone_name]()
        for attribute, value in self.config.model_overrides.items():
            setattr(model_config, attribute, value)

        in_channels  = dataset.input_channels
        out_channels = GaussianHead.total_channels(gaussian_cfg.params_per_gaussian, gaussian_cfg.n_default_gaussians, gaussian_cfg.predict_presence)

        overrides = {"in_channels": in_channels, "out_channels": out_channels}
        if self.config.backbone_name in BACKBONE_IMAGE_SIZE_MODELS:
            overrides["image_size"] = dataset_config.patch.size[0]

        model, model_cfg = get_backbone(self.config.backbone_name, config=model_config, **overrides)

        trainer = Trainer(model=model, model_cfg=model_cfg, x_axis=dataset_config.x_axis, config=trainer_config, run_dir=work_dir, logger=logger, norm_stats=dataset.normalizer, emit_docs=False)
        trainer.criterion.set_curriculum(trainer_config.curriculum.complete)
        trainer.model.train()

        feed = TrainerFeed(trainer)

        return PretrainContext(
            dataset        = dataset,
            model          = model,
            to_model_input = feed.to_model_input,
            forward_loss   = feed.forward_loss,
            trial_step     = TrainStepMemoryProbe(trainer, dataset, self.config.pretrain.measure_steps, device, 0.0),
            run_overfit    = self._overfit_loss,
            device         = device,
            use_amp        = trainer.use_amp,
            context_gb     = 0.0,
            on_oom         = lambda: self._release(trainer),
        )

    def _release(self, trainer) -> None:
        trainer.optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _overfit_loss(self):
        from configuration.training      import OverfitConfig
        from models                      import BACKBONE_CONFIG_REGISTRY
        from pipelines.benchmark.workers import OverfitModelPreparer

        pretrain = self.config.pretrain
        work_dir = Path(self.config.logdir) / "pretrain" / "overfit"

        trainer_config            = self.factory.training_trainer_config(logdir=work_dir)
        trainer_config.curriculum = self.config.curriculum
        trainer_config.geometry   = self.config.geometry.resolved(self.config.paths.dataset_path, secondary_labels=self.factory._secondary_labels())
        trainer_config.overfit    = OverfitConfig(enabled=True, max_steps=pretrain.overfit_max_steps, stop_threshold=pretrain.overfit_stop_threshold, batch_size=pretrain.overfit_batch_size)

        model_config = BACKBONE_CONFIG_REGISTRY[self.config.backbone_name]()
        for attribute, value in self.config.model_overrides.items():
            setattr(model_config, attribute, value)
        model_config = OverfitModelPreparer(model_config).prepare()

        dataset_config              = self.factory.training_dataset_config()
        dataset_config.input_config = self.config.input

        pipeline = TrainingPipeline(
            trainer_config = trainer_config,
            dataset_config = dataset_config,
            backbone_name  = self.config.backbone_name,
            model_config   = model_config,
            seed           = pretrain.seed,
            run_name       = f"{self.config.backbone_name}_pretrain_overfit",
        )

        outputs      = pipeline.run(probe_config=self._probe_config())
        train_losses = outputs[0] if isinstance(outputs, (tuple, list)) and outputs else None

        return float(train_losses[-1]) if train_losses else None

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
        from models import BACKBONE_CONFIG_REGISTRY

        self._pretrain_preflight()

        trainer_config            = self.factory.training_trainer_config(logdir=self.config.logdir)
        trainer_config.curriculum = self.config.curriculum
        trainer_config.overfit    = self.config.overfit
        trainer_config.geometry   = self.config.geometry.resolved(self.config.paths.dataset_path, secondary_labels=self.factory._secondary_labels())

        model_config = BACKBONE_CONFIG_REGISTRY[self.config.backbone_name]()
        for attribute, value in self.config.model_overrides.items():
            setattr(model_config, attribute, value)

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

    SCHEDULER_FIELDS = ("trials_enabled", "trials_mode", "warmup_losses", "complete_losses", "presence_trials", "secondary_trials", "patch_trials", "input_trials", "gpus", "poll_interval_s")

    def __init__(self, config, cli_overrides: dict, entry_script: Path, stage: str) -> None:
        self.config       = config
        self.entry_script = Path(entry_script)
        self.stage_name   = stage
        self.log_dir      = Path(config.logdir) / "batch_train_logs"
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

        raise ValueError(f"Unknown trials_mode '{mode}', expected 'curriculum', 'warmup', 'presence', 'secondary', 'patch' or 'input'")

    def _job(self, run_name: str, overrides: dict) -> GpuJob:
        argv = ConfigCli.to_argv({**self.forward_overrides, **overrides, "run_name": run_name})

        return GpuJob(
            name     = run_name,
            command  = [sys.executable, str(self.entry_script), "--mode", self.stage_name, "--trial"] + argv,
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
