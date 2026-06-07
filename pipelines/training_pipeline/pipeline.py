from __future__ import annotations

import sys
from dataclasses import asdict, replace
from datetime    import datetime
from pathlib     import Path

import numpy as np
import torch

from tensorboard.summary.writer.event_file_writer import EventFileWriter as _  # noqa: F401
from torch.utils.tensorboard import SummaryWriter

from configuration.dataset_config            import DatasetConfiguration
from configuration.training_config           import TrainerConfig
from pipelines.dataset_pipeline.pipeline     import DatasetPipeline
from pipelines.shared.io                     import FileIO
from pipelines.shared.orchestration          import GpuJob, GpuQueue
from pipelines.training_pipeline.docs        import LossScaleProbeConfig
from pipelines.training_pipeline.experiments import CurriculumTrialPlanner, SecondaryTrialPlanner, WarmupTrialPlanner
from pipelines.training_pipeline.trainer     import Trainer
from tools.config_cli                        import ConfigCli
from tools.logger                            import Logger

_IMAGE_SIZE_MODELS = {"swin_unet", "transunet", "unetr"}


class TrainingRunMetadata:
    def __init__(self, trainer_config : TrainerConfig, model_name : str, base_logdir : Path, run_name : str | None = None, logger : Logger | None = None) -> None:
        self.trainer_config = trainer_config
        self.model_name     = model_name
        self.base_logdir    = Path(base_logdir)

        timestamp           = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved_name       = run_name or f"run_{model_name}_{timestamp}"

        self.run_directory       = self.base_logdir / resolved_name
        self.tensorboard_dir     = self.run_directory / "tensorboard"
        self.docs_directory      = self.run_directory / "docs"
        self.logs_directory      = self.run_directory / "logs"
        self.metadata_directory  = self.run_directory / "meta"
        self.checkpoint_dir      = self.run_directory / "checkpoints"

        FileIO.ensure_dirs(
            self.run_directory, self.tensorboard_dir, self.docs_directory,
            self.logs_directory, self.metadata_directory,
            self.checkpoint_dir,
        )

        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))

        trainer_config.io.logdir     = str(self.run_directory)
        trainer_config.io.tb_dir     = str(self.tensorboard_dir)
        trainer_config.io.docs_dir   = str(self.docs_directory)
        trainer_config.io.logs_dir   = str(self.logs_directory)
        trainer_config.io.writer     = self.writer

        if hasattr(trainer_config, "resources"):
            trainer_config.resources.logs_dir = str(self.logs_directory)

        self.logger = logger or Logger(log_dir = str(self.logs_directory), name = f"{model_name}_metadata", level = "INFO",)

        self.logger.section("[Training RunMetadata Initialized]")
        devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.logger.kv_table({
            "Run Directory" : self.run_directory,
            "Model"         : self.model_name,
            "Backend"       : "PyTorch",
            "Devices"       : f"{devices} -> {[torch.cuda.get_device_name(i) for i in range(devices)]}",
        })

    def save_trainer_config(self) -> Path:
        out_path                     = self.docs_directory / "trainer_config.json"
        serializable                 = asdict(self.trainer_config)
        serializable["io"]["writer"] = None

        FileIO.save_json(serializable, out_path)
        self.logger.info(f"Trainer config saved: {out_path}")

        return out_path

    def save_run_summary(self, model_name: str, in_channels: int, out_channels: int, x_axis_length: int, param_match: str = "none") -> Path:
        out_path = self.metadata_directory / "run_summary.json"

        payload  = {
            "model_name"    : model_name,
            "in_channels"   : in_channels,
            "out_channels"  : out_channels,
            "x_axis_length" : x_axis_length,
            "run_directory" : str(self.run_directory),
            "framework"     : "pytorch",
            "n_devices"     : torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "param_match"   : param_match,
        }

        FileIO.save_json(payload, out_path)
        self.logger.info(f"Run summary saved: {out_path}")

        return out_path

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


class TrainingPipeline:
    def __init__(
        self,
        trainer_config : TrainerConfig,
        dataset_config : DatasetConfiguration,
        model_name     : str,
        model_config   = None,
        seed           : int = 0,
        run_name       : str | None = None,
    ) -> None:

        patch_height, patch_width = dataset_config.patch.size

        self.trainer_config = trainer_config
        self.dataset_config = dataset_config
        self.model_name     = model_name
        self.model_config   = model_config
        self.image_size     = patch_height
        self.seed           = seed

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True

        self.run_metadata = TrainingRunMetadata(
            trainer_config = trainer_config,
            model_name     = model_name,
            base_logdir    = Path(trainer_config.io.logdir),
            run_name       = run_name,
        )
        self.logger = self.run_metadata.logger

        self.dataset_pipeline = DatasetPipeline(
            config                 = dataset_config,
            training_run_directory = self.run_metadata.run_directory,
            logger                 = self.logger,
        )

    def _build_model(self, in_channels: int, out_channels: int):
        from models import get_model

        overrides = {"in_channels": in_channels, "out_channels": out_channels}
        if self.model_name in _IMAGE_SIZE_MODELS:
            overrides["image_size"] = self.image_size

        model, model_cfg = get_model(self.model_name, config=self.model_config, **overrides)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.section("[Model Built]")
        self.logger.subsection(f"Architecture : {self.model_name}")
        self.logger.subsection(f"In Channels  : {in_channels}")
        self.logger.subsection(f"Out Channels : {out_channels}")
        self.logger.subsection(f"Parameters   : {n_params:,}")
        return model, model_cfg

    def run(self, probe_config=None):
        self.logger.section("[PyTorch Training Pipeline Execution]")

        gaussian_cfg                    = self.trainer_config.gaussian
        self.dataset_config.n_gaussians = gaussian_cfg.n_default_gaussians

        tomo_path     = self.dataset_pipeline.layout.artifact_path("tomogram_full")
        tomo_mmap     = np.load(str(tomo_path), mmap_mode="r", allow_pickle=False)
        x_axis_length = int(tomo_mmap.shape[0])

        self.dataset_config.x_axis = np.linspace(gaussian_cfg.x_min, gaussian_cfg.x_max, x_axis_length, dtype=np.float32)

        train_loader, val_loader, test_loader, datasets = self.dataset_pipeline.run()

        train_dataset = datasets["train"]
        in_channels   = train_dataset.input_channels
        n_gaussians   = gaussian_cfg.n_default_gaussians
        out_channels  = gaussian_cfg.params_per_gaussian * n_gaussians
        x_axis        = np.asarray(self.dataset_config.x_axis, dtype=np.float32)

        model, model_cfg = self._build_model(in_channels=in_channels, out_channels=out_channels)

        self.run_metadata.save_trainer_config()

        self.run_metadata.save_run_summary(
            model_name    = self.model_name,
            in_channels   = in_channels,
            out_channels  = out_channels,
            x_axis_length = x_axis_length,
            param_match   = self.trainer_config.curriculum.warmup.param_match,
        )

        trainer = self._make_trainer(model, model_cfg, x_axis, getattr(train_dataset, "normalizer", None))

        try:
            trainer.maybe_run_loss_probe(train_loader, probe_config)
            results = trainer.train(train_loader, val_loader, test_loader)
        finally:
            self.run_metadata.close()
            self.logger.close()

        return results

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


class SingleTrainRunner:
    def __init__(self, config) -> None:
        from pipelines.benchmark_pipeline.config_factory import ConfigFactory

        self.config  = config
        self.factory = ConfigFactory(config)

    def run(self):
        from models import CONFIG_REGISTRY

        trainer_config            = self.factory.training_trainer_config(logdir=self.config.logdir)
        trainer_config.curriculum = self.config.curriculum
        trainer_config.overfit    = self.config.overfit
        trainer_config.geometry   = self.config.geometry.resolved(self.config.paths.dataset_path, secondary_labels=self.factory._secondary_labels())

        model_config = CONFIG_REGISTRY[self.config.model_name]()
        for attribute, value in self.config.model_overrides.items():
            setattr(model_config, attribute, value)

        pipeline = TrainingPipeline(
            trainer_config = trainer_config,
            dataset_config = self.factory.training_dataset_config(),
            model_name     = self.config.model_name,
            model_config   = model_config,
            seed           = self.config.seed,
            run_name       = self.config.run_name,
        )

        results = pipeline.run(probe_config=self._probe_config())

        if self.config.infer_after:
            self._run_inference(pipeline.run_metadata.run_directory)

        return results

    def _run_inference(self, run_directory: Path):
        import gc

        from pipelines.inference_pipeline.pipeline import InferencePipeline

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inference_config = replace(self.config.inference, run_directory=Path(run_directory), output_subdir=None)

        return InferencePipeline(inference_config).run()

    def _probe_config(self) -> LossScaleProbeConfig:
        return LossScaleProbeConfig(
            enabled        = self.config.probe_enabled,
            n_batches      = self.config.probe_n_batches,
            reference      = self.config.probe_reference,
            exit_after     = self.config.probe_exit_after,
            enabled_losses = {},
        )


class TrainScheduler:

    SCHEDULER_FIELDS = ("trials_enabled", "trials_mode", "warmup_losses", "complete_losses", "secondary_trials", "gpus", "poll_interval_s")

    def __init__(self, config, cli_overrides: dict, entry_script: Path) -> None:
        self.config       = config
        self.entry_script = Path(entry_script)
        self.log_dir      = Path(config.logdir) / "batch_train_logs"

        self.forward_overrides = {path: value for path, value in cli_overrides.items() if path.split(".")[0] not in self.SCHEDULER_FIELDS}

        self.logger = Logger(log_dir=str(self.log_dir), name="train_scheduler")

    def planner(self):
        mode = self.config.trials_mode

        if mode == "curriculum":
            return CurriculumTrialPlanner(self.config.model_name, self.config.warmup_losses, self.config.complete_losses)
        if mode == "warmup":
            return WarmupTrialPlanner(self.config.model_name, self.config.warmup_losses)
        if mode == "secondary":
            return SecondaryTrialPlanner.from_dataset(self.config.model_name, self.config.secondary_trials, self.config.geometry, self.config.paths.dataset_path)

        raise ValueError(f"Unknown trials_mode '{mode}', expected 'curriculum', 'warmup' or 'secondary'")

    def run(self) -> None:
        planner     = self.planner()
        experiments = planner.plan()

        self.logger.section(f"Training trials: {self.config.trials_mode}")
        self.logger.kv_table({
            "Model"         : self.config.model_name,
            "Mode"          : self.config.trials_mode,
            **planner.summary(),
            "Trials"        : len(experiments),
            "GPUs"          : self.config.gpus,
            "Infer after"   : self.config.infer_after,
            "CLI overrides" : self.forward_overrides or "—",
            "Log dir"       : str(self.log_dir),
        }, title="Configuration")

        queue   = GpuQueue(gpus=self.config.gpus, logger=self.logger, poll_interval_s=self.config.poll_interval_s)
        results = queue.run([self._job(run_name, overrides) for run_name, overrides in experiments])

        self.logger.section("Summary")
        rows = [{"Trial": r.name, "Status": r.status, "Duration": f"{r.duration_s / 60:.1f} min"} for r in results]
        self.logger.metrics_table(rows, columns=["Trial", "Status", "Duration"])

        self.logger.close()

    def _job(self, run_name: str, overrides: dict) -> GpuJob:
        argv = ConfigCli.to_argv({**self.forward_overrides, **overrides, "run_name": run_name})

        return GpuJob(
            name     = run_name,
            command  = [sys.executable, str(self.entry_script), "--trial"] + argv,
            log_path = self.log_dir / f"{run_name}.log",
        )
