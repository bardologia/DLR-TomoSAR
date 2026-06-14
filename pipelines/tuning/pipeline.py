from __future__ import annotations

import sys
from pathlib import Path

import optuna
from optuna.trial import TrialState

from configuration.experiments.benchmark_config import TrainingQueueConfig
from configuration.data.dataset_config          import DatasetConfig
from configuration.data.dataset_config          import PatchConfig
from configuration.data.dataset_config          import SplitRegions
from configuration.sar.gaussian_config          import GaussianConfig
from configuration.sar.geometry_config          import GeometryConfig
from configuration.training.loss_config         import LossConfig, LossCurriculumConfig
from configuration.training.optimization_config import EarlyStoppingConfig, GradientClipperConfig, OptimizerConfig, SchedulerConfig, WarmupConfig
from configuration.training.runtime_config      import IOConfig, TrainingLoopConfig
from configuration.training.training_config     import TrainerConfig
from configuration.experiments.tuning_config    import TuningConfig
from models                                     import CONFIG_REGISTRY
from tools                                      import FileIO
from tools                                      import GpuJob
from tools                                      import GpuQueue
from pipelines.tuning.plots                     import StudyPlotter
from pipelines.tuning.tuners                    import AeTuner
from pipelines.tuning.tuners                    import BestConfigWriter
from pipelines.tuning.tuners                    import Tuner
from tools.runtime.config_cli                   import ConfigCli
from tools.monitoring.logger                    import Logger
from tools.data.regions                         import CropRegion


class TuningOrchestrator:
    def __init__(self, tag: str, config, entry_script: Path) -> None:
        self.tag          = tag
        self.config       = config
        self.entry_script = Path(entry_script)
        self.run_dir      = Path(config.paths.log_base_dir) / tag
        self.storage_url  = f"sqlite:///{self.run_dir / 'optuna.db'}"
        self.summary_path = self.run_dir / "tuning_results.json"
        self.db_path      = self.run_dir / "optuna.db"

        self.logger  = None
        self.results = []

        self._quiet_optuna()

    def _quiet_optuna(self) -> None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _distribute_trials(self, total: int, n_workers: int) -> list[int]:
        base  = total // n_workers
        extra = total % n_workers
        return [base + (1 if i < extra else 0) for i in range(n_workers)]

    def _study_name(self, model_name: str) -> str:
        return f"{model_name}_{self.tag}"

    def _registry(self) -> dict:
        if self.config.training_type == "autoencoder":
            from models.autoencoder import AE_CONFIG_REGISTRY
            return AE_CONFIG_REGISTRY
        return CONFIG_REGISTRY

    def _search_space(self, model_name: str) -> dict:
        config_cls = self._registry()[model_name]
        return {**config_cls.tunable_lr_params(), **config_cls.tunable_arch_params()}

    def _ae_entry_template(self):
        from configuration.training.autoencoder_config import ProfileAeEntryConfig

        return ProfileAeEntryConfig(
            seed            = self.config.tuning.base_seed,
            n_gaussians     = self.config.n_gaussians,
            pixel_subsample = self.config.pixel_subsample,
            keep_empty_frac = self.config.keep_empty_frac,
            ae_loss         = self.config.ae_loss,
            overfit         = self.config.overfit,
            paths           = self.config.paths,
            training        = self.config.training,
        )

    def _load_layout(self, dataset_path: Path) -> dict:
        return FileIO.load_json(dataset_path / "data" / "dataset.json")

    def _split_regions(self, global_crop: CropRegion) -> SplitRegions:
        canonical = TrainingQueueConfig()

        return SplitRegions(
            train = CropRegion(canonical.train_azimuth[0], canonical.train_azimuth[1], global_crop.range_start, global_crop.range_end),
            val   = CropRegion(canonical.val_azimuth[0],   canonical.val_azimuth[1],   global_crop.range_start, global_crop.range_end),
            test  = CropRegion(canonical.test_azimuth[0],  canonical.test_azimuth[1],  global_crop.range_start, global_crop.range_end),
        )

    def _dataset_config(self, dataset_path: Path, split_regions: SplitRegions, secondary_labels) -> DatasetConfig:
        return DatasetConfig(
            preprocessing_run_directory = dataset_path,
            parameters_path             = self.config.paths.parameters_path,
            split_regions               = split_regions,
            secondary_labels            = secondary_labels,
            patch                       = PatchConfig(size=(64, 64), stride=32, use_reflective_padding=True),
            batch_size                  = self.config.batch_size,
            num_workers                 = self.config.num_workers,
            shuffle_train               = True,
            pin_memory                  = True,
        )

    def _trainer_config(self, dataset_path: Path, secondary_labels) -> TrainerConfig:
        return TrainerConfig(
            gaussian         = GaussianConfig.from_dataset(dataset_path, n_gaussians=5),
            geometry         = GeometryConfig().resolved(dataset_path, secondary_labels=secondary_labels),
            early_stopping   = EarlyStoppingConfig(patience=8, restore_best=True),
            warmup           = WarmupConfig(warmup_steps=self.config.warmup_steps, warmup_start_factor=0.1, warmup_enabled=True, warmup_mode="linear"),
            scheduler        = SchedulerConfig(type="cosine_annealing", epochs=60, eta_min=self.config.eta_min),
            optimizer        = OptimizerConfig(betas=(0.9, 0.999), eps=1e-8),
            gradient_clipper = GradientClipperConfig(clip_mode="fixed", max_grad_norm=1.0),
            io               = IOConfig(logdir=str(self.config.paths.log_base_dir)),
            training         = TrainingLoopConfig(epochs=60, validation_frequency=1, gradient_accumulation_steps=1),

            curriculum = LossCurriculumConfig(
                enabled  = False,
                warmup   = LossConfig(use_param_l1=True, weight_param_l1=1.0, param_weights=(1.0, 1.0, 1.0)),
                complete = LossConfig(use_param_l1=True, weight_param_l1=1.0),
            ),
        )

    def _build_base_configs(self):
        dataset_path = Path(self.config.paths.dataset_path)

        layout      = self._load_layout(dataset_path)
        global_crop = CropRegion(*layout["global_crop"])

        secondary_labels = tuple(self.config.paths.secondary_labels) if self.config.paths.secondary_labels else None

        split_regions  = self._split_regions(global_crop)
        dataset_config = self._dataset_config(dataset_path, split_regions, secondary_labels)
        trainer_config = self._trainer_config(dataset_path, secondary_labels)

        return trainer_config, dataset_config

    def _worker_job(self, model_name: str, n_trials: int, worker_index: int) -> GpuJob:
        command = [
            sys.executable, str(self.entry_script),
            "--worker",
            "--model",        model_name,
            "--n-trials",     str(n_trials),
            "--study-name",   self._study_name(model_name),
            "--storage-url",  self.storage_url,
            "--run-tag",      self.tag,
            "--run-dir",      str(self.run_dir),
        ]

        return GpuJob(
            name     = f"{model_name}_w{worker_index}",
            command  = command,
            log_path = self.run_dir / model_name / f"worker{worker_index}.log",
        )

    def _dispatch_workers(self, model_name: str, trial_counts: list[int]) -> bool:
        jobs = []
        for worker_index, n_trials in enumerate(trial_counts):
            if n_trials == 0:
                continue

            jobs.append(self._worker_job(model_name, n_trials, worker_index))

        queue   = GpuQueue(gpus=self.config.gpus, logger=self.logger)
        results = queue.run(jobs)

        return all(result.status == "DONE" for result in results)

    def _load_or_create_study(self, model_name: str) -> optuna.Study:
        return optuna.create_study(
            study_name     = self._study_name(model_name),
            storage        = self.storage_url,
            direction      = "minimize",
            load_if_exists = True,
        )

    def _fail_stale_trials(self, study: optuna.Study) -> int:
        stale = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))

        for trial in stale:
            study._storage.set_trial_state_values(trial._trial_id, state=TrialState.FAIL)

        return len(stale)

    def _count_done(self, study: optuna.Study) -> int:
        return len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED)))

    def _count_state(self, study: optuna.Study, state: TrialState) -> int:
        return len(study.get_trials(deepcopy=False, states=(state,)))

    def _save_results(self) -> None:
        FileIO.save_json(self.results, self.summary_path, indent=2)

    def _plot_study(self, study: optuna.Study, model_name: str) -> None:
        plot_dir = self.run_dir / model_name / "study_plots"
        plotter  = StudyPlotter(self.logger)
        saved    = plotter.render(study, plot_dir)
        self.logger.info(f"Saved {len(saved)} study plots to {plot_dir}")

    def _record(self, model_name: str, status: str, study: optuna.Study, payload, path) -> dict:
        return {
            "model"           : model_name,
            "status"          : status,
            "trials_completed": self._count_state(study, TrialState.COMPLETE),
            "trials_pruned"   : self._count_state(study, TrialState.PRUNED),
            "trials_failed"   : self._count_state(study, TrialState.FAIL),
            "val_loss"        : payload["val_loss"] if payload is not None else None,
            "best_config"     : str(path) if path is not None else None,
        }

    def _tune_model(self, model_name: str, tune_cfg: TuningConfig) -> None:
        self.logger.section(f"[{model_name}]")

        study = self._load_or_create_study(model_name)
        stale = self._fail_stale_trials(study)
        if stale > 0:
            self.logger.info(f"Marked {stale} stale running trials as failed")

        done      = self._count_done(study)
        remaining = max(0, tune_cfg.n_trials - done)
        n_gpus    = len(self.config.gpus)
        ok        = True

        if remaining == 0:
            self.logger.subsection(f"Target reached — {done}/{tune_cfg.n_trials} trials already complete")
        else:
            self.logger.subsection(f"{done}/{tune_cfg.n_trials} trials done — running {remaining} across {n_gpus} GPUs")
            counts = self._distribute_trials(remaining, n_gpus)
            ok     = self._dispatch_workers(model_name, counts)

        writer  = BestConfigWriter(model_name, self._search_space(model_name), self.run_dir / model_name / "best_config.json")
        payload = writer.write(study)

        if payload is None:
            self.logger.error(f"No completed trials for {model_name}")
            self.results.append(self._record(model_name, "FAILED", study, None, None))
            self._save_results()
            return

        self.logger.subsection(f"Best — trial {payload['trial']}  best validation loss (normalized param L1) = {payload['val_loss']:.6f}")
        self.logger.kv_table(payload["params"], title="Best Params")

        if tune_cfg.emit_study_plots:
            self._plot_study(study, model_name)

        self.results.append(self._record(model_name, "DONE" if ok else "PARTIAL", study, payload, writer.path))
        self._save_results()

    def schedule(self, target_model: str | None = None, resume: bool = False) -> None:
        if resume and not self.db_path.exists():
            sys.exit(f"ERROR: --resume given but no study found at {self.db_path}")
        if not resume and self.db_path.exists():
            sys.exit(f"ERROR: run tag '{self.tag}' already has a study at {self.db_path} — pass --resume to continue it")

        tune_cfg = self.config.tuning
        self.run_dir.mkdir(parents=True, exist_ok=True)
        ConfigCli.save_resolved(self.config, self.run_dir / "resolved_config.json")

        self.logger = Logger(log_dir=str(self.run_dir), name="tune_scheduler")

        registry   = self._registry()
        all_models = list(registry.keys())
        if target_model is not None:
            if target_model not in registry:
                self.logger.close()
                sys.exit(f"ERROR: unknown model '{target_model}'. Available: {list(registry.keys())}")
            models_to_run = [target_model]
        else:
            models_to_run = [m for m in all_models if m not in set(self.config.skip_models)]

        self.logger.section("Hyperparameter Tuning")
        self.logger.kv_table({
            "Run tag"      : self.tag,
            "Mode"         : "resume" if resume else "new",
            "Models"       : len(models_to_run),
            "GPUs"         : self.config.gpus,
            "Trial target" : tune_cfg.n_trials,
            "Storage"      : self.storage_url,
            "Log dir"      : str(self.run_dir),
        }, title="Configuration")

        for model_name in models_to_run:
            self._tune_model(model_name, tune_cfg)

        self.logger.section("Summary")
        self.logger.kv_table({
            "Models tuned" : len(self.results),
            "Results saved": str(self.summary_path),
        }, title="Done")
        self.logger.close()

    def run_worker(
        self,
        model_name : str,
        gpu_id     : int,
        n_trials   : int,
        study_name : str,
        storage_url: str,
    ) -> None:
        tune_cfg      = self.config.tuning
        model_log_dir = str(self.run_dir / model_name / f"worker_gpu{gpu_id}")
        logger        = Logger(log_dir=model_log_dir, name=f"tune_worker_gpu{gpu_id}_{model_name}")

        logger.section(f"[GPU {gpu_id}] Tuning Worker — {model_name}")
        logger.kv_table({
            "GPU"        : gpu_id,
            "Trials"     : n_trials,
            "Study"      : study_name,
            "Storage"    : storage_url,
        })

        if self.config.training_type == "autoencoder":
            from models.autoencoder import AE_CONFIG_REGISTRY

            tuner = AeTuner(
                model_name     = model_name,
                config_cls     = AE_CONFIG_REGISTRY[model_name],
                entry_template = self._ae_entry_template(),
                tune_cfg       = tune_cfg,
                log_dir        = str(self.run_dir / model_name),
                logger         = logger,
            )
        else:
            trainer_cfg, dataset_cfg = self._build_base_configs()

            tuner = Tuner(
                model_name          = model_name,
                model_config_cls    = CONFIG_REGISTRY[model_name],
                base_trainer_config = trainer_cfg,
                base_dataset_config = dataset_cfg,
                tune_cfg            = tune_cfg,
                log_dir             = str(self.run_dir / model_name),
                logger              = logger,
                emit_trial_docs     = tune_cfg.emit_trial_docs,
            )

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials = tune_cfg.pruner_n_startup_trials,
            n_warmup_steps   = tune_cfg.pruner_n_warmup_steps,
        )

        sampler = optuna.samplers.TPESampler(
            n_startup_trials = tune_cfg.pruner_n_startup_trials,
            multivariate     = True,
            constant_liar    = True,
            seed             = tune_cfg.base_seed + gpu_id,
        )

        study = optuna.load_study(study_name=study_name, storage=storage_url, sampler=sampler, pruner=pruner)
        tuner.run(study, n_trials)

        logger.info(f"[GPU {gpu_id}] — {model_name}  DONE")
        logger.close()
