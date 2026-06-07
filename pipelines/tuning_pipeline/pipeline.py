from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import optuna
from optuna.trial import TrialState

from configuration.tuning_config       import TuningConfig
from pipelines.shared.io               import FileIO
from pipelines.tuning_pipeline.tuners  import BestConfigWriter, Tuner


class TuningOrchestrator:
    def __init__(self, tag: str, config, entry_script: Path) -> None:
        self.tag          = tag
        self.config       = config
        self.entry_script = Path(entry_script)
        self.run_dir      = Path(config.paths.log_base_dir) / tag
        self.storage_url  = f"sqlite:///{self.run_dir / 'optuna.db'}"
        self.summary_path = self.run_dir / "tuning_results.json"
        self.db_path      = self.run_dir / "optuna.db"

        self.logger        = None
        self.results       = []
        self.active_procs  = []

    def _distribute_trials(self, total: int, n_workers: int) -> list[int]:
        base  = total // n_workers
        extra = total % n_workers
        return [base + (1 if i < extra else 0) for i in range(n_workers)]

    def _study_name(self, model_name: str) -> str:
        return f"{model_name}_{self.tag}"

    def _search_space(self, model_name: str) -> dict:
        from models import CONFIG_REGISTRY

        config_cls = CONFIG_REGISTRY[model_name]
        return {**config_cls.tunable_lr_params(), **config_cls.tunable_arch_params()}

    def _build_base_configs(self):
        from configuration.dataset_config import (
            DatasetConfiguration, PatchConfiguration,
            SplitRegions,
        )
        from configuration.training_config import (
            LossCurriculumConfig, EarlyStoppingConfig, EMAConfig, GaussianConfig,
            GradientClipperConfig, IOConfig, LossConfig, OptimizerConfig,
            SchedulerConfig, TrainerConfig, TrainingConfigInner, WarmupConfig,
        )
        from tools.regions import CropRegion

        dataset_path = Path(self.config.paths.dataset_path)

        with open(dataset_path / "data" / "dataset.json", "r", encoding="utf-8") as f:
            layout = json.load(f)
        global_crop = CropRegion(*layout["global_crop"])

        split_regions = SplitRegions(
            train = CropRegion(1000,  9120,  global_crop.range_start, global_crop.range_end),
            val   = CropRegion(9120,  12400, global_crop.range_start, global_crop.range_end),
            test  = CropRegion(12400, 16000, global_crop.range_start, global_crop.range_end),
        )

        dataset_config = DatasetConfiguration(
            preprocessing_run_directory = dataset_path,
            parameters_path             = self.config.paths.parameters_path,
            split_regions               = split_regions,
            patch                       = PatchConfiguration(size=(64, 64), stride=32, use_reflective_padding=True),
            batch_size                  = self.config.batch_size,
            num_workers                 = self.config.num_workers,
            shuffle_train               = True,
            pin_memory                  = True,
        )

        trainer_config = TrainerConfig(
            gaussian         = GaussianConfig.from_dataset(dataset_path, n_gaussians=5),
            early_stopping   = EarlyStoppingConfig(patience=8, min_delta=0.0001, restore_best=True),
            warmup           = WarmupConfig(warmup_steps=self.config.warmup_steps, warmup_start_factor=0.1, warmup_enabled=True, warmup_mode="linear"),
            scheduler        = SchedulerConfig(type="cosine_annealing", epochs=60, eta_min=self.config.eta_min),
            ema              = EMAConfig(use_ema=False, ema_decay=0.999),
            optimizer        = OptimizerConfig(betas=(0.9, 0.999), eps=1e-8),
            gradient_clipper = GradientClipperConfig(clip_mode="fixed", max_grad_norm=1.0),
            io               = IOConfig(logdir=str(self.config.paths.log_base_dir)),
            training         = TrainingConfigInner(device="gpu", epochs=60, validation_frequency=1, gradient_accumulation_steps=1, max_grad_norm=None, verbose=True),

            curriculum = LossCurriculumConfig(
                enabled  = False,
                warmup   = LossConfig(use_param_l1=True, weight_param_l1=1.0, param_weights=(1.0, 1.0, 1.0)),
                complete = LossConfig(use_param_l1=True, weight_param_l1=1.0),
            ),
        )

        return trainer_config, dataset_config

    def _spawn_workers(self, model_name: str, trial_counts: list[int]) -> list[tuple]:
        sname = self._study_name(model_name)
        procs = []
        for gpu_id, n_trials in zip(self.config.gpus, trial_counts):
            if n_trials == 0:
                continue

            log_path = self.run_dir / model_name / f"gpu{gpu_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, str(self.entry_script),
                "--worker",
                "--model",        model_name,
                "--gpu",          str(gpu_id),
                "--n-trials",     str(n_trials),
                "--study-name",   sname,
                "--storage-url",  self.storage_url,
                "--run-tag",      self.tag,
                "--run-dir",      str(self.run_dir),
            ]
            log_fh = open(log_path, "a")
            proc   = subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh)
            procs.append((proc, gpu_id, log_path, log_fh))
            self.logger.info(f"[GPU {gpu_id}] worker — {model_name}  ({n_trials} trials)")

        self.active_procs = list(procs)
        return procs

    def _wait_workers(self, procs: list[tuple], model_name: str) -> bool:
        ok = True
        while procs:
            time.sleep(5)
            still = []
            for proc, gpu_id, log_path, log_fh in procs:
                ret = proc.poll()
                if ret is None:
                    still.append((proc, gpu_id, log_path, log_fh))
                else:
                    log_fh.close()
                    if ret == 0:
                        self.logger.info(f"[GPU {gpu_id}] worker — {model_name}  DONE")
                    else:
                        self.logger.error(f"[GPU {gpu_id}] worker — {model_name}  FAILED (exit {ret}, see {log_path})")
                        ok = False
            procs             = still
            self.active_procs = list(still)
        return ok

    def _terminate_workers(self, signum, frame) -> None:
        for proc, gpu_id, log_path, log_fh in self.active_procs:
            if proc.poll() is None:
                proc.terminate()

        deadline = time.time() + 30
        for proc, gpu_id, log_path, log_fh in self.active_procs:
            try:
                proc.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                proc.kill()

        if self.logger is not None:
            self.logger.warning(f"Received signal {signum} — terminated {len(self.active_procs)} workers, resume with --resume")
            self.logger.close()

        sys.exit(128 + signum)

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGTERM, self._terminate_workers)
        signal.signal(signal.SIGINT,  self._terminate_workers)

    def _load_or_create_study(self, model_name: str) -> optuna.Study:
        return optuna.create_study(
            study_name     = self._study_name(model_name),
            storage        = self.storage_url,
            direction      = "minimize",
            load_if_exists = True,
        )

    def _fail_stale_trials(self, study: optuna.Study) -> int:
        stale = [t for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))]
        for trial in stale:
            study._storage.set_trial_state_values(trial._trial_id, state=TrialState.FAIL)
        return len(stale)

    def _count_done(self, study: optuna.Study) -> int:
        return len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED)))

    def _save_results(self) -> None:
        FileIO.save_json(self.results, self.summary_path, indent=2)

    def _plot_study(self, study: optuna.Study, model_name: str) -> None:
        from pipelines.tuning_pipeline.plots import StudyPlotter

        plot_dir = self.run_dir / model_name / "study_plots"
        plotter  = StudyPlotter(self.logger)
        saved    = plotter.render(study, plot_dir)
        self.logger.info(f"Saved {len(saved)} study plots to {plot_dir}")

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
            procs  = self._spawn_workers(model_name, counts)
            ok     = self._wait_workers(procs, model_name)

        writer  = BestConfigWriter(model_name, self._search_space(model_name), self.run_dir / model_name / "best_config.json")
        payload = writer.write(study)

        if payload is None:
            self.logger.error(f"No completed trials for {model_name}")
            self.results.append({
                "model"           : model_name,
                "status"          : "FAILED",
                "trials_completed": self._count_done(study),
                "val_loss"        : None,
                "best_config"     : None,
            })
            self._save_results()
            return

        self.logger.subsection(f"Best — trial {payload['trial']}  val_loss={payload['val_loss']:.6f}")
        self.logger.kv_table(payload["params"], title="Best Params")

        if tune_cfg.emit_study_plots:
            self._plot_study(study, model_name)

        self.results.append({
            "model"           : model_name,
            "status"          : "DONE" if ok else "PARTIAL",
            "trials_completed": self._count_done(study),
            "val_loss"        : payload["val_loss"],
            "best_config"     : str(writer.path),
        })
        self._save_results()

    def schedule(self, target_model: str | None = None, resume: bool = False) -> None:
        from models import CONFIG_REGISTRY
        from tools.config_cli import ConfigCli
        from tools.logger import Logger

        if resume and not self.db_path.exists():
            sys.exit(f"ERROR: --resume given but no study found at {self.db_path}")
        if not resume and self.db_path.exists():
            sys.exit(f"ERROR: run tag '{self.tag}' already has a study at {self.db_path} — pass --resume to continue it")

        self._install_signal_handlers()

        tune_cfg = self.config.tuning
        self.run_dir.mkdir(parents=True, exist_ok=True)
        ConfigCli.save_resolved(self.config, self.run_dir / "resolved_config.json")

        self.logger = Logger(log_dir=str(self.run_dir), name="tune_scheduler")

        all_models = list(CONFIG_REGISTRY.keys())
        if target_model is not None:
            if target_model not in CONFIG_REGISTRY:
                self.logger.close()
                sys.exit(f"ERROR: unknown model '{target_model}'. Available: {list(CONFIG_REGISTRY.keys())}")
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

        optuna.logging.set_verbosity(optuna.logging.WARNING)

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
        from models import CONFIG_REGISTRY
        from tools.logger import Logger

        optuna.logging.set_verbosity(optuna.logging.WARNING)

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
            seed             = 42 + gpu_id,
        )

        study = optuna.load_study(study_name=study_name, storage=storage_url, sampler=sampler, pruner=pruner)
        tuner.run(study, n_trials)

        logger.info(f"[GPU {gpu_id}] — {model_name}  DONE")
        logger.close()
