from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import optuna

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

class TuningOrchestrator:
    def __init__(self, tag: str, config) -> None:
        self.tag          = tag
        self.config       = config
        self.run_dir      = Path(config.paths.log_base_dir) / tag
        self.storage_url  = f"sqlite:///{self.run_dir / 'optuna.db'}"
        self.summary_path = self.run_dir / "tuning_results.json"

        self.logger  = None
        self.results = []

    def _distribute_trials(self, total: int, n_workers: int) -> list[int]:
        base  = total // n_workers
        extra = total % n_workers
        return [base + (1 if i < extra else 0) for i in range(n_workers)]

    def _study_name(self, model_name: str, phase: int) -> str:
        return f"{model_name}_phase{phase}_{self.tag}"

    def _build_base_configs(self):
        import json
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

    def _spawn_workers(self, model_name: str, phase: int, trial_counts: list[int]) -> list[tuple]:
        sname = self._study_name(model_name, phase)
        procs = []
        for gpu_id, n_trials in zip(self.config.gpus, trial_counts):
            log_path = self.run_dir / model_name / f"phase{phase}_gpu{gpu_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, __file__,
                "--worker",
                "--model",        model_name,
                "--gpu",          str(gpu_id),
                "--phase",        str(phase),
                "--n-trials",     str(n_trials),
                "--study-name",   sname,
                "--storage-url",  self.storage_url,
                "--run-tag",      self.tag,
                "--run-dir",      str(self.run_dir),
            ]
            log_fh = open(log_path, "w")
            proc   = subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh)
            procs.append((proc, gpu_id, log_path, log_fh))
            self.logger.info(f"[GPU {gpu_id}] phase-{phase} worker — {model_name}  ({n_trials} trials)")
        return procs

    def _wait_workers(self, procs: list[tuple], model_name: str, phase: int) -> bool:
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
                        self.logger.info(f"[GPU {gpu_id}] phase-{phase} worker — {model_name}  DONE")
                    else:
                        self.logger.error(f"[GPU {gpu_id}] phase-{phase} worker — {model_name}  FAILED (exit {ret}, see {log_path})")
                        ok = False
            procs = still
        return ok

    def _create_study(self, model_name: str, phase: int, phase_cfg) -> None:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials = phase_cfg.pruner_n_startup_trials,
            n_warmup_steps   = phase_cfg.pruner_n_warmup_steps,
        )

        sampler = optuna.samplers.TPESampler(
            n_startup_trials = phase_cfg.pruner_n_startup_trials,
            multivariate     = True,
            constant_liar    = True,
            seed             = 42,
        )

        optuna.create_study(
            study_name     = self._study_name(model_name, phase),
            storage        = self.storage_url,
            direction      = "minimize",
            sampler        = sampler,
            pruner         = pruner,
            load_if_exists = True,
        )

    def _decode_phase1_best(self, model_name: str, best_p1_trial) -> dict:
        from models import CONFIG_REGISTRY

        lr_space   = CONFIG_REGISTRY[model_name].tunable_lr_params()
        raw_p1     = dict(best_p1_trial.params)
        decoded_p1 = {}
        for k, v in raw_p1.items():
            if k.endswith("__idx"):
                param_name = k[:-5]
                spec       = lr_space.get(param_name, {})
                if spec.get("type") == "indexed_categorical":
                    decoded_p1[param_name] = spec["choices"][v]
            else:
                decoded_p1[k] = v
        return decoded_p1

    def _save_results(self) -> None:
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

    def _tune_model(self, model_name: str, tune_cfg) -> None:
        self.logger.section(f"[{model_name}]")

        p1_counts = self._distribute_trials(tune_cfg.phase1.n_trials, len(gpus))
        self._create_study(model_name, 1, tune_cfg.phase1)
        self.logger.subsection(f"Phase 1 — {sum(p1_counts)} trials across {len(gpus)} GPUs")

        p1_procs = self._spawn_workers(model_name, 1, p1_counts)
        p1_ok    = self._wait_workers(p1_procs, model_name, 1)

        p1_study      = optuna.load_study(study_name=self._study_name(model_name, 1), storage=self.storage_url)
        best_p1_trial = p1_study.best_trial
        best_p1_path  = self.run_dir / model_name / "phase1_best.json"
        best_p1_path.parent.mkdir(parents=True, exist_ok=True)

        decoded_p1 = self._decode_phase1_best(model_name, best_p1_trial)

        with open(best_p1_path, "w", encoding="utf-8") as f:
            json.dump(decoded_p1, f, indent=2)

        self.logger.subsection(f"Phase 1 best — trial {best_p1_trial.number}  val_loss={best_p1_trial.value:.6f}")
        self.logger.kv_table(decoded_p1, title="Best Phase-1 Params")

        if not p1_ok:
            self.logger.error(f"Phase 1 had failures for {model_name} — skipping Phase 2")
            self.results.append({
                "model"          : model_name,
                "status"         : "PARTIAL",
                "phase1_val_loss": best_p1_trial.value,
                "phase2_val_loss": None,
                "best_config"    : None,
            })
            self._save_results()
            return

        p2_counts = self._distribute_trials(tune_cfg.phase2.n_trials, len(gpus))
        self._create_study(model_name, 2, tune_cfg.phase2)
        self.logger.subsection(f"Phase 2 — {sum(p2_counts)} trials across {len(gpus)} GPUs")

        p2_procs = self._spawn_workers(model_name, 2, p2_counts)
        p2_ok    = self._wait_workers(p2_procs, model_name, 2)

        p2_study      = optuna.load_study(study_name=self._study_name(model_name, 2), storage=self.storage_url)
        best_p2_trial = p2_study.best_trial

        self.logger.subsection(f"Phase 2 best — trial {best_p2_trial.number}  val_loss={best_p2_trial.value:.6f}")
        self.logger.kv_table(best_p2_trial.params, title="Best Phase-2 Params")

        merged_params = {**best_p1_trial.params, **best_p2_trial.params}
        best_cfg_path = self.run_dir / model_name / "best_config.json"
        with open(best_cfg_path, "w", encoding="utf-8") as f:
            json.dump({
                "model"          : model_name,
                "phase1_val_loss": best_p1_trial.value,
                "phase2_val_loss": best_p2_trial.value,
                "params"         : merged_params,
            }, f, indent=2)

        self.results.append({
            "model"          : model_name,
            "status"         : "DONE" if (p1_ok and p2_ok) else "PARTIAL",
            "phase1_val_loss": best_p1_trial.value,
            "phase2_val_loss": best_p2_trial.value,
            "best_config"    : str(best_cfg_path),
        })
        self._save_results()

    def schedule(self, target_model: str | None = None) -> None:
        from models import CONFIG_REGISTRY
        from tools.config_cli import ConfigCli
        from tools.logger import Logger

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
            "Run tag"         : self.tag,
            "Models"          : len(models_to_run),
            "GPUs"            : self.config.gpus,
            "Phase-1 trials"  : tune_cfg.phase1.n_trials,
            "Phase-2 trials"  : tune_cfg.phase2.n_trials,
            "Storage"         : self.storage_url,
            "Log dir"         : str(self.run_dir),
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
        phase      : int,
        n_trials   : int,
        study_name : str,
        storage_url: str,
    ) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"]    = str(gpu_id)
        os.environ["MKL_NUM_THREADS"]         = "4"
        os.environ["NUMEXPR_NUM_THREADS"]     = "4"
        os.environ["OMP_NUM_THREADS"]         = "4"

        import optuna
        from models import CONFIG_REGISTRY
        from tools.logger import Logger
        from pipelines.tuning_pipeline.pipeline import TuningPipeline

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        model_log_dir = str(self.run_dir / model_name / f"phase{phase}_gpu{gpu_id}")
        logger        = Logger(log_dir=model_log_dir, name=f"tune_worker_p{phase}_gpu{gpu_id}_{model_name}")

        logger.section(f"[GPU {gpu_id}] Phase {phase} Worker — {model_name}")
        logger.kv_table({
            "GPU"        : gpu_id,
            "Phase"      : phase,
            "Trials"     : n_trials,
            "Study"      : study_name,
            "Storage"    : storage_url,
        })

        tune_cfg                 = self.config.tuning
        trainer_cfg, dataset_cfg = self._build_base_configs()
        model_config_cls         = CONFIG_REGISTRY[model_name]

        pipeline = TuningPipeline(
            model_name          = model_name,
            model_config_cls    = model_config_cls,
            base_trainer_config = trainer_cfg,
            base_dataset_config = dataset_cfg,
            tune_cfg            = tune_cfg,
            log_dir             = str(self.run_dir / model_name),
            logger              = logger,
        )

        study = optuna.load_study(study_name=study_name, storage=storage_url)

        if phase == 1:
            pipeline.run_phase1(study, n_trials)

        elif phase == 2:
            p1_best_path = self.run_dir / model_name / "phase1_best.json"
            if not p1_best_path.exists():
                logger.error(f"Phase-1 best params not found at {p1_best_path}")
                sys.exit(1)
            with open(p1_best_path, "r", encoding="utf-8") as f:
                best_p1_params = json.load(f)
            pipeline.run_phase2(study, n_trials, best_p1_params)

        else:
            logger.error(f"Unknown phase: {phase}")
            sys.exit(1)

        logger.info(f"[GPU {gpu_id}] Phase {phase} — {model_name}  DONE")
        logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker",      action="store_true")
    parser.add_argument("--model",       type=str,  default=None, help="(scheduler) tune only this model; (worker) model being tuned")
    parser.add_argument("--gpu",         type=int,  default=0)
    parser.add_argument("--phase",       type=int,  default=1)
    parser.add_argument("--n-trials",    type=int,  default=8)
    parser.add_argument("--study-name",  type=str,  default=None)
    parser.add_argument("--storage-url", type=str,  default=None)
    parser.add_argument("--run-tag",     type=str,  default=None)
    parser.add_argument("--run-dir",     type=str,  default=None)
    args, _ = parser.parse_known_args()

    from configuration.tuning_config import TuningEntryConfig
    from tools.config_cli import ConfigCli

    if args.worker:
        if args.model is None:
            sys.exit("ERROR: --worker requires --model")
        if args.study_name is None or args.storage_url is None:
            sys.exit("ERROR: --worker requires --study-name and --storage-url")

        tag    = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        config = TuningEntryConfig()
        if args.run_dir:
            config = ConfigCli.load_resolved(config, Path(args.run_dir) / "resolved_config.json")

        orchestrator = TuningOrchestrator(tag=tag, config=config)
        orchestrator.run_worker(
            model_name  = args.model,
            gpu_id      = args.gpu,
            phase       = args.phase,
            n_trials    = args.n_trials,
            study_name  = args.study_name,
            storage_url = args.storage_url,
        )

    else:
        config = ConfigCli(TuningEntryConfig(), description="Two-phase hyperparameter tuning").apply()
        tag    = args.run_tag or config.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")

        orchestrator = TuningOrchestrator(tag=tag, config=config)
        orchestrator.schedule(target_model=args.model)


if __name__ == "__main__":
    main()
