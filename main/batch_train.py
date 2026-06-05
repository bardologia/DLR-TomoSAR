from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from configuration.single_train_config import BatchTrainConfig
from pipelines.benchmark_pipeline.gpu_queue import GpuJob, GpuQueue
from tools.config_cli import ConfigCli
from tools.logger import Logger

_LOSS_SWEEP = [
    ("mse",   "curriculum.complete.use_mse_curve",          "curriculum.complete.weight_mse_curve"),
    ("l1",    "curriculum.complete.use_l1_curve",           "curriculum.complete.weight_l1_curve"),
    ("huber", "curriculum.complete.use_huber_curve",        "curriculum.complete.weight_huber_curve"),
    ("charb", "curriculum.complete.use_charbonnier_curve",  "curriculum.complete.weight_charbonnier_curve"),
    ("cos",   "curriculum.complete.use_cosine_curve",       "curriculum.complete.weight_cosine_curve"),
    ("spec",  "curriculum.complete.use_spectral_coherence", "curriculum.complete.weight_spectral_coh"),
    ("ssim",  "curriculum.complete.use_ssim_curve",         "curriculum.complete.weight_ssim_curve"),
]

_SWEEP_WEIGHTS = [0.01, 0.05, 0.02]


class BatchTrainScheduler:
    def __init__(self, config: BatchTrainConfig, base_overrides: dict, entry_script: Path) -> None:
        self.config         = config
        self.base_overrides = base_overrides
        self.entry_script   = entry_script
        self.log_dir        = Path(config.base.logdir) / "batch_train_logs"

        self.logger = Logger(log_dir=str(self.log_dir), name="batch_train")

    def experiments(self) -> list[tuple[str, dict]]:
        model = self.config.base.model_name

        experiments = []
        for label, use_key, weight_key in _LOSS_SWEEP:
            for weight in _SWEEP_WEIGHTS:
                run_name  = f"{model}_w-pL11_c-pL11-{label}{weight:g}"
                overrides = {
                    "curriculum.enabled"                  : True,
                    "curriculum.swap_epoch"               : 30,
                    "curriculum.reset_early_stopping"     : True,
                    "curriculum.reset_lr"                 : True,
                    "curriculum.reset_warmup"             : True,
                    "curriculum.reset_optimizer"          : False,
                    "curriculum.warmup.use_param_l1"      : True,
                    "curriculum.warmup.weight_param_l1"   : 1.0,
                    "curriculum.complete.use_param_l1"    : True,
                    "curriculum.complete.weight_param_l1" : 1.0,
                    use_key                               : True,
                    weight_key                            : weight,
                }
                experiments.append((run_name, overrides))

        return experiments

    def run(self) -> None:
        experiments = self.experiments()

        self.logger.section("Batch train")
        self.logger.kv_table({
            "Model"          : self.config.base.model_name,
            "Experiments"    : len(experiments),
            "GPUs"           : self.config.gpus,
            "Infer after"    : self.config.base.infer_after,
            "Base overrides" : self.base_overrides or "—",
            "Log dir"        : str(self.log_dir),
        }, title="Configuration")

        queue   = GpuQueue(gpus=self.config.gpus, logger=self.logger, poll_interval_s=self.config.poll_interval_s)
        results = queue.run([self._job(run_name, overrides) for run_name, overrides in experiments])

        self.logger.section("Summary")
        rows = [{"Experiment": r.name, "Status": r.status, "Duration": f"{r.duration_s / 60:.1f} min"} for r in results]
        self.logger.metrics_table(rows, columns=["Experiment", "Status", "Duration"])

        self.logger.close()

    def _job(self, run_name: str, overrides: dict) -> GpuJob:
        argv = ConfigCli.to_argv({**self.base_overrides, **overrides, "run_name": run_name})

        return GpuJob(
            name     = run_name,
            command  = [sys.executable, str(self.entry_script)] + argv,
            log_path = self.log_dir / f"{run_name}.log",
        )


def main() -> None:
    cli    = ConfigCli(BatchTrainConfig(), description="Batch training sweep")
    config = cli.apply()

    base_overrides = {path.removeprefix("base."): value for path, value in cli.overrides.items() if path.startswith("base.")}

    scheduler = BatchTrainScheduler(
        config         = config,
        base_overrides = base_overrides,
        entry_script   = Path(__file__).resolve().parent / "single_train.py",
    )
    scheduler.run()


if __name__ == "__main__":
    main()
