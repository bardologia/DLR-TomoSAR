from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from configuration.cross_validation_config import CrossValidationConfig
from pipelines.benchmark_pipeline.gpu_queue import GpuJob, GpuQueue
from tools.logger import Logger


class FoldTrainingStage:
    def __init__(self, config: CrossValidationConfig, entry_script: Path, run_tag: str, logger: Logger) -> None:
        self.config       = config
        self.entry_script = entry_script
        self.run_tag      = run_tag
        self.logger       = logger

        self.run_dir      = Path(config.paths.log_base_dir) / run_tag
        self.folds_dir    = self.run_dir / "folds"
        self.results_path = self.run_dir / "pipeline" / "training_results.json"

        self.fold_names   = [f"fold_{index}" for index in range(config.folds.n_folds)]

    def run(self) -> list[dict]:
        self.logger.section("Fold training queue")
        self.logger.kv_table({
            "Model"      : self.config.model_name,
            "Folds"      : self.config.folds.n_folds,
            "Epochs"     : self.config.training.epochs,
            "Batch size" : self.config.training.batch_size,
            "GPUs"       : self.config.gpus,
            "Folds dir"  : str(self.folds_dir),
        }, title="Configuration")

        cached  = [name for name in self.fold_names if self._has_checkpoint(name)]
        pending = [name for name in self.fold_names if name not in cached]

        for fold_name in cached:
            self.logger.info(f"{fold_name}: existing checkpoint reused")

        ran = []
        if pending:
            queue = GpuQueue(gpus=self.config.gpus, logger=self.logger, poll_interval_s=self.config.poll_interval_s)
            ran   = [asdict(r) for r in queue.run([self._job(name) for name in pending])]

        results  = ran + [self._cached_result(name) for name in cached]
        by_name  = {r["name"]: r for r in results}
        results  = [by_name[name] for name in self.fold_names if name in by_name]

        self._write_results(results)
        self._log_summary(results)

        return results

    def _job(self, fold_name: str) -> GpuJob:
        fold_index = fold_name.split("_")[-1]

        return GpuJob(
            name     = fold_name,
            command  = [sys.executable, str(self.entry_script), "--worker", "train", "--fold", fold_index, "--run-tag", self.run_tag, "--run-dir", str(self.run_dir)],
            log_path = self.folds_dir / fold_name / "worker.log",
        )

    def _has_checkpoint(self, fold_name: str) -> bool:
        if not self.config.resume:
            return False

        fold_dir = self.folds_dir / fold_name
        if not fold_dir.is_dir():
            return False

        return next(fold_dir.rglob(self.config.inference.checkpoint_name), None) is not None

    def _cached_result(self, fold_name: str) -> dict:
        return {
            "name"       : fold_name,
            "gpu"        : None,
            "status"     : "DONE",
            "returncode" : 0,
            "duration_s" : None,
            "log_file"   : str(self.folds_dir / fold_name / "worker.log"),
        }

    def _write_results(self, results: list[dict]) -> None:
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    def _log_summary(self, results: list[dict]) -> None:
        done   = [r for r in results if r["status"] == "DONE"]
        failed = [r for r in results if r["status"] != "DONE"]

        self.logger.subsection("Fold training summary")
        self.logger.kv_table({
            "Total"  : len(results),
            "Done"   : len(done),
            "Failed" : len(failed),
        }, title=f"{len(done)}/{len(results)} finished")

        for r in failed:
            self.logger.error(f"FAILED  {r['name']}  (see {r['log_file']})")
