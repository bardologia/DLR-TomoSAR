from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from configuration.benchmark_config import BenchmarkConfig
from pipelines.benchmark_pipeline.gpu_queue import GpuJob, GpuQueue
from tools.logger import Logger


class TrainingStage:
    def __init__(self, config: BenchmarkConfig, entry_script: Path, run_tag: str, models: list[str], logger: Logger) -> None:
        self.config       = config
        self.entry_script = entry_script
        self.run_tag      = run_tag
        self.models       = models
        self.logger       = logger

        self.run_dir      = Path(config.paths.log_base_dir) / run_tag
        self.stage_dir    = self.run_dir / "training"
        self.results_path = self.run_dir / "pipeline" / "training_results.json"

    def run(self) -> list[dict]:
        self.logger.section("Training queue")
        self.logger.kv_table({
            "Models"     : len(self.models),
            "Epochs"     : self.config.training.epochs,
            "Batch size" : self.config.training.batch_size,
            "GPUs"       : self.config.gpus,
            "Stage dir"  : str(self.stage_dir),
        }, title="Configuration")

        cached  = [m for m in self.models if self._has_checkpoint(m)]
        pending = [m for m in self.models if m not in cached]

        for model_name in cached:
            self.logger.info(f"{model_name}: existing checkpoint reused")

        ran = []
        if pending:
            queue = GpuQueue(gpus=self.config.gpus, logger=self.logger, poll_interval_s=self.config.poll_interval_s)
            ran   = [asdict(r) for r in queue.run([self._job(m) for m in pending])]

        results  = ran + [self._cached_result(m) for m in cached]
        by_model = {r["name"]: r for r in results}
        results  = [by_model[m] for m in self.models if m in by_model]

        self._write_results(results)
        self._log_summary(results)

        return results

    def _job(self, model_name: str) -> GpuJob:
        return GpuJob(
            name     = model_name,
            command  = [sys.executable, str(self.entry_script), "--worker", "train", "--model", model_name, "--run-tag", self.run_tag, "--run-dir", str(self.run_dir)],
            log_path = self.stage_dir / model_name / "worker.log",
        )

    def _has_checkpoint(self, model_name: str) -> bool:
        if not self.config.resume:
            return False

        model_dir = self.stage_dir / model_name
        if not model_dir.is_dir():
            return False

        return next(model_dir.rglob(self.config.inference.checkpoint_name), None) is not None

    def _cached_result(self, model_name: str) -> dict:
        return {
            "name"       : model_name,
            "gpu"        : None,
            "status"     : "DONE",
            "returncode" : 0,
            "duration_s" : None,
            "log_file"   : str(self.stage_dir / model_name / "worker.log"),
        }

    def _write_results(self, results: list[dict]) -> None:
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    def _log_summary(self, results: list[dict]) -> None:
        done   = [r for r in results if r["status"] == "DONE"]
        failed = [r for r in results if r["status"] != "DONE"]

        self.logger.subsection("Training summary")
        self.logger.kv_table({
            "Total"  : len(results),
            "Done"   : len(done),
            "Failed" : len(failed),
        }, title=f"{len(done)}/{len(results)} finished")

        for r in failed:
            self.logger.error(f"FAILED  {r['name']}  (see {r['log_file']})")
