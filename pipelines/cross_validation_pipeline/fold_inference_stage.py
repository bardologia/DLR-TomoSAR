from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from configuration.cross_validation_config import CrossValidationConfig
from pipelines.benchmark_pipeline.gpu_queue import GpuJob, GpuQueue
from pipelines.cross_validation_pipeline.fold_planner import FoldPlanner
from tools.logger import Logger


class FoldInferenceStage:
    def __init__(self, config: CrossValidationConfig, entry_script: Path, run_tag: str, planner: FoldPlanner, logger: Logger) -> None:
        self.config       = config
        self.entry_script = entry_script
        self.run_tag      = run_tag
        self.planner      = planner
        self.logger       = logger

        self.run_dir      = Path(config.paths.log_base_dir) / run_tag
        self.folds_dir    = self.run_dir / "folds"
        self.results_path = self.run_dir / "pipeline" / "inference_results.json"

    def run(self) -> list[dict]:
        self.logger.section("Fold inference queue")
        self.logger.kv_table({
            "Folds"  : self.config.folds.n_folds,
            "Splits" : ", ".join(self.config.inference_splits),
            "EMA"    : self.config.inference.use_ema,
            "GPUs"   : self.config.gpus,
        }, title="Configuration")

        jobs    = []
        results = []

        for fold_index in range(self.config.folds.n_folds):
            fold_name = f"fold_{fold_index}"
            plan      = self.planner.plan(fold_index)

            if not self._has_checkpoint(fold_name):
                self.logger.warning(f"{fold_name}: no checkpoint, inference skipped")
                for split in self.config.inference_splits:
                    results.append(self._static_result(fold_name, split, "SKIPPED"))
                continue

            for split in self.config.inference_splits:
                if len(plan.split_regions.regions(split)) != 1:
                    self.logger.warning(f"{fold_name}:{split}: split is disjoint, inference skipped (stitching requires one contiguous region)")
                    results.append(self._static_result(fold_name, split, "SKIPPED"))
                    continue

                if self._has_inference(fold_name, split):
                    self.logger.info(f"{fold_name}:{split}: existing inference reused")
                    results.append(self._static_result(fold_name, split, "DONE"))
                    continue

                jobs.append(self._job(fold_index, split))

        if jobs:
            queue = GpuQueue(gpus=self.config.gpus, logger=self.logger, poll_interval_s=self.config.poll_interval_s)
            results.extend(asdict(r) for r in queue.run(jobs))

        self._write_results(results)
        self._log_summary(results)

        return results

    def _job(self, fold_index: int, split: str) -> GpuJob:
        fold_name = f"fold_{fold_index}"

        return GpuJob(
            name     = f"{fold_name}:{split}",
            command  = [sys.executable, str(self.entry_script), "--worker", "infer", "--fold", str(fold_index), "--split", split, "--run-tag", self.run_tag, "--run-dir", str(self.run_dir)],
            log_path = self.folds_dir / fold_name / f"inference_{split}_worker.log",
        )

    def _has_checkpoint(self, fold_name: str) -> bool:
        fold_dir = self.folds_dir / fold_name
        if not fold_dir.is_dir():
            return False

        return next(fold_dir.rglob(self.config.inference.checkpoint_name), None) is not None

    def _has_inference(self, fold_name: str, split: str) -> bool:
        if not self.config.resume:
            return False

        return (self.folds_dir / fold_name / "inference" / split / "metrics.json").exists()

    def _static_result(self, fold_name: str, split: str, status: str) -> dict:
        return {
            "name"       : f"{fold_name}:{split}",
            "gpu"        : None,
            "status"     : status,
            "returncode" : 0 if status == "DONE" else None,
            "duration_s" : None,
            "log_file"   : str(self.folds_dir / fold_name / f"inference_{split}_worker.log"),
        }

    def _write_results(self, results: list[dict]) -> None:
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    def _log_summary(self, results: list[dict]) -> None:
        done   = [r for r in results if r["status"] == "DONE"]
        failed = [r for r in results if r["status"] == "FAILED"]

        self.logger.subsection("Fold inference summary")
        self.logger.kv_table({
            "Total"   : len(results),
            "Done"    : len(done),
            "Failed"  : len(failed),
            "Skipped" : len(results) - len(done) - len(failed),
        }, title=f"{len(done)}/{len(results)} finished")

        for r in failed:
            self.logger.error(f"FAILED  {r['name']}  (see {r['log_file']})")
