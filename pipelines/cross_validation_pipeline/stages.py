from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from configuration.cross_validation_config         import CrossValidationConfig
from pipelines.cross_validation_pipeline.cv_report import CrossValidationReport
from pipelines.cross_validation_pipeline.folds     import FoldCollector, FoldPlanner
from pipelines.shared.orchestration                import ExperimentStage, GpuJob
from tools.logger                                  import Logger


class FoldTrainingStage(ExperimentStage):
    def __init__(self, config: CrossValidationConfig, entry_script: Path, run_tag: str, logger: Logger) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger, entry_script=entry_script)
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
            ran = self._run_queue([self._job(name) for name in pending])

        results = self._order_results(ran + [self._cached_result(name) for name in cached], self.fold_names)

        self._write_results(results, self.results_path)
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

    def _log_summary(self, results: list[dict]) -> None:
        done   = [r for r in results if r["status"] == "DONE"]
        failed = [r for r in results if r["status"] != "DONE"]

        self.logger.subsection("Fold training summary")
        self.logger.kv_table({
            "Total"  : len(results),
            "Done"   : len(done),
            "Failed" : len(failed),
        }, title=f"{len(done)}/{len(results)} finished")

        self._log_failures(failed)


class FoldInferenceStage(ExperimentStage):
    def __init__(self, config: CrossValidationConfig, entry_script: Path, run_tag: str, planner: FoldPlanner, logger: Logger) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger, entry_script=entry_script)
        self.planner      = planner
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
            results.extend(self._run_queue(jobs))

        self._write_results(results, self.results_path)
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

        self._log_failures(failed)


class CrossValidationReportStage(ExperimentStage):
    def __init__(self, config: CrossValidationConfig, run_tag: str, planner: FoldPlanner, logger: Logger) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger)
        self.planner = planner

    def run(self) -> Path:
        self.logger.section("Cross-validation reports")

        collector = FoldCollector(run_dir=self.run_dir, splits=self.config.inference_splits, logger=self.logger)
        base_records, records_by_split = collector.collect_by_split()

        out_dir = self.run_dir / "reports" / datetime.now().strftime("%Y%m%d_%H%M%S")

        report = CrossValidationReport(
            base_records     = base_records,
            records_by_split = records_by_split,
            planner          = self.planner,
            out_dir          = out_dir,
            model_name       = self.config.model_name,
            embed_images     = self.config.comparison.embed_images,
            logger           = self.logger,
        )

        written = report.write_all()

        self.logger.subsection("Reports written")
        for path in written:
            self.logger.info(f"{path}")

        return out_dir
