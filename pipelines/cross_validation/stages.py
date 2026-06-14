from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from configuration.experiments.cross_validation_config         import CrossValidationConfig
from pipelines.cross_validation.cv_report import CrossValidationReport
from pipelines.cross_validation.folds     import FoldNaming, FoldPlanner
from pipelines.cross_validation.workers   import FoldCollector
from tools                              import ExperimentStage, GpuJob, QueuedInferenceStage, QueuedTrainingStage
from tools.monitoring.logger                                  import Logger


class FoldTrainingStage(QueuedTrainingStage):
    stage_subdir = "folds"

    def __init__(self, config: CrossValidationConfig, entry_script: Path, run_tag: str, logger: Logger) -> None:
        fold_names = [FoldNaming.name(index) for index in range(config.folds.n_folds)]
        super().__init__(config=config, entry_script=entry_script, run_tag=run_tag, items=fold_names, logger=logger)

    def _config_kv(self) -> dict:
        return {
            "Model"      : self.config.model_name,
            "Folds"      : self.config.folds.n_folds,
            "Epochs"     : self.config.training.epochs,
            "Batch size" : self.config.training.batch_size,
            "GPUs"       : self.config.gpus,
            "Folds dir"  : str(self.stage_dir),
        }

    def _worker_flag(self) -> str:
        return "--fold"

    def _worker_value(self, item: str) -> str:
        return str(FoldNaming.index(item))


class FoldInferenceStage(QueuedInferenceStage):
    stage_subdir = "folds"

    def __init__(self, config: CrossValidationConfig, entry_script: Path, run_tag: str, planner: FoldPlanner, logger: Logger) -> None:
        fold_names = [FoldNaming.name(index) for index in range(config.folds.n_folds)]
        super().__init__(config=config, entry_script=entry_script, run_tag=run_tag, items=fold_names, logger=logger)
        self.planner = planner
        self.splits  = config.inference_splits

    def _split_log(self, fold_name: str, split: str) -> Path:
        return self.stage_dir / fold_name / f"inference_{split}_worker.log"

    def _split_result(self, fold_name: str, split: str, status: str) -> dict:
        return {
            "name"       : f"{fold_name}:{split}",
            "gpu"        : None,
            "status"     : status,
            "returncode" : 0 if status == "DONE" else None,
            "duration_s" : None,
            "log_file"   : str(self._split_log(fold_name, split)),
        }

    def _has_split_inference(self, fold_name: str, split: str) -> bool:
        if not self.config.resume:
            return False

        return (self.stage_dir / fold_name / "inference" / split / "metrics.json").exists()

    def _split_job(self, fold_index: int, split: str) -> GpuJob:
        fold_name = FoldNaming.name(fold_index)

        return GpuJob(
            name     = f"{fold_name}:{split}",
            command  = [sys.executable, str(self.entry_script), "--worker", "infer", "--fold", str(fold_index), "--split", split, "--run-tag", self.run_tag, "--run-dir", str(self.run_dir)],
            log_path = self._split_log(fold_name, split),
        )

    def run(self) -> list[dict]:
        self.logger.section("Fold inference queue")
        self.logger.kv_table({
            "Folds"  : self.config.folds.n_folds,
            "Splits" : ", ".join(self.splits),
            "GPUs"   : self.config.gpus,
        }, title="Configuration")

        jobs    = []
        results = []

        for fold_name in self.items:
            fold_index = FoldNaming.index(fold_name)
            plan       = self.planner.plan(fold_index)

            if not self._has_checkpoint(fold_name):
                self.logger.warning(f"{fold_name}: no checkpoint, inference skipped")
                for split in self.splits:
                    results.append(self._split_result(fold_name, split, "SKIPPED"))
                continue

            for split in self.splits:
                if len(plan.split_regions.regions(split)) != 1:
                    self.logger.warning(f"{fold_name}:{split}: split is disjoint, inference skipped (stitching requires one contiguous region)")
                    results.append(self._split_result(fold_name, split, "SKIPPED"))
                    continue

                if self._has_split_inference(fold_name, split):
                    self.logger.info(f"{fold_name}:{split}: existing inference reused")
                    results.append(self._split_result(fold_name, split, "DONE"))
                    continue

                jobs.append(self._split_job(fold_index, split))

        if jobs:
            results.extend(self._run_queue(jobs))

        self._write_results(results, self.results_path)
        self._log_summary(results)

        return results


class CrossValidationReportStage(ExperimentStage):
    def __init__(self, config: CrossValidationConfig, run_tag: str, planner: FoldPlanner, logger: Logger) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger)
        self.planner = planner

    def run(self) -> Path:
        self.logger.section("Cross-validation reports")

        splits     = self.config.inference_splits if self.config.runs_inference() else []
        model_name = self.config.model_name if self.config.training_type != "autoencoder" else "profile_ae"

        collector = FoldCollector(run_dir=self.run_dir, splits=splits, logger=self.logger)
        base_records, records_by_split = collector.collect_by_split()

        out_dir = self.run_dir / "reports" / datetime.now().strftime("%Y%m%d_%H%M%S")

        report = CrossValidationReport(
            base_records     = base_records,
            records_by_split = records_by_split,
            planner          = self.planner,
            out_dir          = out_dir,
            model_name       = model_name,
            embed_images     = self.config.comparison.embed_images,
            logger           = self.logger,
        )

        written = report.write_all()

        self.logger.subsection("Reports written")
        for path in written:
            self.logger.info(f"{path}")

        return out_dir
