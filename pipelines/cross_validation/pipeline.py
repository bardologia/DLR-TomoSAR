from __future__ import annotations

from datetime import datetime
from pathlib  import Path

from configuration.experiments.cross_validation_config import CrossValidationConfig
from tools.runtime.config_cli                          import ConfigCli
from pipelines.cross_validation.folds                  import FoldConfigFactory, FoldPlanner
from pipelines.cross_validation.stages                 import CrossValidationReportStage, FoldInferenceStage, FoldTrainingStage
from tools.data.io                                     import FileIO
from tools.monitoring.logger                           import Logger


class CrossValidationPipeline:
    def __init__(self, config: CrossValidationConfig, entry_script: Path) -> None:
        self.config       = config
        self.entry_script = entry_script
        self.run_tag      = config.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_dir      = Path(config.paths.log_base_dir) / self.run_tag
        self.pipeline_dir = self.run_dir / "pipeline"
        self.state_path   = self.pipeline_dir / "state.json"

        FileIO.ensure_dir(self.pipeline_dir)
        ConfigCli.save_resolved(config, self.pipeline_dir / "resolved_config.json")

        self.logger  = Logger(log_dir=str(self.pipeline_dir), name="cross_validation_pipeline")
        self.factory = FoldConfigFactory(config)
        self.state   = {"run_tag": self.run_tag, "stages": {}}

    def _run_training(self) -> list[dict]:
        stage   = FoldTrainingStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, logger=self.logger)
        results = stage.run()

        failed = [r for r in results if r["status"] != "DONE"]
        self._mark_stage("training", "completed" if not failed else "partial")
        return results

    def _run_inference(self, planner: FoldPlanner) -> list[dict]:
        if not self.config.runs_inference():
            self.logger.subsection(f"Inference skipped for training_type '{self.config.training_type}'; folds evaluated by reconstruction loss")
            self._mark_stage("inference", "skipped")
            return []

        stage   = FoldInferenceStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, planner=planner, logger=self.logger)
        results = stage.run()

        failed = [r for r in results if r["status"] == "FAILED"]
        self._mark_stage("inference", "completed" if not failed else "partial")
        return results

    def _run_reports(self, planner: FoldPlanner) -> Path:
        stage   = CrossValidationReportStage(config=self.config, run_tag=self.run_tag, planner=planner, logger=self.logger)
        out_dir = stage.run()

        self._mark_stage("reports", "completed")
        return out_dir

    def _mark_stage(self, stage_name: str, status: str) -> None:
        self.state["stages"][stage_name] = {
            "status"    : status,
            "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        FileIO.save_json(self.state, self.state_path, indent=2)

    def run(self) -> None:
        planner = self.factory.planner()

        self.logger.section("Cross-validation pipeline")
        self.logger.kv_table({
            "Run tag"          : self.run_tag,
            "Model"            : self.config.model_name,
            "Folds"            : self.config.folds.n_folds,
            "Azimuth extent"   : f"[{self.config.folds.azimuth_start}, {self.config.folds.azimuth_end})",
            "Inference splits" : ", ".join(self.config.inference_splits),
            "GPUs"             : self.config.gpus,
            "Resume"           : self.config.resume,
            "Run dir"          : str(self.run_dir),
        }, title="Configuration")

        self.logger.subsection("Fold plan")
        rows = []
        for plan in planner.plans():
            train_regions = plan.split_regions.regions("train")
            rows.append({
                "Fold"  : plan.fold_index,
                "Test"  : f"[{plan.split_regions.regions('test')[0].azimuth_start}, {plan.split_regions.regions('test')[0].azimuth_end})",
                "Val"   : f"[{plan.split_regions.regions('val')[0].azimuth_start}, {plan.split_regions.regions('val')[0].azimuth_end})",
                "Train" : ", ".join(f"[{r.azimuth_start}, {r.azimuth_end})" for r in train_regions),
            })
        self.logger.metrics_table(rows, ["Fold", "Test", "Val", "Train"])

        try:
            training_results  = self._run_training()
            inference_results = self._run_inference(planner)
            reports_dir       = self._run_reports(planner)

            self._mark_stage("pipeline", "completed")

            self.logger.section("Pipeline summary")
            self.logger.kv_table({
                "Folds"         : self.config.folds.n_folds,
                "Trained"       : sum(1 for r in training_results if r["status"] == "DONE"),
                "Inferred"      : sum(1 for r in inference_results if r["status"] == "DONE"),
                "Reports"       : str(reports_dir),
                "Pipeline logs" : str(self.pipeline_dir),
            }, title="Done")
        finally:
            self.logger.close()
