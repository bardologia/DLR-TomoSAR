from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from configuration.benchmark_config import BenchmarkConfig
from models import CONFIG_REGISTRY
from tools.config_cli import ConfigCli
from tools.logger import Logger

from pipelines.benchmark_pipeline.comparison_stage import ComparisonStage
from pipelines.benchmark_pipeline.inference_stage import InferenceStage
from pipelines.benchmark_pipeline.overfit_stage import OverfitStage
from pipelines.benchmark_pipeline.size_match_stage import SizeMatchStage
from pipelines.benchmark_pipeline.training_stage import TrainingStage


class BenchmarkPipeline:
    def __init__(self, config: BenchmarkConfig, entry_script: Path) -> None:
        self.config       = config
        self.entry_script = entry_script
        self.run_tag      = config.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_dir      = Path(config.paths.log_base_dir) / self.run_tag
        self.pipeline_dir = self.run_dir / "pipeline"
        self.state_path   = self.pipeline_dir / "state.json"

        self.pipeline_dir.mkdir(parents=True, exist_ok=True)
        ConfigCli.save_resolved(config, self.pipeline_dir / "resolved_config.json")

        self.logger = Logger(log_dir=str(self.pipeline_dir), name="benchmark_pipeline")
        self.models = [m for m in CONFIG_REGISTRY.keys() if m not in set(config.skip_models)]
        self.state  = {"run_tag": self.run_tag, "stages": {}}

    def run(self) -> None:
        self.logger.section("Benchmark pipeline")
        self.logger.kv_table({
            "Run tag"   : self.run_tag,
            "Models"    : len(self.models),
            "GPUs"      : self.config.gpus,
            "Reference" : self.config.size_match.reference_model,
            "Epochs"    : self.config.training.epochs,
            "Resume"    : self.config.resume,
            "Run dir"   : str(self.run_dir),
        }, title="Configuration")

        try:
            gate_passed = self._run_overfit_gate()
            if not gate_passed and self.config.overfit.abort_on_fail:
                self._mark_stage("pipeline", "aborted")
                self.logger.error("Overfit gate failed — pipeline aborted. See the overfit report before retrying.")
                raise SystemExit(1)

            self._run_size_match()
            training_results  = self._run_training()
            inference_results = self._run_inference()
            comparison_dir    = self._run_comparison()

            self._mark_stage("pipeline", "completed")

            self.logger.section("Pipeline summary")
            self.logger.kv_table({
                "Models"        : len(self.models),
                "Trained"       : sum(1 for r in training_results if r["status"] == "DONE"),
                "Inferred"      : sum(1 for r in inference_results if r["status"] == "DONE"),
                "Comparison"    : str(comparison_dir),
                "Pipeline logs" : str(self.pipeline_dir),
            }, title="Done")
        finally:
            self.logger.close()

    def _run_overfit_gate(self) -> bool:
        stage   = OverfitStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, models=self.models, logger=self.logger)
        results = stage.run()
        passed  = stage.passed(results)

        self._mark_stage("overfit", "completed" if passed else "failed")
        return passed

    def _run_size_match(self) -> dict:
        stage   = SizeMatchStage(config=self.config, run_tag=self.run_tag, models=self.models, logger=self.logger)
        records = stage.run()

        self._mark_stage("size_match", "completed")
        return records

    def _run_training(self) -> list[dict]:
        stage   = TrainingStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, models=self.models, logger=self.logger)
        results = stage.run()

        failed = [r for r in results if r["status"] != "DONE"]
        self._mark_stage("training", "completed" if not failed else "partial")
        return results

    def _run_inference(self) -> list[dict]:
        stage   = InferenceStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, models=self.models, logger=self.logger)
        results = stage.run()

        failed = [r for r in results if r["status"] == "FAILED"]
        self._mark_stage("inference", "completed" if not failed else "partial")
        return results

    def _run_comparison(self) -> Path:
        stage   = ComparisonStage(config=self.config, run_tag=self.run_tag, logger=self.logger)
        out_dir = stage.run()

        self._mark_stage("comparison", "completed")
        return out_dir

    def _mark_stage(self, stage_name: str, status: str) -> None:
        self.state["stages"][stage_name] = {
            "status"    : status,
            "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
