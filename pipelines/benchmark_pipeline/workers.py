from __future__ import annotations

import json
import traceback
from pathlib import Path

from configuration.benchmark_config import BenchmarkConfig
from pipelines.benchmark_pipeline.config_factory import ConfigFactory
from pipelines.training_pipeline.docs import LossScaleProbeConfig


class BenchmarkWorker:
    def __init__(self, config: BenchmarkConfig, run_tag: str) -> None:
        self.config  = config
        self.run_tag = run_tag
        self.run_dir = Path(config.paths.log_base_dir) / run_tag
        self.factory = ConfigFactory(config)

    def _probe_config(self) -> LossScaleProbeConfig:
        return LossScaleProbeConfig(
            enabled        = False,
            n_batches      = 100,
            reference      = "param_l1",
            exit_after     = True,
            enabled_losses = {},
        )


class OverfitWorker(BenchmarkWorker):
    def run(self, model_name: str) -> None:
        from models import CONFIG_REGISTRY
        from pipelines.training_pipeline.pipeline import TrainingPipeline

        stage_dir   = self.run_dir / "overfit"
        result_path = stage_dir / model_name / "overfit_result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)

        threshold    = self.config.overfit.stop_threshold
        model_config = self.factory.prepare_overfit_model_config(CONFIG_REGISTRY[model_name]())

        result = {
            "model"      : model_name,
            "status"     : None,
            "final_loss" : None,
            "converged"  : None,
            "threshold"  : threshold,
            "error"      : None,
        }

        try:
            pipeline = TrainingPipeline(
                trainer_config = self.factory.overfit_trainer_config(logdir=stage_dir),
                dataset_config = self.factory.overfit_dataset_config(),
                model_name     = model_name,
                model_config   = model_config,
                seed           = self.config.overfit.seed,
                run_name       = model_name,
            )

            outputs = pipeline.run(probe_config=self._probe_config())

            result["status"]     = "PASS"
            result["final_loss"] = self._final_loss(outputs)
        except SystemExit:
            result["status"] = "PASS"
        except Exception:
            result["status"] = "FAIL"
            result["error"]  = traceback.format_exc()

        if result["final_loss"] is not None:
            result["converged"] = bool(result["final_loss"] <= threshold)

        if result["status"] == "PASS" and self.config.overfit.require_convergence and result["converged"] is False:
            result["status"] = "FAIL"
            result["error"]  = f"final loss {result['final_loss']:.3e} above stop threshold {threshold:.0e}"

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        if result["status"] == "FAIL":
            raise SystemExit(1)

    def _final_loss(self, outputs) -> float | None:
        if not isinstance(outputs, (tuple, list)) or len(outputs) == 0:
            return None

        train_losses = outputs[0]
        if not isinstance(train_losses, (list, tuple)) or len(train_losses) == 0:
            return None

        return float(train_losses[-1])


class TrainingWorker(BenchmarkWorker):
    def run(self, model_name: str) -> None:
        from models import CONFIG_REGISTRY
        from pipelines.training_pipeline.pipeline import TrainingPipeline

        stage_dir    = self.run_dir / "training"
        model_config = CONFIG_REGISTRY[model_name]()

        for attribute, value in self._size_overrides(model_name).items():
            setattr(model_config, attribute, value)

        pipeline = TrainingPipeline(
            trainer_config = self.factory.training_trainer_config(logdir=stage_dir),
            dataset_config = self.factory.training_dataset_config(),
            model_name     = model_name,
            model_config   = model_config,
            seed           = self.config.seed,
            run_name       = model_name,
        )

        pipeline.run(probe_config=self._probe_config())

    def _size_overrides(self, model_name: str) -> dict:
        size_match_path = self.run_dir / "pipeline" / "size_match.json"
        if not size_match_path.exists():
            return {}

        with open(size_match_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        entry = records.get(model_name, {})
        return entry.get("overrides", {})


class InferenceWorker(BenchmarkWorker):
    def run(self, model_name: str) -> None:
        from pipelines.inference_pipeline.pipeline import InferencePipeline

        run_directory = self.run_dir / "training" / model_name

        pipeline = InferencePipeline(self.factory.inference_config(run_directory))
        pipeline.run()
