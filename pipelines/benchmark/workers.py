from __future__ import annotations

import traceback
from pathlib import Path

from configuration.experiments.benchmark_config import BenchmarkConfig
from pipelines.shared.config_factory import ConfigFactory
from tools.data.io import FileIO
from pipelines.backbone.training.docs import LossScaleProbeConfig


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

    def _ae_entry_config(self, model_name: str, logdir: Path, overfit):
        from configuration.training.autoencoder_config import ProfileAeEntryConfig
        from models.autoencoder import AE_CONFIG_REGISTRY

        return ProfileAeEntryConfig(
            run_name        = model_name,
            seed            = self.config.seed,
            n_gaussians     = self.config.n_gaussians,
            logdir          = logdir,
            pixel_subsample = self.config.pixel_subsample,
            keep_empty_frac = self.config.keep_empty_frac,
            ae_model_name   = model_name,
            autoencoder     = AE_CONFIG_REGISTRY[model_name](),
            ae_loss         = self.config.ae_loss,
            overfit         = overfit,
            paths           = self.config.paths,
            training        = self.config.training,
        )

    def _finalize_overfit(self, result: dict, result_path: Path) -> None:
        threshold = result["threshold"]

        if result["final_loss"] is not None:
            result["converged"] = bool(result["final_loss"] <= threshold)

        if result["status"] == "PASS" and result["final_loss"] is None:
            result["status"] = "FAIL"
            result["error"]  = result["error"] or "overfit gate produced no final loss to evaluate"

        if result["status"] == "PASS" and self.config.overfit.require_convergence and result["converged"] is not True:
            result["status"] = "FAIL"
            result["error"]  = f"final loss {result['final_loss']:.3e} above stop threshold {threshold:.0e}"

        FileIO.save_json(result, result_path, indent=2)

        if result["status"] == "FAIL":
            raise SystemExit(1)


class OverfitWorker(BenchmarkWorker):
    def _final_loss(self, outputs) -> float | None:
        if not isinstance(outputs, (tuple, list)) or len(outputs) == 0:
            return None

        train_losses = outputs[0]
        if not isinstance(train_losses, (list, tuple)) or len(train_losses) == 0:
            return None

        return float(train_losses[-1])

    def run(self, model_name: str) -> None:
        if self.config.training_type == "autoencoder":
            self._run_ae(model_name)
            return

        from models import CONFIG_REGISTRY
        from pipelines.backbone.training.pipeline import TrainingPipeline

        stage_dir    = self.run_dir / "overfit"
        result_path  = stage_dir / model_name / "overfit_result.json"
        model_config = self.factory.prepare_overfit_model_config(CONFIG_REGISTRY[model_name]())

        result = {
            "model"      : model_name,
            "status"     : None,
            "final_loss" : None,
            "converged"  : None,
            "threshold"  : self.config.overfit.stop_threshold,
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
            result["error"]  = "worker exited via SystemExit before reporting a final loss"
        except Exception:
            result["status"] = "FAIL"
            result["error"]  = traceback.format_exc()

        self._finalize_overfit(result, result_path)

    def _run_ae(self, model_name: str) -> None:
        from configuration.training.runtime_config import OverfitConfig
        from pipelines.profile_autoencoder.training.pipeline import TrainingPipeline

        gate        = self.config.overfit
        stage_dir   = self.run_dir / "overfit"
        result_path = stage_dir / model_name / "overfit_result.json"

        overfit = OverfitConfig(
            enabled        = True,
            max_steps      = gate.max_steps,
            stop_threshold = gate.stop_threshold,
            batch_size     = gate.batch_size,
        )

        result = {
            "model"      : model_name,
            "status"     : None,
            "final_loss" : None,
            "converged"  : None,
            "threshold"  : gate.stop_threshold,
            "error"      : None,
        }

        try:
            entry                   = self._ae_entry_config(model_name, stage_dir, overfit)
            (train_losses, _, _), _ = TrainingPipeline(entry).run()

            result["status"]     = "PASS"
            result["final_loss"] = float(train_losses[-1]) if train_losses else None
        except SystemExit:
            result["status"] = "PASS"
            result["error"]  = "worker exited via SystemExit before reporting a final loss"
        except Exception:
            result["status"] = "FAIL"
            result["error"]  = traceback.format_exc()

        self._finalize_overfit(result, result_path)


class TrainingWorker(BenchmarkWorker):
    def _size_overrides(self, model_name: str) -> dict:
        size_match_path = self.run_dir / "pipeline" / "size_match.json"
        if not size_match_path.exists():
            return {}

        records = FileIO.load_json(size_match_path)

        entry = records.get(model_name, {})
        return entry.get("overrides", {})

    def run(self, model_name: str) -> None:
        if self.config.training_type == "autoencoder":
            from configuration.training.runtime_config import OverfitConfig
            from pipelines.profile_autoencoder.training.pipeline import TrainingPipeline

            entry = self._ae_entry_config(model_name, self.run_dir / "training", OverfitConfig(enabled=False))
            TrainingPipeline(entry).run()
            return

        from models import CONFIG_REGISTRY
        from pipelines.backbone.training.pipeline import TrainingPipeline

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


class InferenceWorker(BenchmarkWorker):
    def run(self, model_name: str) -> None:
        from pipelines.backbone.inference.pipeline import InferencePipeline

        run_directory = self.run_dir / "training" / model_name

        pipeline = InferencePipeline(self.factory.inference_config(run_directory))
        pipeline.run()
