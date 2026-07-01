from __future__ import annotations

import traceback
from pathlib import Path

from configuration.benchmark import BenchmarkConfig
from pipelines.shared.config.config_factory            import ConfigFactory
from pipelines.shared.training.seed_sweep                import SeedSet
from tools.data.io                              import FileIO
from tools.training.pretraining.overfit_gate    import OverfitGate
from pipelines.backbone.training.loss_probe     import LossScaleProbeConfig


class BenchmarkWorker:
    def __init__(self, config: BenchmarkConfig, run_tag: str) -> None:
        self.config  = config
        self.run_tag = run_tag
        self.run_dir = Path(config.paths.log_base_dir) / run_tag
        self.factory = ConfigFactory(config)

    def _run_name(self, model_name: str, component: str | None, seed: int | None) -> str:
        unit = model_name if component is None else f"{model_name}__{component}"
        return unit if seed is None else SeedSet.run_name(unit, seed)

    def _apply_loss_component(self, trainer_config, component: str | None) -> None:
        if component is None:
            return

        from pipelines.backbone.training.loss import LossComponentCatalog

        trainer_config.curriculum = LossComponentCatalog.curriculum(component, base=self.config.curriculum.complete)

    def _size_overrides(self, model_name: str) -> dict:
        size_match_path = self.run_dir / "pipeline" / "size_match.json"
        if not size_match_path.exists():
            return {}

        records = FileIO.load_json(size_match_path)

        entry = records.get(model_name, {})
        return entry.get("overrides", {})

    def _max_batch_size(self, model_name: str) -> int | None:
        path = self.run_dir / "pipeline" / "max_batch.json"
        if not path.exists():
            return None

        records = FileIO.load_json(path)

        if model_name not in records:
            raise SystemExit(f"max_batch.json present but missing an entry for '{model_name}'")

        entry = records[model_name]
        if entry.get("status") != "PASS" or not entry.get("batch_size"):
            raise SystemExit(f"model '{model_name}' has no usable measured batch size: {entry}")

        return int(entry["batch_size"])

    def _probe_config(self) -> LossScaleProbeConfig:
        return LossScaleProbeConfig(
            enabled        = False,
            n_batches      = 100,
            reference      = "param_l1",
            exit_after     = True,
            enabled_losses = {},
        )

    def _ae_entry_config(self, model_name: str, logdir: Path, run_name: str | None = None, seed: int | None = None):
        from configuration.training import ProfileAeEntryConfig
        from models.profile_autoencoder                        import PROFILE_AE_CONFIG_REGISTRY

        return ProfileAeEntryConfig(
            run_name        = run_name or model_name,
            seed            = self.config.seed if seed is None else seed,
            n_gaussians     = self.config.n_gaussians,
            logdir          = logdir,
            pixel_subsample = self.config.pixel_subsample,
            keep_empty_frac = self.config.keep_empty_frac,
            ae_model_name   = model_name,
            autoencoder     = PROFILE_AE_CONFIG_REGISTRY[model_name](),
            ae_loss         = self.config.ae_loss,
            paths           = self.config.paths,
            training        = self.config.training,
        )

    def _jepa_entry_config(self, model_name: str, logdir: Path, run_name: str | None = None, seed: int | None = None):
        from configuration.training import JepaEntryConfig

        jepa = self.config.jepa

        return JepaEntryConfig(
            run_name                   = run_name or model_name,
            backbone_name              = model_name,
            seed                       = self.config.seed if seed is None else seed,
            n_gaussians                = self.config.n_gaussians,
            logdir                     = logdir,
            profile_autoencoder_logdir = jepa.profile_autoencoder_logdir,
            profile_autoencoder_run    = jepa.profile_autoencoder_run,
            profile_autoencoder_mode   = jepa.profile_autoencoder_mode,
            target_provider            = jepa.target_provider,
            embedding_loss             = jepa.embedding_loss,
            paths                      = self.config.paths,
            training                   = self.config.training,
        )

    def _finalize_overfit(self, result: dict, result_path: Path) -> None:
        OverfitGate.evaluate(result, self.config.overfit.require_convergence)

        FileIO.save_json(result, result_path, indent=2)

        if result["status"] == "FAIL":
            raise SystemExit(1)


class OverfitModelPreparer:
    def __init__(self, model_config) -> None:
        self.model_config = model_config

    def _disable_regularization(self) -> None:
        for attribute in ("dropout", "attention_dropout", "stochastic_depth_rate"):
            if hasattr(self.model_config, attribute):
                setattr(self.model_config, attribute, 0.0)

        for attribute in vars(self.model_config):
            if attribute.endswith("_wd"):
                setattr(self.model_config, attribute, 0.0)

    def _boost_learning_rates(self) -> None:
        for attribute in vars(self.model_config):
            if attribute.endswith("_lr"):
                setattr(self.model_config, attribute, getattr(self.model_config, attribute) * 10.0)

    def prepare(self):
        self._disable_regularization()
        self._boost_learning_rates()

        return self.model_config


class OverfitWorker(BenchmarkWorker):
    def _final_loss(self, outputs) -> float | None:
        if not isinstance(outputs, (tuple, list)) or len(outputs) == 0:
            return None

        train_losses = outputs[0]
        if not isinstance(train_losses, (list, tuple)) or len(train_losses) == 0:
            return None

        return float(train_losses[-1])

    def _execute_overfit(self, model_name: str, threshold: float, result_path: Path, run_body) -> None:
        result = {
            "model"      : model_name,
            "status"     : None,
            "final_loss" : None,
            "converged"  : None,
            "threshold"  : threshold,
            "error"      : None,
        }

        try:
            result["final_loss"] = run_body()
            result["status"]     = "PASS"
        except SystemExit:
            result["status"] = "PASS"
            result["error"]  = "worker exited via SystemExit before reporting a final loss"
        except Exception:
            result["status"] = "FAIL"
            result["error"]  = traceback.format_exc()

        self._finalize_overfit(result, result_path)

    def run(self, model_name: str, seed: int | None = None, loss_component: str | None = None) -> None:
        run_name = self._run_name(model_name, loss_component, seed)

        if self.config.training_type == "profile_autoencoder":
            self._run_ae(model_name, run_name, seed)
            return
        if self.config.training_type == "jepa":
            self._run_jepa(model_name, run_name, seed)
            return

        from models                               import BACKBONE_CONFIG_REGISTRY
        from pipelines.backbone.training.pipeline import TrainingPipeline

        stage_dir    = self.run_dir / "overfit"
        result_path  = stage_dir / run_name / "overfit_result.json"
        model_config = BACKBONE_CONFIG_REGISTRY[model_name]()

        for attribute, value in self._size_overrides(model_name).items():
            setattr(model_config, attribute, value)

        model_config = OverfitModelPreparer(model_config).prepare()

        def run_body():
            trainer_config          = self.factory.overfit_trainer_config(logdir=stage_dir)
            trainer_config.geometry = self.config.geometry.resolved(self.config.paths.dataset_path, secondary_labels=self.factory._secondary_labels())

            dataset_config              = self.factory.overfit_dataset_config()
            dataset_config.input_config = self.config.input

            pipeline = TrainingPipeline(
                trainer_config = trainer_config,
                dataset_config = dataset_config,
                backbone_name  = model_name,
                model_config   = model_config,
                seed           = self.config.overfit.seed if seed is None else seed,
                run_name       = run_name,
            )

            return self._final_loss(pipeline.run(probe_config=self._probe_config()))

        self._execute_overfit(run_name, self.config.overfit.stop_threshold, result_path, run_body)

    def _overfit_config(self):
        from configuration.training import OverfitConfig

        gate = self.config.overfit
        return OverfitConfig(
            enabled        = True,
            max_steps      = gate.max_steps,
            stop_threshold = gate.stop_threshold,
            batch_size     = gate.batch_size,
        )

    def _run_ae(self, model_name: str, run_name: str, seed: int | None) -> None:
        from pipelines.profile_autoencoder.training.pipeline import TrainingPipeline

        gate        = self.config.overfit
        stage_dir   = self.run_dir / "overfit"
        result_path = stage_dir / run_name / "overfit_result.json"

        overfit = self._overfit_config()

        def run_body():
            entry                   = self._ae_entry_config(model_name, stage_dir, run_name=run_name, seed=seed)
            (train_losses, _, _), _ = TrainingPipeline(entry, overfit=overfit).run()

            return float(train_losses[-1]) if train_losses else None

        self._execute_overfit(run_name, gate.stop_threshold, result_path, run_body)

    def _run_jepa(self, model_name: str, run_name: str, seed: int | None) -> None:
        from pipelines.jepa.training.pipeline import TrainingPipeline

        gate        = self.config.overfit
        stage_dir   = self.run_dir / "overfit"
        result_path = stage_dir / run_name / "overfit_result.json"

        overfit = self._overfit_config()

        def run_body():
            entry                   = self._jepa_entry_config(model_name, stage_dir, run_name=run_name, seed=seed)
            (train_losses, _, _), _ = TrainingPipeline(entry, overfit=overfit).run()

            return float(train_losses[-1]) if train_losses else None

        self._execute_overfit(run_name, gate.stop_threshold, result_path, run_body)


class MaxBatchWorker(BenchmarkWorker):
    def run(self, model_name: str) -> None:
        from pipelines.benchmark.batch_probe import MaxBatchProbe

        stage_dir   = self.run_dir / "max_batch"
        result_path = stage_dir / model_name / "max_batch_result.json"

        probe  = MaxBatchProbe(config=self.config, model_name=model_name, overrides=self._size_overrides(model_name))
        result = probe.run()

        FileIO.save_json(result, result_path, indent=2)

        if result["status"] == "FAIL":
            raise SystemExit(1)


class TrainingWorker(BenchmarkWorker):
    def run(self, model_name: str, seed: int | None = None, loss_component: str | None = None) -> None:
        run_name = self._run_name(model_name, loss_component, seed)

        if self.config.training_type == "profile_autoencoder":
            from pipelines.profile_autoencoder.training.pipeline import TrainingPipeline

            entry = self._ae_entry_config(model_name, self.run_dir / "training", run_name=run_name, seed=seed)
            TrainingPipeline(entry).run()
            return

        if self.config.training_type == "jepa":
            from pipelines.jepa.training.pipeline      import TrainingPipeline

            entry = self._jepa_entry_config(model_name, self.run_dir / "training", run_name=run_name, seed=seed)
            TrainingPipeline(entry).run()
            return

        from models                               import BACKBONE_CONFIG_REGISTRY
        from pipelines.backbone.training.pipeline import TrainingPipeline

        stage_dir    = self.run_dir / "training"
        model_config = BACKBONE_CONFIG_REGISTRY[model_name]()

        for attribute, value in self._size_overrides(model_name).items():
            setattr(model_config, attribute, value)

        measured_batch = self._max_batch_size(model_name)
        if measured_batch is not None:
            self.config.training.batch_size = measured_batch

        trainer_config          = self.factory.training_trainer_config(logdir=stage_dir)
        trainer_config.geometry = self.config.geometry.resolved(self.config.paths.dataset_path, secondary_labels=self.factory._secondary_labels())
        self._apply_loss_component(trainer_config, loss_component)

        dataset_config              = self.factory.training_dataset_config()
        dataset_config.input_config = self.config.input

        pipeline = TrainingPipeline(
            trainer_config = trainer_config,
            dataset_config = dataset_config,
            backbone_name  = model_name,
            model_config   = model_config,
            seed           = self.config.seed if seed is None else seed,
            run_name       = run_name,
        )

        pipeline.run(probe_config=self._probe_config())


class InferenceWorker(BenchmarkWorker):
    def run(self, model_name: str, seed: int | None = None, loss_component: str | None = None) -> None:
        from pipelines.backbone.inference.pipeline import InferencePipeline
        from pipelines.shared.inference.inference_components import InferenceComponentsResolver

        run_directory = self.run_dir / "training" / self._run_name(model_name, loss_component, seed)

        components = InferenceComponentsResolver.for_run(run_directory)
        pipeline   = InferencePipeline(self.factory.inference_config(run_directory), components=components)
        pipeline.run()
