from __future__ import annotations

import inspect

import pytest
import torch

import pipelines.benchmark.batch_probe as module
from configuration.benchmark import BenchmarkConfig
from pipelines.backbone.training.loss_terms import LossComponentCatalog
from pipelines.backbone.training.pipeline import TrainingPipeline
from pipelines.benchmark.batch_probe import MaxBatchProbe
from tools.monitoring.logger import Logger
from tools.training.pretraining.batch_finder import TrainStepMemoryProbe


GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="batch probe memory measurement requires CUDA")


def _bare_probe(ceiling):
    probe         = MaxBatchProbe.__new__(MaxBatchProbe)
    probe.ceiling = ceiling
    return probe


def test_candidates_are_powers_of_two_up_to_ceiling():
    probe = _bare_probe(ceiling=16)

    assert probe._candidates() == [1, 2, 4, 8, 16]


def test_candidates_stop_below_non_power_of_two_ceiling():
    probe = _bare_probe(ceiling=10)

    assert probe._candidates() == [1, 2, 4, 8]


def test_trial_uses_real_trainer_loss_not_surrogate():
    source = inspect.getsource(TrainStepMemoryProbe.__call__)

    assert "_compute_loss" in source
    assert "total_loss" in source
    assert "loss.backward()" in source
    assert "mse_loss" not in source


def test_trial_measures_peak_reserved_memory():
    source = inspect.getsource(TrainStepMemoryProbe.__call__)

    assert "reset_peak_memory_stats" in source
    assert "max_memory_reserved" in source


def test_build_trainer_constructs_real_trainer():
    source = inspect.getsource(MaxBatchProbe._build_trainer)

    assert "Trainer(" in source
    assert "set_curriculum" in source


def test_build_context_imports_real_dataset_and_trainer_pipeline():
    source = inspect.getsource(module)

    assert "DatasetPipeline" in source
    assert "from pipelines.backbone.training.trainer import Trainer" in source


def test_run_catches_exceptions_and_reports_failure_dict(monkeypatch, tmp_path):
    config                    = BenchmarkConfig()
    config.paths.log_base_dir = tmp_path

    probe = _bare_probe(ceiling=config.max_batch.max_batch)
    probe.config        = config
    probe.model_name    = "unet"
    probe.overrides     = {}
    probe.budget_gb     = config.max_batch.vram_budget_gb
    probe.measure_steps = config.max_batch.measure_steps
    probe.seed          = config.max_batch.seed
    probe.device        = torch.device("cpu")
    probe.context_gb    = 0.0
    probe.work_dir      = tmp_path / "work"
    probe.work_dir.mkdir(parents=True, exist_ok=True)

    probe.logger = Logger(log_dir=str(tmp_path / "logs"), name="probe", level="INFO")

    monkeypatch.setattr(probe, "_measure_context", lambda: (_ for _ in ()).throw(RuntimeError("no cuda")))

    result = probe.run()

    assert result["status"] == "FAIL"
    assert result["model"]  == "unet"
    assert result["error"] is not None


@GPU
def test_probe_init_requires_cuda_device(tmp_path):
    config                    = BenchmarkConfig()
    config.paths.log_base_dir = tmp_path

    probe = MaxBatchProbe(config=config, model_name="unet", overrides={})

    assert probe.device.type == "cuda"
    assert probe.budget_gb == config.max_batch.vram_budget_gb
    assert probe.ceiling   == config.max_batch.max_batch


def test_run_caps_the_ceiling_at_the_dataset_size():
    source = inspect.getsource(MaxBatchProbe.run)

    assert "min(self.ceiling, len(dataset))" in source


def test_run_validates_size_match_channels_against_the_dataset():
    source = inspect.getsource(MaxBatchProbe.run)

    assert "dataset.input_channels != self.config.size_match.in_channels" in source


def test_run_records_the_measured_context():
    source = inspect.getsource(MaxBatchProbe.run)

    assert 'result["context_gb"] = self.context_gb' in source


def test_build_context_probes_with_the_swept_loss_union():
    source = inspect.getsource(MaxBatchProbe._build_context)

    assert "LossComponentCatalog.combined_curriculum(self.config.sweep_loss_components, base=self.config.loss)" in source
    assert "build_geometry_field=TrainingPipeline.physics_geometry_active(trainer_config)" in source


def test_probe_curriculum_activates_geometry_field_for_physics_sweeps():
    config = BenchmarkConfig()

    config.sweep_loss_components = ["param_l1"]
    plain = LossComponentCatalog.combined_curriculum(config.sweep_loss_components, base=config.loss)
    stub  = type("Cfg", (), {"curriculum": plain})()
    assert TrainingPipeline.physics_geometry_active(stub) is False

    config.sweep_loss_components = ["param_l1", "covariance_match"]
    physics = LossComponentCatalog.combined_curriculum(config.sweep_loss_components, base=config.loss)
    stub    = type("Cfg", (), {"curriculum": physics})()
    assert TrainingPipeline.physics_geometry_active(stub) is True
