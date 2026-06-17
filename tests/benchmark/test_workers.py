from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.benchmark.workers import (
    BenchmarkWorker,
    MaxBatchWorker,
    OverfitModelPreparer,
    OverfitWorker,
)

from configuration.benchmark import BenchmarkConfig
from models import BACKBONE_CONFIG_REGISTRY
from tools.data.io import FileIO


@pytest.fixture
def config(tmp_path):
    config                    = BenchmarkConfig()
    config.paths.log_base_dir = tmp_path
    return config


def test_overfit_preparer_zeroes_regularization():
    model_config = BACKBONE_CONFIG_REGISTRY["unet"]()
    model_config.dropout = 0.3

    prepared = OverfitModelPreparer(model_config).prepare()

    assert prepared.dropout    == 0.0
    assert prepared.encoder_wd == 0.0


def test_overfit_preparer_boosts_learning_rates_tenfold():
    model_config = BACKBONE_CONFIG_REGISTRY["unet"]()
    base_lr      = model_config.encoder_lr

    prepared = OverfitModelPreparer(model_config).prepare()

    assert prepared.encoder_lr == pytest.approx(base_lr * 10.0)


def test_size_overrides_empty_without_file(config):
    worker = BenchmarkWorker(config, "tag")

    assert worker._size_overrides("unet") == {}


def test_size_overrides_reads_overrides(config):
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "size_match.json"

    FileIO.save_json({"unet": {"overrides": {"features": [32, 64]}}}, path)

    assert worker._size_overrides("unet") == {"features": [32, 64]}


def test_max_batch_size_none_without_file(config):
    worker = BenchmarkWorker(config, "tag")

    assert worker._max_batch_size("unet") is None


def test_max_batch_size_returns_measured_value(config):
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "max_batch.json"

    FileIO.save_json({"unet": {"status": "PASS", "batch_size": 128}}, path)

    assert worker._max_batch_size("unet") == 128


def test_max_batch_size_raises_when_model_missing(config):
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "max_batch.json"

    FileIO.save_json({"resunet": {"status": "PASS", "batch_size": 128}}, path)

    with pytest.raises(SystemExit):
        worker._max_batch_size("unet")


def test_max_batch_size_raises_when_not_passed(config):
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "max_batch.json"

    FileIO.save_json({"unet": {"status": "FAIL", "batch_size": None}}, path)

    with pytest.raises(SystemExit):
        worker._max_batch_size("unet")


def test_probe_config_is_disabled(config):
    worker = BenchmarkWorker(config, "tag")
    probe  = worker._probe_config()

    assert probe.enabled is False
    assert probe.exit_after is True


def test_overfit_final_loss_extracts_last_train_loss(config):
    worker = OverfitWorker(config, "tag")

    assert worker._final_loss(([1.0, 0.5, 0.1], [], [])) == pytest.approx(0.1)


def test_overfit_final_loss_none_for_empty(config):
    worker = OverfitWorker(config, "tag")

    assert worker._final_loss(([], [], [])) is None
    assert worker._final_loss(None) is None


def _result(status, final_loss, threshold, error=None):
    return {
        "model"      : "unet",
        "status"     : status,
        "final_loss" : final_loss,
        "converged"  : None,
        "threshold"  : threshold,
        "error"      : error,
    }


def test_finalize_overfit_marks_converged(config, tmp_path):
    worker      = OverfitWorker(config, "tag")
    config.overfit.require_convergence = False
    result      = _result("PASS", final_loss=1e-4, threshold=1e-3)
    result_path = tmp_path / "overfit_result.json"

    worker._finalize_overfit(result, result_path)

    assert result["converged"] is True
    assert result["status"]    == "PASS"
    assert result_path.exists()


def test_finalize_overfit_fails_without_final_loss(config, tmp_path):
    worker      = OverfitWorker(config, "tag")
    result      = _result("PASS", final_loss=None, threshold=1e-3)
    result_path = tmp_path / "overfit_result.json"

    with pytest.raises(SystemExit):
        worker._finalize_overfit(result, result_path)

    assert result["status"] == "FAIL"


def test_finalize_overfit_fails_on_nonconvergence_when_required(config, tmp_path):
    worker      = OverfitWorker(config, "tag")
    config.overfit.require_convergence = True
    result      = _result("PASS", final_loss=1.0, threshold=1e-3)
    result_path = tmp_path / "overfit_result.json"

    with pytest.raises(SystemExit):
        worker._finalize_overfit(result, result_path)

    assert result["status"]   == "FAIL"
    assert result["converged"] is False


def test_max_batch_worker_invokes_real_probe(config, monkeypatch):
    worker = MaxBatchWorker(config, "tag")

    calls = {}

    class FakeProbe:
        def __init__(self, config, model_name, overrides):
            calls["model_name"] = model_name
            calls["overrides"]  = overrides
        def run(self):
            return {"model": "unet", "status": "PASS", "batch_size": 64, "peak_gb": 12.0}

    monkeypatch.setattr("pipelines.benchmark.batch_probe.MaxBatchProbe", FakeProbe)

    worker.run("unet")

    assert calls["model_name"] == "unet"
    result_path = worker.run_dir / "max_batch" / "unet" / "max_batch_result.json"
    assert result_path.exists()
    assert FileIO.load_json(result_path)["batch_size"] == 64


def test_max_batch_worker_raises_on_probe_fail(config, monkeypatch):
    worker = MaxBatchWorker(config, "tag")

    class FakeProbe:
        def __init__(self, *args, **kwargs):
            pass
        def run(self):
            return {"model": "unet", "status": "FAIL", "batch_size": None, "peak_gb": None}

    monkeypatch.setattr("pipelines.benchmark.batch_probe.MaxBatchProbe", FakeProbe)

    with pytest.raises(SystemExit):
        worker.run("unet")
