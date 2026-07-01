from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pipelines.benchmark.stages import (
    ComparisonStage,
    InferenceStage,
    MaxBatchStage,
    OverfitStage,
    SizeMatchStage,
    TrainingStage,
)

from configuration.benchmark import BenchmarkConfig
from tools.data.io import FileIO


class _SilentLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.fixture
def logger_stub():
    return _SilentLogger()


@pytest.fixture
def config(tmp_path):
    config                    = BenchmarkConfig()
    config.paths.log_base_dir = tmp_path
    config.resume             = False
    return config


ENTRY = Path("/entry/run_benchmark.py")


def test_overfit_passed_true_when_all_pass(config, logger_stub):
    stage = OverfitStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    assert stage.passed([{"status": "PASS"}, {"status": "PASS"}]) is True


def test_overfit_passed_false_on_any_fail(config, logger_stub):
    stage = OverfitStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    assert stage.passed([{"status": "PASS"}, {"status": "FAIL"}]) is False


def test_overfit_passed_false_when_empty(config, logger_stub):
    stage = OverfitStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    assert stage.passed([]) is False


def test_overfit_job_command_carries_worker_flags(config, logger_stub):
    stage = OverfitStage(config=config, entry_script=ENTRY, run_tag="tag9", models=["unet"], logger=logger_stub)

    job = stage._job("unet")

    assert job.command[0] == sys.executable
    assert str(ENTRY) in job.command
    assert job.command[job.command.index("--worker")  + 1] == "overfit"
    assert job.command[job.command.index("--model")   + 1] == "unet"
    assert "--loss-component" not in job.command
    assert job.command[job.command.index("--run-tag") + 1] == "tag9"


def test_maxbatch_job_command_uses_maxbatch_worker(config, logger_stub):
    stage = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    job = stage._job("unet")

    assert job.command[job.command.index("--worker") + 1] == "maxbatch"


def test_overfit_missing_result_is_failure_record(config, logger_stub):
    stage  = OverfitStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)
    record = stage._load_result("unet")

    assert record["status"] == "FAIL"
    assert record["model"]  == "unet"
    assert "missing result file" in record["error"]


def test_maxbatch_missing_result_carries_budget_and_ceiling(config, logger_stub):
    stage  = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)
    record = stage._load_result("unet")

    assert record["status"]    == "FAIL"
    assert record["budget_gb"] == config.max_batch.vram_budget_gb
    assert record["ceiling"]   == config.max_batch.max_batch


def test_overfit_run_orders_results_and_writes_files(config, logger_stub, monkeypatch):
    models = ["unet", "resunet"]
    stage  = OverfitStage(config=config, entry_script=ENTRY, run_tag="t", models=models, logger=logger_stub)

    def fake_queue(jobs):
        for job in jobs:
            path = stage._result_path(job.name)
            FileIO.save_json({"model": job.name, "status": "PASS", "final_loss": 1e-4, "converged": True}, path)
        return []

    monkeypatch.setattr(stage, "_run_queue", fake_queue)

    results = stage.run()

    assert [r["model"] for r in results] == ["unet", "resunet"]
    assert stage.results_path.exists()
    assert stage.report_path.exists()


def test_maxbatch_run_collects_records_keyed_by_model(config, logger_stub, monkeypatch):
    models = ["unet", "resunet"]
    stage  = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=models, logger=logger_stub)

    def fake_queue(jobs):
        for job in jobs:
            path = stage._result_path(job.name)
            FileIO.save_json({"model": job.name, "status": "PASS", "batch_size": 64, "peak_gb": 10.0}, path)
        return []

    monkeypatch.setattr(stage, "_run_queue", fake_queue)

    records = stage.run()

    assert set(records) == set(models)
    assert records["unet"]["batch_size"] == 64
    assert stage.records_path.exists()


def test_overfit_resume_reuses_cached_result(config, logger_stub, monkeypatch):
    config.resume = True
    stage = OverfitStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    FileIO.save_json({"model": "unet", "status": "PASS", "final_loss": 1e-4, "converged": True}, stage._result_path("unet"))

    called = {"queue": False}

    def fake_queue(jobs):
        called["queue"] = True
        return []

    monkeypatch.setattr(stage, "_run_queue", fake_queue)

    results = stage.run()

    assert called["queue"] is False
    assert results[0]["status"] == "PASS"


def test_sizematch_reference_record_is_zero_deviation(config, logger_stub):
    stage  = SizeMatchStage(config=config, run_tag="t", models=["unet"], logger=logger_stub)
    record = stage._reference_record("unet", 1234)

    assert record["model"]         == "unet"
    assert record["scale"]         == 1.0
    assert record["parameters"]    == 1234
    assert record["target"]        == 1234
    assert record["deviation_pct"] == 0.0
    assert record["iterations"]    == 0


@pytest.mark.real_data
@pytest.mark.slow
def test_sizematch_run_emits_reference_and_matched(monkeypatch, logger_stub, test_data_dir, tmp_path):
    config                    = BenchmarkConfig()
    config.paths.log_base_dir = tmp_path
    config.paths.dataset_path = test_data_dir
    config.resume             = False
    config.skip_models        = []

    stage = SizeMatchStage(config=config, run_tag="t", models=["unet", "resunet"], logger=logger_stub)

    records = stage.run()

    assert "unet" in records
    assert "resunet" in records
    assert records["unet"]["scale"] == 1.0
    assert stage.records_path.exists()


def test_sizematch_load_cached_requires_all_models(config, logger_stub):
    config.resume = True
    stage = SizeMatchStage(config=config, run_tag="t", models=["unet", "resunet"], logger=logger_stub)

    FileIO.save_json({"unet": {}}, stage.records_path)

    assert stage._load_cached() is None


def test_sizematch_load_cached_returns_when_complete(config, logger_stub):
    config.resume = True
    stage = SizeMatchStage(config=config, run_tag="t", models=["unet", "resunet"], logger=logger_stub)

    FileIO.save_json({"unet": {}, "resunet": {}}, stage.records_path)

    cached = stage._load_cached()
    assert set(cached) == {"unet", "resunet"}


def test_training_and_inference_stages_subclass_queued_bases(config, logger_stub):
    from tools.orchestration import QueuedInferenceStage, QueuedTrainingStage

    train = TrainingStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)
    infer = InferenceStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    assert isinstance(train, QueuedTrainingStage)
    assert isinstance(infer, QueuedInferenceStage)
    assert train.worker_action == "train"
    assert infer.worker_action == "infer"


def test_comparison_stage_run_invokes_collector_and_report(config, logger_stub, monkeypatch):
    stage = ComparisonStage(config=config, run_tag="t", logger=logger_stub)

    collected = {}

    class FakeCollector:
        def __init__(self, run_dir, logger):
            collected["run_dir"]  = run_dir
            self.seed_dispersion  = {"unet": {"n_seeds": 2}}
        def collect(self):
            return []

    class FakeReport:
        def __init__(self, **kwargs):
            collected["kwargs"] = kwargs
        def write_all(self):
            return [Path("/tmp/x.md")]

    monkeypatch.setattr("pipelines.benchmark.stages.BenchmarkSeedCollector", FakeCollector)
    monkeypatch.setattr("pipelines.benchmark.stages.ComparisonReport", FakeReport)

    out_dir = stage.run()

    assert collected["run_dir"] == stage.run_dir
    assert collected["kwargs"]["reference_model"] == config.size_match.reference_model
    assert collected["kwargs"]["seed_dispersion"] == {"unet": {"n_seeds": 2}}
    assert "comparison" in str(out_dir)


def test_overfit_names_are_model_only(config, logger_stub):
    stage = OverfitStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet", "vit"], logger=logger_stub)

    assert stage.names == ["unet", "vit"]
    assert "--seed" not in stage._job("unet").command


def test_overfit_gate_is_loss_agnostic(config, logger_stub):
    config.sweep_loss_components = ["param_l1", "covariance_match"]
    stage = OverfitStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet", "vit"], logger=logger_stub)

    assert stage.names == ["unet", "vit"]

    job = stage._job("vit")
    assert job.command[job.command.index("--model") + 1] == "vit"
    assert "--loss-component" not in job.command


def test_overfit_seed_sweep_expands_runs_per_model_and_seed(config, logger_stub):
    config.seeds                 = [1, 2]
    config.sweep_loss_components = ["param_l1", "covariance_match"]
    stage = OverfitStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet", "vit"], logger=logger_stub)

    assert stage.names == ["unet_seed1", "unet_seed2", "vit_seed1", "vit_seed2"]

    job = stage._job("unet_seed2")
    assert job.command[job.command.index("--model")  + 1] == "unet"
    assert "--loss-component" not in job.command
    assert job.command[job.command.index("--seed")   + 1] == "2"
    assert job.command[job.command.index("--worker") + 1] == "overfit"
    assert job.log_path == stage.stage_dir / "unet_seed2" / "worker.log"
    assert stage._result_path("unet_seed2") == stage.stage_dir / "unet_seed2" / "overfit_result.json"


def test_training_seed_sweep_job_uses_base_model_and_seed(config, logger_stub):
    config.seeds = [7]
    stage = TrainingStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    assert stage.items == ["unet__param_l1_seed7"]

    job = stage._job("unet__param_l1_seed7")
    assert job.command[job.command.index("--model") + 1] == "unet"
    assert job.command[job.command.index("--seed")  + 1] == "7"
    assert job.command[job.command.index("--worker") + 1] == "train"
    assert job.log_path == stage.stage_dir / "unet__param_l1_seed7" / "worker.log"


def test_inference_seed_sweep_matches_training_run_names(config, logger_stub):
    config.seeds = [3, 4]
    stage = InferenceStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    assert stage.items == ["unet__param_l1_seed3", "unet__param_l1_seed4"]

    job = stage._job("unet__param_l1_seed4")
    assert job.command[job.command.index("--model") + 1] == "unet"
    assert job.command[job.command.index("--seed")  + 1] == "4"
    assert job.command[job.command.index("--worker") + 1] == "infer"
