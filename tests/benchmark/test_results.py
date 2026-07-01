from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pipelines.benchmark.results import BenchmarkSeedCollector, ComparisonReport, TrialCollector, TrialRecord

from tools.data.io import FileIO


class _SilentLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.fixture
def logger_stub():
    return _SilentLogger()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_trial_record_has_inference_reflects_inference_dir(tmp_path):
    record = TrialRecord(name="unet", run_dir=tmp_path)
    assert record.has_inference is False

    record.inference_dir = tmp_path / "inf"
    assert record.has_inference is True


def test_collector_returns_empty_without_training_dir(tmp_path, logger_stub):
    (tmp_path / "pipeline").mkdir(parents=True)
    _write_json(tmp_path / "pipeline" / "overfit_results.json", [])
    _write_json(tmp_path / "pipeline" / "training_results.json", [])

    records = TrialCollector(run_dir=tmp_path, logger=logger_stub).collect()

    assert records == []


def test_collector_parses_parameters_from_model_doc(tmp_path, logger_stub):
    (tmp_path / "pipeline").mkdir(parents=True)
    _write_json(tmp_path / "pipeline" / "size_match.json", {})
    _write_json(tmp_path / "pipeline" / "overfit_results.json", [])
    _write_json(tmp_path / "pipeline" / "training_results.json", [])

    trial_docs = tmp_path / "training" / "unet" / "docs"
    trial_docs.mkdir(parents=True)
    (trial_docs / "model_doc.md").write_text("**Total Parameters:** `12,345,678`\n", encoding="utf-8")

    records = TrialCollector(run_dir=tmp_path, logger=logger_stub).collect()

    assert len(records) == 1
    assert records[0].parameters == 12345678


def test_collector_falls_back_to_size_match_parameters(tmp_path, logger_stub):
    (tmp_path / "pipeline").mkdir(parents=True)
    _write_json(tmp_path / "pipeline" / "size_match.json", {"unet": {"parameters": 999}})
    _write_json(tmp_path / "pipeline" / "overfit_results.json", [])
    _write_json(tmp_path / "pipeline" / "training_results.json", [])

    (tmp_path / "training" / "unet").mkdir(parents=True)

    records = TrialCollector(run_dir=tmp_path, logger=logger_stub).collect()

    assert records[0].parameters == 999


def test_collector_reads_checkpoint_with_weights_only_false(tmp_path, logger_stub):
    (tmp_path / "pipeline").mkdir(parents=True)
    _write_json(tmp_path / "pipeline" / "size_match.json", {})
    _write_json(tmp_path / "pipeline" / "overfit_results.json", [])
    _write_json(tmp_path / "pipeline" / "training_results.json", [])

    trial_dir = tmp_path / "training" / "unet"
    trial_dir.mkdir(parents=True)

    checkpoint = {
        "best_val_loss" : 0.123,
        "best_epoch"    : 7,
        "epoch"         : 9,
        "global_step"   : 900,
        "train_losses"  : [1.0, 0.5, 0.123],
        "val_losses"    : [1.1, 0.6],
    }
    torch.save(checkpoint, trial_dir / "best_model.pt")

    records = TrialCollector(run_dir=tmp_path, logger=logger_stub).collect()

    assert records[0].checkpoint["best_val_loss"]  == pytest.approx(0.123)
    assert records[0].checkpoint["best_epoch"]     == 7
    assert records[0].checkpoint["n_train_epochs"] == 3
    assert records[0].checkpoint["n_val_epochs"]   == 2


def test_collector_attaches_inference_metrics_and_media(tmp_path, logger_stub):
    (tmp_path / "pipeline").mkdir(parents=True)
    _write_json(tmp_path / "pipeline" / "size_match.json", {})
    _write_json(tmp_path / "pipeline" / "overfit_results.json", [])
    _write_json(tmp_path / "pipeline" / "training_results.json", [])

    (tmp_path / "training" / "unet").mkdir(parents=True)

    inference_dir = tmp_path / "training" / "unet" / "inference" / "run_a"
    (inference_dir / "figures").mkdir(parents=True)
    (inference_dir / "animations").mkdir(parents=True)
    _write_json(inference_dir / "metrics.json", {"curve_rmse_gt": 0.5})
    (inference_dir / "figures" / "profiles_1.png").write_bytes(b"x")
    (inference_dir / "animations" / "elev.gif").write_bytes(b"x")
    (inference_dir / "report.md").write_text("ok", encoding="utf-8")

    records = TrialCollector(run_dir=tmp_path, logger=logger_stub).collect()

    record = records[0]
    assert record.has_inference
    assert record.metrics["curve_rmse_gt"] == 0.5
    assert [p.name for p in record.figures]    == ["profiles_1.png"]
    assert [p.name for p in record.animations] == ["elev.gif"]
    assert record.report_path is not None


def test_seed_collector_aggregates_runs_per_model(tmp_path, logger_stub):
    import numpy as np

    pipe = tmp_path / "pipeline"
    pipe.mkdir(parents=True)
    _write_json(pipe / "size_match.json", {"unet": {"parameters": 100, "overrides": {}}})
    _write_json(pipe / "overfit_results.json", [
        {"model": "unet_seed1", "status": "PASS", "final_loss": 0.1, "converged": True},
        {"model": "unet_seed2", "status": "PASS", "final_loss": 0.3, "converged": True},
    ])
    _write_json(pipe / "training_results.json", [
        {"name": "unet_seed1", "status": "DONE", "duration_s": 10.0},
        {"name": "unet_seed2", "status": "DONE", "duration_s": 20.0},
    ])

    for seed, rmse in ((1, 2.0), (2, 4.0)):
        inference_dir = tmp_path / "training" / f"unet_seed{seed}" / "inference" / "run_a"
        inference_dir.mkdir(parents=True)
        _write_json(inference_dir / "metrics.json", {"curve_rmse_gt": rmse})

    collector = BenchmarkSeedCollector(run_dir=tmp_path, logger=logger_stub)
    records   = collector.collect()

    assert [record.name for record in records] == ["unet"]

    record = records[0]
    assert record.metrics["curve_rmse_gt"]          == 3.0
    assert record.parameters                        == 100
    assert record.overfit["final_loss"]             == pytest.approx(0.2)
    assert record.training_result["duration_s"]     == pytest.approx(15.0)

    dispersion = collector.seed_dispersion["unet"]
    assert dispersion["n_seeds"]                     == 2
    assert dispersion["metrics"]["curve_rmse_gt"]    == pytest.approx(float(np.std([2.0, 4.0], ddof=1)))


def test_overfit_key_strips_loss_component_but_keeps_seed():
    assert TrialCollector._overfit_key("unet__covariance_match_seed3") == "unet_seed3"
    assert TrialCollector._overfit_key("resunet_multihead__mse_curve") == "resunet_multihead"
    assert TrialCollector._overfit_key("unet_seed2")                   == "unet_seed2"
    assert TrialCollector._overfit_key("unet")                         == "unet"


def test_model_of_strips_both_loss_component_and_seed():
    assert TrialCollector._model_of("unet__covariance_match_seed3") == "unet"
    assert TrialCollector._model_of("resunet_multihead__mse_curve") == "resunet_multihead"
    assert TrialCollector._model_of("unet_seed2")                   == "unet"
    assert TrialCollector._model_of("unet")                         == "unet"


def test_size_match_and_overfit_attach_to_every_component(tmp_path, logger_stub):
    pipe = tmp_path / "pipeline"
    pipe.mkdir(parents=True)
    _write_json(pipe / "size_match.json", {"unet": {"parameters": 12345, "overrides": {}}})
    _write_json(pipe / "overfit_results.json", [
        {"model": "unet", "status": "PASS", "final_loss": 1e-4, "converged": True},
    ])
    _write_json(pipe / "training_results.json", [])

    for component in ("param_l1", "covariance_match"):
        (tmp_path / "training" / f"unet__{component}").mkdir(parents=True)

    records = TrialCollector(run_dir=tmp_path, logger=logger_stub).collect()

    assert {record.name for record in records} == {"unet__param_l1", "unet__covariance_match"}
    for record in records:
        assert record.overfit["final_loss"]      == pytest.approx(1e-4)
        assert record.size_match["parameters"]   == 12345
        assert record.parameters                 == 12345


def test_seed_collector_single_run_is_identity(tmp_path, logger_stub):
    pipe = tmp_path / "pipeline"
    pipe.mkdir(parents=True)
    _write_json(pipe / "size_match.json", {})
    _write_json(pipe / "overfit_results.json", [])
    _write_json(pipe / "training_results.json", [])
    (tmp_path / "training" / "unet").mkdir(parents=True)

    collector = BenchmarkSeedCollector(run_dir=tmp_path, logger=logger_stub)
    records   = collector.collect()

    assert [record.name for record in records] == ["unet"]
    assert collector.seed_dispersion           == {}


def test_metric_table_annotates_seed_dispersion(tmp_path, logger_stub):
    records = _records_with_metrics(tmp_path)
    report  = ComparisonReport(
        records         = records,
        out_dir         = tmp_path,
        reference_model = "unet",
        embed_images    = False,
        logger          = logger_stub,
        seed_dispersion = {"unet": {"n_seeds": 2, "best_val_loss_std": 0.01, "metrics": {"overall_r2_gt": 0.05}}},
    )

    assert report.has_seed_sweep is True

    lines = "\n".join(report._metric_table(["overall_r2_gt"], records))

    assert "± " in lines


def test_summary_json_includes_seed_dispersion(tmp_path, logger_stub):
    records = _records_with_metrics(tmp_path)
    report  = ComparisonReport(
        records         = records,
        out_dir         = tmp_path / "out",
        reference_model = "unet",
        embed_images    = False,
        logger          = logger_stub,
        seed_dispersion = {"unet": {"n_seeds": 2, "best_val_loss_std": None, "metrics": {}}},
    )

    report.write_all()

    payload = json.loads((report.out_dir / "comparison_summary.json").read_text())
    by_name = {entry["name"]: entry for entry in payload}

    assert by_name["unet"]["seed_dispersion"]["n_seeds"] == 2
    assert "seed_dispersion" not in by_name["resunet"]


def _records_with_metrics(out_dir):
    a = TrialRecord(name="unet", run_dir=out_dir / "unet")
    b = TrialRecord(name="resunet", run_dir=out_dir / "resunet")

    a.inference_dir = out_dir / "unet" / "inf"
    b.inference_dir = out_dir / "resunet" / "inf"

    a.metrics = {"curve_rmse_gt": 0.1, "overall_r2_gt": 0.9, "psnr_db_gt": 40.0}
    b.metrics = {"curve_rmse_gt": 0.5, "overall_r2_gt": 0.5, "psnr_db_gt": 20.0}

    return [a, b]


def _report(records, out_dir, logger_stub, rank=True):
    return ComparisonReport(
        records         = records,
        out_dir         = out_dir,
        reference_model = "unet",
        embed_images    = False,
        logger          = logger_stub,
        rank_models     = rank,
    )


def test_leaderboard_ranks_better_model_first(tmp_path, logger_stub):
    records = _records_with_metrics(tmp_path)
    report  = _report(records, tmp_path, logger_stub)

    lines = "\n".join(report._leaderboard())

    unet_pos    = lines.index("`unet`")
    resunet_pos = lines.index("`resunet`")
    assert unet_pos < resunet_pos


def test_leaderboard_dominant_model_has_lower_mean_rank(tmp_path, logger_stub):
    records = _records_with_metrics(tmp_path)
    report  = _report(records, tmp_path, logger_stub)

    lines = [line for line in report._leaderboard() if line.startswith("| ")]
    data  = [line for line in lines if "`unet`" in line or "`resunet`" in line]

    unet_rank    = float([line for line in data if "`unet`" in line][0].split("|")[4])
    resunet_rank = float([line for line in data if "`resunet`" in line][0].split("|")[4])

    assert unet_rank < resunet_rank


def test_leaderboard_empty_without_metrics(tmp_path, logger_stub):
    records = [TrialRecord(name="unet", run_dir=tmp_path / "unet")]
    report  = _report(records, tmp_path, logger_stub)

    lines = report._leaderboard()

    assert any("No inference metrics" in line for line in lines)


def test_metric_table_bolds_best_value(tmp_path, logger_stub):
    records = _records_with_metrics(tmp_path)
    report  = _report(records, tmp_path, logger_stub)

    lines = "\n".join(report._metric_table(["overall_r2_gt"], records))

    assert "**0.9**" in lines


def test_metric_table_does_not_bold_in_fold_mode(tmp_path, logger_stub):
    records = _records_with_metrics(tmp_path)
    report  = _report(records, tmp_path, logger_stub, rank=False)

    lines = "\n".join(report._metric_table(["overall_r2_gt"], records))

    assert "**" not in lines


def test_capacity_table_renders_defaults_marker(tmp_path, logger_stub):
    record = TrialRecord(name="unet", run_dir=tmp_path / "unet", parameters=1000)
    report = _report([record], tmp_path, logger_stub)

    lines = "\n".join(report._capacity_table())

    assert "1,000" in lines
    assert "_(defaults)_" in lines


def test_write_all_round_trips_summary_json(tmp_path, logger_stub):
    records = _records_with_metrics(tmp_path)
    out_dir = tmp_path / "comparison"
    report  = _report(records, out_dir, logger_stub)

    written = report.write_all()

    summary_path = out_dir / "comparison_summary.json"
    assert summary_path in written

    payload = FileIO.load_json(summary_path)
    names   = {entry["name"] for entry in payload}
    assert names == {"unet", "resunet"}
    assert (out_dir / "benchmark_overview.md").exists()
    assert (out_dir / "metrics_comparison.md").exists()
