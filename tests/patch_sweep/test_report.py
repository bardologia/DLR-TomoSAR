from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from configuration.patch_sweep import PatchSweepConfig
from pipelines.patch_sweep.planner import PatchSweepPlanner
from pipelines.patch_sweep.report import PatchSweepReport, SweepCollector
from tools.monitoring.logger import Logger


def make_planner(track_counts: list[int], maximum: int = 64) -> PatchSweepPlanner:
    config               = PatchSweepConfig(track_counts=track_counts)
    config.patch.maximum = maximum
    candidates           = [f"FL01_PS{i:02d}" for i in range(3, 31)]
    return PatchSweepPlanner(config, candidates)


def synthetic_loss(planner: PatchSweepPlanner, unit) -> float:
    optimum = planner.predicted_optimum(unit.track_count)
    return 0.1 + 0.02 * ((unit.patch_size - optimum) / 32) ** 2


def populate_runs(root: Path, planner: PatchSweepPlanner, drop_metrics: set[str] = frozenset()) -> None:
    results = []
    for unit in planner.units():
        run_dir = root / "training" / unit.name

        if unit.name not in drop_metrics:
            meta = run_dir / "meta"
            meta.mkdir(parents=True, exist_ok=True)
            (meta / "test_metrics.json").write_text(json.dumps({"avg_loss": synthetic_loss(planner, unit), "num_batches": 4}))

        checkpoints = run_dir / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        torch.save({"best_val_loss": synthetic_loss(planner, unit) * 1.05, "best_epoch": 12}, checkpoints / "best_model.pt")

        results.append({"name": unit.name, "status": "DONE", "duration_s": 60.0, "gpu": 0, "returncode": 0, "log_file": ""})

    pipeline_dir = root / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    (pipeline_dir / "training_results.json").write_text(json.dumps(results))


@pytest.fixture
def logger(tmp_path):
    log = Logger(log_dir=str(tmp_path / "logs"), name="patch_sweep_test")
    yield log
    log.close()


def test_collector_reads_every_unit(tmp_path, logger):
    planner = make_planner([5, 9])
    populate_runs(tmp_path, planner)

    records = SweepCollector(run_dir=tmp_path, planner=planner, logger=logger).collect()

    assert len(records) == len(planner.units())
    assert all(record.complete for record in records)
    assert all(record.best_val_loss is not None for record in records)


def test_collector_requires_training_results(tmp_path, logger):
    planner = make_planner([5])

    with pytest.raises(FileNotFoundError, match="training_results.json"):
        SweepCollector(run_dir=tmp_path, planner=planner, logger=logger).collect()


def test_collector_marks_units_without_metrics_incomplete(tmp_path, logger):
    planner = make_planner([5])
    populate_runs(tmp_path, planner, drop_metrics={"n05-p032"})

    records = SweepCollector(run_dir=tmp_path, planner=planner, logger=logger).collect()
    by_name = {record.unit.name: record for record in records}

    assert not by_name["n05-p032"].complete
    assert by_name["n05-p016"].complete


def test_report_ranks_the_synthetic_optimum(tmp_path, logger):
    planner = make_planner([5, 9], maximum=96)
    populate_runs(tmp_path, planner)

    records = SweepCollector(run_dir=tmp_path, planner=planner, logger=logger).collect()
    out_dir = tmp_path / "report"
    PatchSweepReport(records=records, planner=planner, out_dir=out_dir, logger=logger).write_all()

    payload = json.loads((out_dir / "patch_sweep.json").read_text())

    assert payload["track_counts"]["5"]["best_patch_size"] == 48
    assert payload["track_counts"]["9"]["best_patch_size"] == 32
    assert payload["track_counts"]["5"]["predicted_optimum"] == pytest.approx(planner.predicted_optimum(5))


def test_report_writes_one_curve_per_track_count_and_the_summary_figure(tmp_path, logger):
    planner = make_planner([5, 9])
    populate_runs(tmp_path, planner)

    records = SweepCollector(run_dir=tmp_path, planner=planner, logger=logger).collect()
    out_dir = tmp_path / "report"
    written = PatchSweepReport(records=records, planner=planner, out_dir=out_dir, logger=logger).write_all()

    assert (out_dir / "curves" / "n05.png").exists()
    assert (out_dir / "curves" / "n09.png").exists()
    assert (out_dir / "best_patch_vs_tracks.png").exists()
    assert (out_dir / "report.md").exists()
    assert all(path.exists() for path in written)


def test_report_survives_a_track_count_with_no_metrics(tmp_path, logger):
    planner = make_planner([5, 9])
    populate_runs(tmp_path, planner, drop_metrics={unit.name for unit in planner.units() if unit.track_count == 9})

    records = SweepCollector(run_dir=tmp_path, planner=planner, logger=logger).collect()
    out_dir = tmp_path / "report"
    PatchSweepReport(records=records, planner=planner, out_dir=out_dir, logger=logger).write_all()

    payload = json.loads((out_dir / "patch_sweep.json").read_text())

    assert payload["track_counts"]["9"]["best_patch_size"] is None
    assert payload["track_counts"]["5"]["best_patch_size"] is not None
    assert not (out_dir / "best_patch_vs_tracks.png").exists()


def test_markdown_reports_the_ranking_metric_and_prediction(tmp_path, logger):
    planner = make_planner([5])
    populate_runs(tmp_path, planner)

    records = SweepCollector(run_dir=tmp_path, planner=planner, logger=logger).collect()
    out_dir = tmp_path / "report"
    PatchSweepReport(records=records, planner=planner, out_dir=out_dir, logger=logger).write_all()

    text = (out_dir / "report.md").read_text()

    assert "W* = w sqrt(N/n)" in text
    assert "48.2" in text
    assert "n = 5 tracks" in text
