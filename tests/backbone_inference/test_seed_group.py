from __future__ import annotations

import json
import math
from pathlib import Path
from types   import SimpleNamespace

import pytest

from configuration.inference.general import InferencePaths
from pipelines.backbone.inference.seed_group import BackboneInferenceScheduler, SeedGroupReport
from pipelines.shared.inference.inference_scheduler import InferenceScheduler


class _SilentLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _inference_config(stamp: str = "stamp") -> SimpleNamespace:
    return SimpleNamespace(output_subdir=stamp, paths=InferencePaths())


def _seed_run(group: Path, seed_name: str, metrics: dict, stamp: str = "stamp") -> Path:
    run_dir = group / seed_name
    out_dir = run_dir / "inference" / stamp
    out_dir.mkdir(parents=True)

    (out_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (out_dir / "report.md").write_text("# per-seed report", encoding="utf-8")

    return run_dir


def test_seed_group_report_requires_at_least_two_runs(tmp_path):
    run = _seed_run(tmp_path / "group", "seed0", {"curve_mse_gt": 1.0})

    with pytest.raises(ValueError):
        SeedGroupReport([run], _inference_config(), _SilentLogger())


def test_seed_group_report_aggregates_mean_and_sample_std(tmp_path):
    group = tmp_path / "group"
    runs  = [
        _seed_run(group, "seed0", {"curve_mse_gt": 1.0, "overall_r2_gt": 0.5, "split": "test"}),
        _seed_run(group, "seed1", {"curve_mse_gt": 3.0, "overall_r2_gt": 0.7, "split": "test"}),
    ]

    SeedGroupReport(runs, _inference_config(), _SilentLogger()).run()

    payload = json.loads((group / "inference" / "stamp" / "metrics.json").read_text(encoding="utf-8"))

    assert payload["n_seeds"] == 2
    assert list(payload["seed_runs"]) == ["seed0", "seed1"]
    assert payload["mean"]["curve_mse_gt"] == pytest.approx(2.0)
    assert payload["std"]["curve_mse_gt"]  == pytest.approx(math.sqrt(2.0))
    assert payload["mean"]["overall_r2_gt"] == pytest.approx(0.6)
    assert "split" not in payload["mean"]


def test_seed_group_report_writes_markdown_with_per_seed_columns_and_links(tmp_path):
    group = tmp_path / "group"
    runs  = [
        _seed_run(group, "seed0", {"curve_mse_gt": 1.0}),
        _seed_run(group, "seed1", {"curve_mse_gt": 3.0}),
    ]

    report_path = SeedGroupReport(runs, _inference_config(), _SilentLogger()).run()

    text = report_path.read_text(encoding="utf-8")

    assert report_path == group / "inference" / "stamp" / "report.md"
    assert "seed0" in text and "seed1" in text
    assert "±" in text
    assert "../../seed0/inference/stamp/report.md" in text
    assert "`curve_mse_gt`" in text


def test_seed_group_report_excludes_per_slice_metrics_from_the_tables(tmp_path):
    group = tmp_path / "group"
    runs  = [
        _seed_run(group, "seed0", {"curve_mse_gt": 1.0, "ssim_gt_elev_3": 0.9}),
        _seed_run(group, "seed1", {"curve_mse_gt": 3.0, "ssim_gt_elev_3": 0.8}),
    ]

    report_path = SeedGroupReport(runs, _inference_config(), _SilentLogger()).run()

    payload = json.loads((group / "inference" / "stamp" / "metrics.json").read_text(encoding="utf-8"))
    text    = report_path.read_text(encoding="utf-8")

    assert payload["mean"]["ssim_gt_elev_3"] == pytest.approx(0.85)
    assert "ssim_gt_elev_3" not in text


def test_seed_group_report_missing_seed_metrics_raises(tmp_path):
    group = tmp_path / "group"
    ok    = _seed_run(group, "seed0", {"curve_mse_gt": 1.0})
    bare  = group / "seed1"
    bare.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        SeedGroupReport([ok, bare], _inference_config(), _SilentLogger()).run()


def _scheduler_config(runs_dir: Path, seed_group: bool, stamp: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        runs_dir        = str(runs_dir),
        run_filter      = [],
        seed_group      = seed_group,
        gpus            = [0],
        poll_interval_s = 1,
        inference       = SimpleNamespace(output_subdir=stamp, paths=InferencePaths()),
    )


def test_scheduler_stamps_a_shared_subdir_only_when_unset(tmp_path):
    stamped = BackboneInferenceScheduler(_scheduler_config(tmp_path, True, None), Path("main/x.py"), "backbone")
    stamped._stamp_group_subdir()
    assert stamped.config.inference.output_subdir

    explicit = BackboneInferenceScheduler(_scheduler_config(tmp_path, True, "fixed"), Path("main/x.py"), "backbone")
    explicit._stamp_group_subdir()
    assert explicit.config.inference.output_subdir == "fixed"


def test_scheduler_writes_group_report_when_all_seeds_done(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    group = tmp_path / "runs" / "group"
    runs  = [
        _seed_run(group, "seed0", {"curve_mse_gt": 1.0}),
        _seed_run(group, "seed1", {"curve_mse_gt": 3.0}),
    ]

    scheduler = BackboneInferenceScheduler(_scheduler_config(tmp_path / "runs", True, "stamp"), Path("main/x.py"), "backbone")

    def fake_run(self):
        self.run_dirs = runs
        return [SimpleNamespace(status="DONE"), SimpleNamespace(status="DONE")]

    monkeypatch.setattr(InferenceScheduler, "run", fake_run)

    scheduler.run()

    assert (group / "inference" / "stamp" / "metrics.json").is_file()
    assert (group / "inference" / "stamp" / "report.md").is_file()


def test_scheduler_skips_group_report_when_a_seed_failed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    group = tmp_path / "runs" / "group"
    runs  = [
        _seed_run(group, "seed0", {"curve_mse_gt": 1.0}),
        _seed_run(group, "seed1", {"curve_mse_gt": 3.0}),
    ]

    scheduler = BackboneInferenceScheduler(_scheduler_config(tmp_path / "runs", True, "stamp"), Path("main/x.py"), "backbone")

    def fake_run(self):
        self.run_dirs = runs
        return [SimpleNamespace(status="DONE"), SimpleNamespace(status="FAILED")]

    monkeypatch.setattr(InferenceScheduler, "run", fake_run)

    scheduler.run()

    assert not (group / "inference" / "stamp" / "metrics.json").exists()
