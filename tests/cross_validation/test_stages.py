from __future__ import annotations

from pathlib import Path

import pytest

from configuration.cross_validation import CrossValidationConfig, FoldConfig
from pipelines.cross_validation.folds  import FoldPlanner
from pipelines.cross_validation.stages import (
    CrossValidationReportStage,
    FoldInferenceStage,
    FoldTrainingStage,
)
import pipelines.cross_validation.stages as stages_module
from tools.monitoring.logger import Logger


def make_logger(tmp_path: Path) -> Logger:
    return Logger(log_dir=str(tmp_path / "logs"), name="stages_test")


def stage_config(tmp_path: Path, resume: bool = False) -> CrossValidationConfig:
    config                    = CrossValidationConfig()
    config.paths.log_base_dir = tmp_path
    config.folds              = FoldConfig(n_folds=5, azimuth_start=1000, azimuth_end=2000)
    config.resume             = resume
    config.inference_splits   = ["val", "test"]
    config.seeds              = []
    return config


def make_planner(config: CrossValidationConfig) -> FoldPlanner:
    return FoldPlanner(config, range_start=500, range_end=1000)


def queue_result(name: str, status: str = "DONE") -> dict:
    return {"name": name, "gpu": 0, "status": status, "returncode": 0, "duration_s": 1.0, "log_file": ""}


def install_fake_queue(stage, status: str = "DONE") -> dict:
    captured = {}

    def fake_queue(jobs):
        captured["jobs"] = jobs
        return [queue_result(job.name, status) for job in jobs]

    stage._run_queue = fake_queue
    return captured


def test_training_stage_builds_one_item_per_fold(tmp_path):
    stage = FoldTrainingStage(config=stage_config(tmp_path), entry_script=Path("e.py"), run_tag="rt", logger=make_logger(tmp_path))

    assert stage.items == [f"fold_{i}" for i in range(5)]


def test_training_stage_job_carries_fold_index(tmp_path):
    stage = FoldTrainingStage(config=stage_config(tmp_path), entry_script=Path("e.py"), run_tag="rt", logger=make_logger(tmp_path))

    job = stage._job("fold_3")

    assert job.command[job.command.index("--worker") + 1] == "train"
    assert job.command[job.command.index("--fold")  + 1] == "3"
    assert "--seed" not in job.command


def test_training_stage_seed_sweep_expands_fold_by_seed(tmp_path):
    config       = stage_config(tmp_path)
    config.seeds = [1, 2]
    stage        = FoldTrainingStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_logger(tmp_path))

    assert stage.items == [f"fold_{i}/seed{s}" for i in range(5) for s in (1, 2)]

    job = stage._job("fold_3/seed2")
    assert job.command[job.command.index("--fold") + 1] == "3"
    assert job.command[job.command.index("--seed") + 1] == "2"
    assert job.log_path == stage.stage_dir / "fold_3/seed2" / stage.worker_logname


def test_training_stage_subdir_and_results_path(tmp_path):
    stage = FoldTrainingStage(config=stage_config(tmp_path), entry_script=Path("e.py"), run_tag="rt", logger=make_logger(tmp_path))

    assert stage.stage_dir.name            == "folds"
    assert stage.results_path.name         == "training_results.json"
    assert stage.results_path.parent.name  == "pipeline"


def test_training_stage_run_executes_all_folds(tmp_path):
    stage    = FoldTrainingStage(config=stage_config(tmp_path), entry_script=Path("e.py"), run_tag="rt", logger=make_logger(tmp_path))
    captured = install_fake_queue(stage)

    results = stage.run()

    assert [job.name for job in captured["jobs"]] == [f"fold_{i}" for i in range(5)]
    assert [r["name"] for r in results]           == [f"fold_{i}" for i in range(5)]
    assert all(r["status"] == "DONE" for r in results)
    assert stage.results_path.exists()


def test_training_stage_job_command_carries_fold_flag(tmp_path):
    stage    = FoldTrainingStage(config=stage_config(tmp_path), entry_script=Path("e.py"), run_tag="rt", logger=make_logger(tmp_path))
    captured = install_fake_queue(stage)

    stage.run()

    command = captured["jobs"][3].command
    assert "--fold" in command
    assert command[command.index("--fold") + 1] == "3"


def test_inference_stage_one_job_per_fold_split_with_checkpoint(tmp_path):
    config  = stage_config(tmp_path)
    planner = make_planner(config)
    stage   = FoldInferenceStage(config=config, entry_script=Path("e.py"), run_tag="rt", planner=planner, logger=make_logger(tmp_path))

    for name in ("fold_0", "fold_1"):
        (stage.stage_dir / name).mkdir(parents=True)
        (stage.stage_dir / name / "best_model.pt").write_text("x")

    captured = install_fake_queue(stage)
    results  = stage.run()

    assert sorted(job.name for job in captured["jobs"]) == ["fold_0:test", "fold_0:val", "fold_1:test", "fold_1:val"]

    statuses = {r["name"]: r["status"] for r in results}
    assert statuses["fold_0:val"]  == "DONE"
    assert statuses["fold_2:test"] == "SKIPPED"


def test_inference_stage_skips_folds_without_checkpoint(tmp_path):
    config  = stage_config(tmp_path)
    planner = make_planner(config)
    stage   = FoldInferenceStage(config=config, entry_script=Path("e.py"), run_tag="rt", planner=planner, logger=make_logger(tmp_path))

    captured = install_fake_queue(stage)
    results  = stage.run()

    assert captured.get("jobs", []) == []
    assert all(r["status"] == "SKIPPED" for r in results)
    assert len(results)             == 5 * 2


def test_inference_stage_reuses_existing_inference_on_resume(tmp_path):
    config  = stage_config(tmp_path, resume=True)
    planner = make_planner(config)
    stage   = FoldInferenceStage(config=config, entry_script=Path("e.py"), run_tag="rt", planner=planner, logger=make_logger(tmp_path))

    fold_dir = stage.stage_dir / "fold_0"
    (fold_dir).mkdir(parents=True)
    (fold_dir / "best_model.pt").write_text("x")

    for split in ("val", "test"):
        metrics_dir = fold_dir / "inference" / split
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "metrics.json").write_text("{}")

    captured = install_fake_queue(stage)
    results  = stage.run()

    statuses = {r["name"]: r["status"] for r in results}
    assert any(job.name.startswith("fold_0") for job in captured.get("jobs", [])) is False
    assert statuses["fold_0:val"]  == "DONE"
    assert statuses["fold_0:test"] == "DONE"


def test_inference_stage_job_command_carries_split(tmp_path):
    config  = stage_config(tmp_path)
    planner = make_planner(config)
    stage   = FoldInferenceStage(config=config, entry_script=Path("e.py"), run_tag="rt", planner=planner, logger=make_logger(tmp_path))

    (stage.stage_dir / "fold_0").mkdir(parents=True)
    (stage.stage_dir / "fold_0" / "best_model.pt").write_text("x")

    captured = install_fake_queue(stage)
    stage.run()

    job     = next(job for job in captured["jobs"] if job.name == "fold_0:test")
    command = job.command
    assert "--split" in command
    assert command[command.index("--split") + 1] == "test"
    assert "--fold"  in command


def test_report_stage_invokes_collector_and_report(tmp_path, monkeypatch):
    config  = stage_config(tmp_path)
    planner = make_planner(config)
    stage   = CrossValidationReportStage(config=config, run_tag="rt", planner=planner, logger=make_logger(tmp_path))

    seen = {}

    class FakeCollector:
        def __init__(self, run_dir, splits, logger):
            seen["splits"]       = splits
            self.seed_dispersion = {}

        def collect_by_split(self):
            return ["base"], {"test": ["rec"]}

    class FakeReport:
        def __init__(self, base_records, records_by_split, planner, out_dir, model_name, embed_images, logger, seed_dispersion=None):
            seen["base_records"]     = base_records
            seen["records_by_split"] = records_by_split
            seen["model_name"]       = model_name
            seen["out_dir"]          = out_dir
            seen["seed_dispersion"]  = seed_dispersion

        def write_all(self):
            seen["wrote"] = True
            return [seen["out_dir"] / "cv_aggregate_report.md"]

    monkeypatch.setattr(stages_module, "FoldCollector",        FakeCollector, raising=True)
    monkeypatch.setattr(stages_module, "CrossValidationReport", FakeReport,    raising=True)

    out_dir = stage.run()

    assert seen["splits"]           == ["val", "test"]
    assert seen["base_records"]     == ["base"]
    assert seen["records_by_split"] == {"test": ["rec"]}
    assert seen["model_name"]       == config.backbone_name
    assert seen["wrote"]            is True
    assert out_dir                  == seen["out_dir"]


def test_report_stage_model_name_for_profile_autoencoder(tmp_path, monkeypatch):
    config               = stage_config(tmp_path)
    config.training_type = "profile_autoencoder"
    planner              = make_planner(config)
    stage                = CrossValidationReportStage(config=config, run_tag="rt", planner=planner, logger=make_logger(tmp_path))

    seen = {}

    class FakeCollector:
        def __init__(self, run_dir, splits, logger):
            seen["splits"]       = splits
            self.seed_dispersion = {}

        def collect_by_split(self):
            return [], {}

    class FakeReport:
        def __init__(self, base_records, records_by_split, planner, out_dir, model_name, embed_images, logger, seed_dispersion=None):
            seen["model_name"] = model_name

        def write_all(self):
            return []

    monkeypatch.setattr(stages_module, "FoldCollector",        FakeCollector, raising=True)
    monkeypatch.setattr(stages_module, "CrossValidationReport", FakeReport,    raising=True)

    stage.run()

    assert seen["splits"]     == []
    assert seen["model_name"] == "profile_ae"
