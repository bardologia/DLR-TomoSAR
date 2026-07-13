from __future__ import annotations

from pathlib import Path

import torch

from configuration.patch_sweep import PatchSweepConfig
from pipelines.patch_sweep.planner import PatchSweepPlanner
from pipelines.patch_sweep.stages import SweepTrainingStage
from tools.monitoring.logger import Logger


def make_logger(tmp_path: Path) -> Logger:
    return Logger(log_dir=str(tmp_path / "logs"), name="stages_test")


def make_stage(tmp_path: Path, resume: bool = False, seeds: list[int] | None = None) -> SweepTrainingStage:
    config                    = PatchSweepConfig()
    config.dataset_paths      = [Path("/data/w20_10"), Path("/data/w20_20")]
    config.patch.maximum      = 64
    config.paths.log_base_dir = tmp_path
    config.resume             = resume
    config.seeds              = seeds or []

    planner = PatchSweepPlanner(config)

    return SweepTrainingStage(config=config, entry_script=Path("e.py"), run_tag="rt", planner=planner, logger=make_logger(tmp_path))


def test_stage_builds_one_item_per_unit(tmp_path):
    stage = make_stage(tmp_path)

    assert len(stage.items) == 16
    assert stage.items[0] == "w20_10-p008"


def test_job_carries_the_unit_and_run_context(tmp_path):
    stage = make_stage(tmp_path)
    job   = stage._job("w20_10-p048")

    assert "--worker" in job.command and "train" in job.command
    assert "--unit"   in job.command and "w20_10-p048" in job.command
    assert "--run-tag" in job.command and "rt" in job.command


def test_resume_reuses_units_with_a_checkpoint(tmp_path):
    stage = make_stage(tmp_path, resume=True)

    checkpoint_dir = stage.stage_dir / "w20_10-p016" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    torch.save({}, checkpoint_dir / "best_model.pt")

    assert stage._has_checkpoint("w20_10-p016")
    assert not stage._has_checkpoint("w20_10-p032")


def test_resume_off_ignores_existing_checkpoints(tmp_path):
    stage = make_stage(tmp_path, resume=False)

    checkpoint_dir = stage.stage_dir / "w20_10-p016" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    torch.save({}, checkpoint_dir / "best_model.pt")

    assert not stage._has_checkpoint("w20_10-p016")


def test_stage_expands_units_by_seed(tmp_path):
    stage = make_stage(tmp_path, seeds=[0, 1])

    assert len(stage.items) == 32
    assert stage.items[0] == "w20_10-p008/seed0"


def test_seeded_job_carries_the_unit_base_and_seed(tmp_path):
    stage = make_stage(tmp_path, seeds=[0, 1])
    job   = stage._job("w20_10-p048/seed1")

    assert "--unit" in job.command and "w20_10-p048" in job.command
    assert "--seed" in job.command and "1" in job.command
    assert "w20_10-p048/seed1" not in job.command


def test_resume_sees_seed_run_checkpoints(tmp_path):
    stage = make_stage(tmp_path, resume=True, seeds=[0, 1])

    checkpoint_dir = stage.stage_dir / "w20_10-p016" / "seed1" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    torch.save({}, checkpoint_dir / "best_model.pt")

    assert stage._has_checkpoint("w20_10-p016/seed1")
    assert not stage._has_checkpoint("w20_10-p016/seed0")
