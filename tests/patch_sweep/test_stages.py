from __future__ import annotations

from pathlib import Path

import torch

from configuration.patch_sweep import PatchSweepConfig
from pipelines.patch_sweep.planner import PatchSweepPlanner
from pipelines.patch_sweep.stages import SweepTrainingStage
from tools.monitoring.logger import Logger


def make_logger(tmp_path: Path) -> Logger:
    return Logger(log_dir=str(tmp_path / "logs"), name="stages_test")


def make_stage(tmp_path: Path, resume: bool = False) -> SweepTrainingStage:
    config                    = PatchSweepConfig(track_counts=[5, 9])
    config.patch.maximum      = 64
    config.paths.log_base_dir = tmp_path
    config.resume             = resume

    candidates = [f"FL01_PS{i:02d}" for i in range(3, 31)]
    planner    = PatchSweepPlanner(config, candidates)

    return SweepTrainingStage(config=config, entry_script=Path("e.py"), run_tag="rt", planner=planner, logger=make_logger(tmp_path))


def test_stage_builds_one_item_per_unit(tmp_path):
    stage = make_stage(tmp_path)

    assert len(stage.items) == 16
    assert stage.items[0] == "n05-p008"


def test_job_carries_the_unit_and_run_context(tmp_path):
    stage = make_stage(tmp_path)
    job   = stage._job("n05-p048")

    assert "--worker" in job.command and "train" in job.command
    assert "--unit"   in job.command and "n05-p048" in job.command
    assert "--run-tag" in job.command and "rt" in job.command


def test_resume_reuses_units_with_a_checkpoint(tmp_path):
    stage = make_stage(tmp_path, resume=True)

    checkpoint_dir = stage.stage_dir / "n05-p016" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    torch.save({}, checkpoint_dir / "best_model.pt")

    assert stage._has_checkpoint("n05-p016")
    assert not stage._has_checkpoint("n05-p032")


def test_resume_off_ignores_existing_checkpoints(tmp_path):
    stage = make_stage(tmp_path, resume=False)

    checkpoint_dir = stage.stage_dir / "n05-p016" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    torch.save({}, checkpoint_dir / "best_model.pt")

    assert not stage._has_checkpoint("n05-p016")
