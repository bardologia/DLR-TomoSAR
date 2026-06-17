from __future__ import annotations

import pytest

from pipelines.jepa.training.pipeline import TrainingPipeline


def test_resolve_ae_run_returns_none_when_no_run():
    assert TrainingPipeline._resolve_ae_run("/anywhere", None, "profile") is None
    assert TrainingPipeline._resolve_ae_run("/anywhere", "", "profile")   is None


def test_resolve_ae_run_raises_for_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        TrainingPipeline._resolve_ae_run(tmp_path, "does_not_exist", "profile")


def test_resolve_ae_run_returns_existing_directory(tmp_path):
    run_dir = tmp_path / "ae_run_01"
    run_dir.mkdir()

    resolved = TrainingPipeline._resolve_ae_run(tmp_path, "ae_run_01", "profile")

    assert resolved == run_dir


def test_validate_checkpoint_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        TrainingPipeline.validate_checkpoint(tmp_path / "best_model.pt", "profile")


def test_validate_checkpoint_passes_when_present(tmp_path):
    ckpt = tmp_path / "best_model.pt"
    ckpt.write_bytes(b"x")

    TrainingPipeline.validate_checkpoint(ckpt, "profile")


@pytest.mark.slow
@pytest.mark.real_data
def test_training_pipeline_one_epoch_produces_checkpoint():
    pytest.skip(
        "Full JEPA training requires a real pretrained profile-autoencoder run directory "
        "(best_model.pt + meta) and a preprocessing run with SAR artifacts; these are not "
        "reconstructable from the crafted test fixtures, so a 1-epoch end-to-end run is not "
        "exercised here. Coupling, loss, trainer step, and checkpoint round-trip are covered "
        "by the unit tests."
    )
