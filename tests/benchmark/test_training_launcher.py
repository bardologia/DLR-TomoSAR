from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.shared.training_launcher import TrainingLauncher


ENTRY = Path("/entry/train.py")


@pytest.fixture
def launcher():
    return TrainingLauncher(entry_script=ENTRY)


def test_modes_are_the_four_training_types(launcher):
    assert set(launcher.MODES) == {"backbone", "jepa", "profile_autoencoder", "image_autoencoder"}


def test_entry_script_is_stored_as_path():
    launcher = TrainingLauncher(entry_script="train.py")

    assert isinstance(launcher.entry_script, Path)


def test_resolve_mode_returns_default_without_flag(launcher):
    assert launcher._resolve_mode("backbone", []) == "backbone"


def test_resolve_mode_honours_explicit_flag(launcher):
    assert launcher._resolve_mode("backbone", ["--mode", "jepa"]) == "jepa"


def test_resolve_mode_ignores_unrelated_flags(launcher):
    resolved = launcher._resolve_mode("profile_autoencoder", ["--epochs", "5", "--seed", "1"])

    assert resolved == "profile_autoencoder"


def test_resolve_mode_rejects_unknown_mode(launcher):
    with pytest.raises(SystemExit):
        launcher._resolve_mode("backbone", ["--mode", "nonsense"])


def test_run_dispatches_to_resolved_mode(launcher, monkeypatch):
    captured = {}

    for mode in launcher.MODES:
        monkeypatch.setattr(launcher, f"_{mode}", lambda config, argv, mode=mode: captured.update(mode=mode, argv=argv))

    monkeypatch.setattr("sys.argv", ["train.py", "--mode", "jepa", "--seed", "3"])

    launcher.run()

    assert captured["mode"] == "jepa"
    assert "--seed" in captured["argv"]


def test_run_uses_default_mode_when_no_flag(launcher, monkeypatch):
    captured = {}

    for mode in launcher.MODES:
        monkeypatch.setattr(launcher, f"_{mode}", lambda config, argv, mode=mode: captured.update(mode=mode))

    monkeypatch.setattr("sys.argv", ["train.py"])

    launcher.run()

    from configuration.training import TrainEntryConfig
    assert captured["mode"] == TrainEntryConfig().mode


def test_run_passes_matching_subconfig_to_handler(launcher, monkeypatch):
    captured = {}

    for mode in launcher.MODES:
        monkeypatch.setattr(launcher, f"_{mode}", lambda config, argv, mode=mode: captured.update(mode=mode, config=config))

    monkeypatch.setattr("sys.argv", ["train.py", "--mode", "image_autoencoder"])

    launcher.run()

    from configuration.training import TrainEntryConfig
    assert captured["config"].__class__ is type(getattr(TrainEntryConfig(), "image_autoencoder"))


def test_backbone_runs_single_trainer_in_trial_mode(launcher, monkeypatch):
    from configuration.training import TrainEntryConfig

    config = TrainEntryConfig().backbone

    ran = {}

    class FakeSingleRunner:
        def __init__(self, cfg):
            ran["single"] = cfg
        def run(self):
            ran["ran"] = True

    class FakeScheduler:
        def __init__(self, **kwargs):
            ran["scheduler"] = True
        def run(self):
            ran["scheduler_ran"] = True

    import pipelines.backbone.training.pipeline as backbone_pipeline
    monkeypatch.setattr(backbone_pipeline, "SingleTrainRunner", FakeSingleRunner)
    monkeypatch.setattr(backbone_pipeline, "TrainScheduler", FakeScheduler)

    launcher._backbone(config, ["--trial"])

    assert ran.get("ran") is True
    assert "scheduler" not in ran
