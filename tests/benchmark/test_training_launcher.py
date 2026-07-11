from __future__ import annotations

from pathlib import Path

import pipelines.backbone.training.launcher as backbone_pipeline
from configuration.training import BackboneEntryConfig, default_curriculum
from pipelines.shared.training import training_launcher as mod


def test_seed_sweep_launcher_runs_runner_over_resolved_config(monkeypatch):
    captured = {}
    resolved = object()

    class FakeCli:
        def __init__(self, config, description):
            captured["description"] = description
        def apply(self, argv):
            captured["argv"] = argv
            return resolved

    class FakeSweep:
        def __init__(self, config, runner_class):
            captured["sweep"] = (config, runner_class)
        def run(self):
            captured["ran"] = True

    runner = object()

    monkeypatch.setattr(mod, "ConfigCli", FakeCli)
    monkeypatch.setattr(mod, "SeedSweepRunner", FakeSweep)

    mod.SeedSweepLauncher(object(), runner, "desc").run([])

    assert captured["description"] == "desc"
    assert captured["sweep"]       == (resolved, runner)
    assert captured["ran"]         is True


def test_backbone_launcher_trial_runs_single_runner(monkeypatch):
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

    class FakeCli:
        def __init__(self, config, description):
            self.overrides = {}
        def apply(self, argv):
            return BackboneEntryConfig()

    monkeypatch.setattr(backbone_pipeline, "SingleTrainRunner", FakeSingleRunner)
    monkeypatch.setattr(backbone_pipeline, "TrainScheduler", FakeScheduler)
    monkeypatch.setattr(backbone_pipeline, "ConfigCli", FakeCli)

    backbone_pipeline.BackboneTrainingLauncher(entry_script=Path("/entry/train_backbone.py")).run(["--trial"])

    assert ran.get("ran") is True
    assert "scheduler" not in ran


def test_backbone_launcher_fans_out_when_trials_enabled(monkeypatch):
    ran = {}

    class FakeSingleRunner:
        def __init__(self, cfg):
            ran["single"] = True
        def run(self):
            ran["single_ran"] = True

    class FakeScheduler:
        def __init__(self, **kwargs):
            ran["entry_script"] = kwargs["entry_script"]
        def run(self):
            ran["scheduler_ran"] = True

    class FakeConfig:
        trials_enabled = True
        curriculum     = default_curriculum()

    class FakeCli:
        def __init__(self, config, description):
            self.overrides = {}
        def apply(self, argv):
            return FakeConfig()

    monkeypatch.setattr(backbone_pipeline, "SingleTrainRunner", FakeSingleRunner)
    monkeypatch.setattr(backbone_pipeline, "TrainScheduler", FakeScheduler)
    monkeypatch.setattr(backbone_pipeline, "ConfigCli", FakeCli)

    entry = Path("/entry/train_backbone.py")
    backbone_pipeline.BackboneTrainingLauncher(entry_script=entry).run([])

    assert ran.get("scheduler_ran") is True
    assert ran["entry_script"] == entry
    assert "single_ran" not in ran


def test_ablation_scheduler_houses_runs_in_ablation_dir(tmp_path):
    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "ablation"

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    assert scheduler.runs_root == tmp_path / "ablation"
    assert scheduler.log_dir   == tmp_path / "ablation" / "batch_train_logs"

    job = scheduler._job("model_abl-0-full", {"curriculum.enabled": True})

    assert job.command[-2:]    == ["--logdir", str(tmp_path / "ablation")]
    assert job.log_path        == tmp_path / "ablation" / "batch_train_logs" / "model_abl-0-full.log"


def test_scheduler_houses_each_mode_in_its_own_dir(tmp_path):
    for mode in ("curriculum", "warmup", "presence", "physics", "secondary", "patch", "input", "ablation"):
        config             = BackboneEntryConfig()
        config.logdir      = tmp_path
        config.trials_mode = mode

        scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

        assert scheduler.runs_root == tmp_path / mode
        assert scheduler.log_dir   == tmp_path / mode / "batch_train_logs"

        job = scheduler._job("model_trial", {})
        assert job.command[-2:] == ["--logdir", str(tmp_path / mode)]
