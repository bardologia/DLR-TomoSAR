from __future__ import annotations

from pathlib import Path

import pytest

import pipelines.dual.training.launcher as dual_launcher
from configuration.training import DualEntryConfig, DualInputTrialsConfig, dual_curriculum
from configuration.training.dual import _default_dual_input_trials
from pipelines.dual.training.experiments import DualInputTrialPlanner
from pipelines.dual.training.pipeline import TrunkChannelMap
from tools.runtime.config_cli import ConfigCli


HALF_FEATURES = [32, 64, 128, 256]


def _planner(trials_config=None, model_overrides=None):
    return DualInputTrialPlanner(trials_config or DualInputTrialsConfig(), model_overrides or {}, TrunkChannelMap.GROUPS)


def test_dual_input_planner_default_catalog_covers_the_seven_ordered_pairs():
    plans = dict(_planner().plan())

    assert list(plans) == ["di-full-full", "di-pass-full", "di-full-pass", "di-ifg-full", "di-full-ifg", "di-pass-ifg", "di-ifg-pass"]

    full   = ["pass", "ifg"]
    passes = ["pass"]
    ifg    = ["ifg"]

    assert plans["di-full-full"]["params_input"]    == full
    assert plans["di-full-full"]["existence_input"] == full
    assert plans["di-pass-full"]["params_input"]    == passes
    assert plans["di-pass-full"]["existence_input"] == full
    assert plans["di-full-pass"]["params_input"]    == full
    assert plans["di-full-pass"]["existence_input"] == passes
    assert plans["di-ifg-full"]["params_input"]     == ifg
    assert plans["di-ifg-full"]["existence_input"]  == full
    assert plans["di-full-ifg"]["params_input"]     == full
    assert plans["di-full-ifg"]["existence_input"]  == ifg
    assert plans["di-pass-ifg"]["params_input"]     == passes
    assert plans["di-pass-ifg"]["existence_input"]  == ifg
    assert plans["di-ifg-pass"]["params_input"]     == ifg
    assert plans["di-ifg-pass"]["existence_input"]  == passes


def test_dual_input_planner_default_catalog_never_selects_the_dem():
    for _, overrides in _planner().plan():
        assert "dem" not in overrides["params_input"]
        assert "dem" not in overrides["existence_input"]


def test_dual_input_planner_fixes_half_width_trunks_on_every_trial():
    for _, overrides in _planner().plan():
        assert overrides["model_overrides"] == {"params_features": HALF_FEATURES, "existence_features": HALF_FEATURES}


def test_dual_input_planner_merges_entry_model_overrides():
    for _, overrides in _planner(model_overrides={"dropout": 0.3}).plan():
        assert overrides["model_overrides"]["dropout"]         == 0.3
        assert overrides["model_overrides"]["params_features"] == HALF_FEATURES


def test_dual_input_planner_rejects_feature_keys_in_model_overrides():
    with pytest.raises(ValueError, match="input_trials fields"):
        _planner(model_overrides={"params_features": [8, 16]})


def test_dual_input_planner_rejects_empty_or_invalid_features():
    with pytest.raises(ValueError, match="at least one feature width"):
        _planner(DualInputTrialsConfig(params_features=[]))

    with pytest.raises(ValueError, match="positive integers"):
        _planner(DualInputTrialsConfig(existence_features=[32, 0]))


def test_dual_input_planner_rejects_empty_trials():
    with pytest.raises(ValueError, match="at least one trunk-input variant"):
        _planner(DualInputTrialsConfig(trials={}))


def test_dual_input_planner_rejects_missing_trunk_key():
    with pytest.raises(ValueError, match="must set \\['existence'\\]"):
        _planner(DualInputTrialsConfig(trials={"bad": {"params": ["pass"]}}))


def test_dual_input_planner_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown keys \\['gate'\\]"):
        _planner(DualInputTrialsConfig(trials={"bad": {"params": ["pass"], "existence": ["ifg"], "gate": ["ifg"]}}))


def test_dual_input_planner_rejects_unknown_groups():
    with pytest.raises(ValueError, match="unknown 'params' groups \\['phase'\\]"):
        _planner(DualInputTrialsConfig(trials={"bad": {"params": ["pass", "phase"], "existence": ["ifg"]}}))


def test_dual_input_planner_rejects_the_dem_group():
    with pytest.raises(ValueError, match="unknown 'params' groups \\['dem'\\]"):
        _planner(DualInputTrialsConfig(trials={"bad": {"params": ["pass", "ifg", "dem"], "existence": ["ifg"]}}))


def test_dual_input_planner_rejects_empty_selection():
    with pytest.raises(ValueError, match="selects no channel groups for 'existence'"):
        _planner(DualInputTrialsConfig(trials={"bad": {"params": ["pass"], "existence": []}}))


def test_dual_input_planner_rejects_duplicate_groups():
    with pytest.raises(ValueError, match="repeats 'params' groups"):
        _planner(DualInputTrialsConfig(trials={"bad": {"params": ["pass", "pass"], "existence": ["ifg"]}}))


def test_dual_input_planner_paths_are_entry_config_leaves():
    config = DualEntryConfig()

    for _, overrides in _planner().plan():
        for path in overrides:
            assert hasattr(config, path), path


def test_dual_trial_overrides_round_trip_through_the_cli():
    _, overrides = _planner().plan()[4]

    config = ConfigCli(DualEntryConfig()).apply(ConfigCli.to_argv(overrides))

    assert config.params_input    == ["pass", "ifg"]
    assert config.existence_input == ["ifg"]
    assert config.model_overrides == {"params_features": HALF_FEATURES, "existence_features": HALF_FEATURES}


def test_dual_default_trials_match_the_config_factory():
    assert DualInputTrialsConfig().trials == _default_dual_input_trials()


def test_dual_scheduler_houses_runs_in_input_dir(tmp_path):
    config        = DualEntryConfig()
    config.logdir = tmp_path

    scheduler = dual_launcher.DualTrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_dual.py"))

    assert scheduler.runs_root == tmp_path / "input"
    assert scheduler.log_dir   == tmp_path / "input" / "batch_train_logs"

    job = scheduler._job("di-full-ifg", {})

    assert job.command[-2:] == ["--logdir", str(tmp_path / "input")]
    assert job.log_path     == tmp_path / "input" / "batch_train_logs" / "di-full-ifg.log"


def test_dual_scheduler_plans_the_trunk_input_grid(tmp_path):
    config        = DualEntryConfig()
    config.logdir = tmp_path

    scheduler = dual_launcher.DualTrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_dual.py"))

    plans = dict(scheduler.planner().plan())

    assert list(plans) == ["di-full-full", "di-pass-full", "di-full-pass", "di-ifg-full", "di-full-ifg", "di-pass-ifg", "di-ifg-pass"]
    assert plans["di-full-ifg"]["model_overrides"]["params_features"] == HALF_FEATURES


def test_dual_scheduler_rejects_unknown_mode(tmp_path):
    config             = DualEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "context"

    with pytest.raises(ValueError, match="Unknown trials_mode 'context'"):
        dual_launcher.DualTrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_dual.py"))


def test_dual_scheduler_forwards_only_non_scheduler_overrides(tmp_path):
    config        = DualEntryConfig()
    config.logdir = tmp_path

    overrides = {
        "trials_enabled"                  : True,
        "trials_mode"                     : "input",
        "input_trials.params_features"    : [16, 32],
        "gpus"                            : [0, 1],
        "poll_interval_s"                 : 1.0,
        "training.max_epochs"             : 5,
        "params_backbone"                 : "unet",
    }

    scheduler = dual_launcher.DualTrainScheduler(config=config, cli_overrides=overrides, entry_script=Path("/entry/train_dual.py"))

    assert scheduler.forward_overrides == {"training.max_epochs": 5, "params_backbone": "unet"}


def test_dual_launcher_trial_flag_runs_single_runner(monkeypatch):
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
            config       = DualEntryConfig()
            config.seeds = [0]
            return config

    monkeypatch.setattr(dual_launcher, "DualSingleTrainRunner", FakeSingleRunner)
    monkeypatch.setattr(dual_launcher, "DualTrainScheduler", FakeScheduler)
    monkeypatch.setattr(dual_launcher, "ConfigCli", FakeCli)

    dual_launcher.DualTrainingLauncher(entry_script=Path("/entry/train_dual.py")).run(["--trial"])

    assert ran.get("ran") is True
    assert "scheduler" not in ran


def test_dual_launcher_fans_multi_seed_runs_across_the_pool(monkeypatch):
    ran = {}

    class FakeSingleRunner:
        def __init__(self, cfg):
            ran["single"] = True
        def run(self):
            ran["single_ran"] = True

    class FakeFanout:
        @classmethod
        def for_runner(cls, config, cli_overrides, entry_script, runner_factory, base_label=None):
            ran["for_runner"] = (entry_script, runner_factory)
            return cls()
        def run(self):
            ran["fanout_ran"] = True

    class FakeCli:
        def __init__(self, config, description):
            self.overrides = {}
        def apply(self, argv):
            return DualEntryConfig()

    monkeypatch.setattr(dual_launcher, "DualSingleTrainRunner", FakeSingleRunner)
    monkeypatch.setattr(dual_launcher, "SeedFanoutScheduler", FakeFanout)
    monkeypatch.setattr(dual_launcher, "ConfigCli", FakeCli)

    entry = Path("/entry/train_dual.py")
    dual_launcher.DualTrainingLauncher(entry_script=entry).run([])

    assert ran.get("fanout_ran") is True
    assert ran["for_runner"]     == (entry, FakeSingleRunner)
    assert "single_ran" not in ran


def test_dual_launcher_fans_out_when_trials_enabled(monkeypatch):
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
        curriculum     = dual_curriculum()

    class FakeCli:
        def __init__(self, config, description):
            self.overrides = {}
        def apply(self, argv):
            return FakeConfig()

    monkeypatch.setattr(dual_launcher, "DualSingleTrainRunner", FakeSingleRunner)
    monkeypatch.setattr(dual_launcher, "DualTrainScheduler", FakeScheduler)
    monkeypatch.setattr(dual_launcher, "ConfigCli", FakeCli)

    entry = Path("/entry/train_dual.py")
    dual_launcher.DualTrainingLauncher(entry_script=entry).run([])

    assert ran.get("scheduler_ran") is True
    assert ran["entry_script"] == entry
    assert "single_ran" not in ran
