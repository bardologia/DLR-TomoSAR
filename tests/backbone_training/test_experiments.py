from __future__ import annotations

import pytest

from configuration.training.backbone        import PatchTrialsConfig, SecondaryTrialsConfig, _default_presence_trials
from pipelines.backbone.training.experiments import (
    CurriculumTrialPlanner,
    PatchSizeTrialPlanner,
    SecondaryTrialPlanner,
    SlotPresenceTrialPlanner,
    WarmupTrialPlanner,
)


CANDIDATES = ["PS04", "PS06", "PS08", "PS10", "PS12", "PS14"]


def test_curriculum_planner_crosses_warmup_and_complete():
    planner = CurriculumTrialPlanner(
        "resunet",
        {"w1": {"use_param_l1": True}, "w2": {"use_l1_curve": True}},
        {"c1": {"use_mse_curve": True}},
    )

    plans = planner.plan()

    assert len(plans) == 2
    assert planner.summary() == {"Warmup losses": 2, "Complete losses": 1}

    run_name, overrides = plans[0]
    assert run_name == "resunet_w-w1_c-c1"
    assert overrides["curriculum.enabled"] is True
    assert overrides["curriculum.warmup.use_param_l1"] is True
    assert overrides["curriculum.complete.use_mse_curve"] is True


def test_warmup_planner_disables_curriculum():
    planner = WarmupTrialPlanner("unet", {"a": {"use_param_l1": True}, "b": {"use_mse_curve": True}})

    plans = planner.plan()

    assert len(plans) == 2
    assert all(overrides["curriculum.enabled"] is False for _, overrides in plans)
    assert plans[0][0] == "unet_nc-a"
    assert plans[0][1]["curriculum.warmup.use_param_l1"] is True


def test_presence_planner_disables_curriculum():
    planner = SlotPresenceTrialPlanner(
        "resunet",
        {"P": {"predict_presence": True, "use_presence_bce": True, "weight_presence_bce": 1.0, "use_active_normalization": True}},
    )

    plans = planner.plan()

    assert len(plans) == 1
    assert [name for name, _ in plans] == ["resunet_pr-P"]
    assert planner.summary()["Total runs"] == 1

    ov = dict(plans)["resunet_pr-P"]

    assert ov["curriculum.enabled"] is False
    assert ov["predict_presence"]  is True
    assert "curriculum.warmup.param_match" not in ov
    assert ov["curriculum.warmup.use_presence_bce"]          is True
    assert ov["curriculum.warmup.weight_presence_bce"]       == 1.0
    assert ov["curriculum.warmup.use_active_normalization"]  is True


def test_presence_planner_default_matrix():
    planner = SlotPresenceTrialPlanner("resunet", _default_presence_trials())

    plans = dict(planner.plan())

    assert len(plans) == 20
    assert "resunet_pr-none" in plans
    assert plans["resunet_pr-none"] == {"curriculum.enabled": False}

    for name, overrides in plans.items():
        if overrides.get("predict_presence"):
            assert overrides["curriculum.warmup.use_presence_bce"] is True
            assert overrides["curriculum.warmup.weight_presence_bce"] > 0.0
        if overrides.get("curriculum.warmup.amp_focal_gamma"):
            assert overrides["curriculum.warmup.amp_focal_gamma"] > 0.0


def test_patch_planner_emits_size_and_stride():
    planner = PatchSizeTrialPlanner("resunet", PatchTrialsConfig(sizes=[32, 64], stride_ratio=0.5))

    plans = planner.plan()

    assert len(plans) == 2

    name, overrides = plans[0]
    assert name == "resunet_p-32"
    assert overrides["training.patch_size"]   == (32, 32)
    assert overrides["training.patch_stride"] == 16

    assert plans[1][1]["training.patch_stride"] == 32


def test_patch_planner_rejects_empty_sizes():
    with pytest.raises(ValueError):
        PatchSizeTrialPlanner("resunet", PatchTrialsConfig(sizes=[], stride_ratio=0.5))


def test_patch_planner_rejects_bad_stride_ratio():
    with pytest.raises(ValueError):
        PatchSizeTrialPlanner("resunet", PatchTrialsConfig(sizes=[32], stride_ratio=1.5))


def test_secondary_consecutive_blocks():
    trials  = SecondaryTrialsConfig(strategy="consecutive", n_secondaries=2, block_step=1)
    planner = SecondaryTrialPlanner("resunet", trials, CANDIDATES)

    plans = planner.plan()

    assert len(plans) == len(CANDIDATES) - 1

    name, overrides = plans[0]
    assert overrides["paths.secondary_labels"] == ("PS04", "PS06")
    assert name.startswith("resunet_sec-consecutive-t00")


def test_secondary_spaced_picks_strided_labels():
    trials  = SecondaryTrialsConfig(strategy="spaced", n_secondaries=2, block_step=1, spacing=2)
    planner = SecondaryTrialPlanner("resunet", trials, CANDIDATES)

    plans = planner.plan()

    assert plans[0][1]["paths.secondary_labels"] == ("PS04", "PS08")


def test_secondary_uniform_produces_distinct_trials():
    trials  = SecondaryTrialsConfig(strategy="uniform", n_secondaries=2, n_trials=4, seed=0)
    planner = SecondaryTrialPlanner("resunet", trials, CANDIDATES)

    plans = planner.plan()

    selections = [overrides["paths.secondary_labels"] for _, overrides in plans]

    assert len(plans) == 4
    assert len(set(selections)) == 4


def test_secondary_gaussian_requires_mean_and_sigma():
    trials = SecondaryTrialsConfig(strategy="gaussian", n_secondaries=2, n_trials=2)

    with pytest.raises(ValueError):
        SecondaryTrialPlanner("resunet", trials, CANDIDATES)


def test_secondary_rejects_unknown_strategy():
    trials = SecondaryTrialsConfig(strategy="bogus", n_secondaries=2)

    with pytest.raises(ValueError):
        SecondaryTrialPlanner("resunet", trials, CANDIDATES)


def test_secondary_rejects_too_many_secondaries():
    trials = SecondaryTrialsConfig(strategy="consecutive", n_secondaries=99)

    with pytest.raises(ValueError):
        SecondaryTrialPlanner("resunet", trials, CANDIDATES)


@pytest.mark.real_data
def test_secondary_from_dataset_loads_candidates(test_data_dir):
    from configuration.sar.geometry_config import GeometryConfig

    trials  = SecondaryTrialsConfig(strategy="consecutive", n_secondaries=2, block_step=1)
    planner = SecondaryTrialPlanner.from_dataset("resunet", trials, GeometryConfig(), test_data_dir)

    plans = planner.plan()

    assert len(plans) >= 1
    assert all("paths.secondary_labels" in overrides for _, overrides in plans)
