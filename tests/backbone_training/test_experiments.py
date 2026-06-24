from __future__ import annotations

import pytest

from configuration.training.backbone        import PatchTrialsConfig, SecondaryTrialsConfig, _default_input_trials, _default_presence_trials
from configuration.training.general.ablation import AblationCatalog
from pipelines.backbone.training.experiments import (
    AblationTrialPlanner,
    CurriculumTrialPlanner,
    InputTrialPlanner,
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


def test_input_planner_default_variant_drops_interferograms():
    planner = InputTrialPlanner("resunet", _default_input_trials(), CANDIDATES)

    plans = planner.plan()

    assert len(plans) == 1

    name, overrides = plans[0]
    assert name == "resunet_in-amp-allsec-noifg"
    assert overrides["input.use_primary"]        is True
    assert overrides["input.use_secondaries"]    is True
    assert overrides["input.use_interferograms"] is False
    assert overrides["paths.secondary_labels"]   == tuple(CANDIDATES)


def test_input_planner_uses_all_tracks_per_variant():
    trials  = {"a": {"use_interferograms": False}, "b": {"use_dem": True}}
    planner = InputTrialPlanner("unet", trials, CANDIDATES)

    plans = dict(planner.plan())

    assert set(plans) == {"unet_in-a", "unet_in-b"}
    assert all(overrides["paths.secondary_labels"] == tuple(CANDIDATES) for overrides in plans.values())
    assert planner.summary() == {"Input variants": 2, "Tracks": f"all ({len(CANDIDATES)} secondaries)"}


def test_input_planner_rejects_unknown_keys():
    with pytest.raises(ValueError):
        InputTrialPlanner("resunet", {"bad": {"use_phase": True}}, CANDIDATES)


def test_input_planner_rejects_empty_trials():
    with pytest.raises(ValueError):
        InputTrialPlanner("resunet", {}, CANDIDATES)


ABL_FEATURES = [
    {"label": "presence", "enable": {"predict_presence": True, "curriculum.warmup.use_presence_bce": True}, "degrade": {"predict_presence": False, "curriculum.warmup.use_presence_bce": False}},
    {"label": "focal",    "enable": {"curriculum.warmup.amp_focal_gamma": 2.0},                                "degrade": {"curriculum.warmup.amp_focal_gamma": 0.0}},
    {"label": "balance",  "enable": {"curriculum.warmup.presence_balance": True},                            "degrade": {"curriculum.warmup.presence_balance": False}},
]


def test_ablation_planner_cumulative_full_to_baseline():
    planner = AblationTrialPlanner("resunet", ABL_FEATURES, include_full=True)

    plans = planner.plan()
    names = [name for name, _ in plans]

    assert names == [
        "resunet_abl-0-full",
        "resunet_abl-1-no_presence",
        "resunet_abl-2-no_focal",
        "resunet_abl-3-baseline",
    ]
    assert planner.summary()["Total runs"] == 4

    full = dict(plans)["resunet_abl-0-full"]
    assert full["predict_presence"]                       is True
    assert full["curriculum.warmup.amp_focal_gamma"]      == 2.0
    assert full["curriculum.warmup.presence_balance"]     is True

    step1 = dict(plans)["resunet_abl-1-no_presence"]
    assert step1["predict_presence"]                      is False
    assert step1["curriculum.warmup.use_presence_bce"]    is False
    assert step1["curriculum.warmup.amp_focal_gamma"]     == 2.0
    assert step1["curriculum.warmup.presence_balance"]    is True

    baseline = dict(plans)["resunet_abl-3-baseline"]
    assert baseline["predict_presence"]                   is False
    assert baseline["curriculum.warmup.amp_focal_gamma"]  == 0.0
    assert baseline["curriculum.warmup.presence_balance"] is False


def test_ablation_planner_without_full_run():
    planner = AblationTrialPlanner("unet", ABL_FEATURES, include_full=False)

    plans = planner.plan()

    assert [name for name, _ in plans] == [
        "unet_abl-1-no_presence",
        "unet_abl-2-no_focal",
        "unet_abl-3-baseline",
    ]
    assert planner.summary()["Total runs"] == 3


def test_ablation_planner_rejects_empty_features():
    with pytest.raises(ValueError):
        AblationTrialPlanner("resunet", [], include_full=True)


def test_ablation_planner_rejects_feature_without_degrade():
    with pytest.raises(ValueError):
        AblationTrialPlanner("resunet", [{"label": "x"}], include_full=True)


def test_ablation_catalog_default_is_the_standard_set():
    features = AblationCatalog.default_features()
    labels   = [feature["label"] for feature in features]

    assert labels == [
        "out_amp", "out_mu", "out_sigma", "pass_mag", "ifg_phase",
        "output_clamp", "augmentation", "curriculum", "warmup_loss",
        "physics_loss", "class_imbalance",
    ]
    for feature in features:
        assert set(feature) >= {"label", "enable", "degrade"}


def test_ablation_catalog_standard_categories_present():
    catalog = AblationCatalog.as_dict()

    assert catalog["out_amp"]["degrade"]["normalization.out_amp"]     == "zscore"
    assert catalog["augmentation"]["degrade"]["augmentation.p_noise"] == 0.0

    physics = catalog["physics_loss"]
    assert physics["enable"]["curriculum.warmup.use_total_power"]  is True
    assert physics["degrade"]["curriculum.warmup.use_moments"]     is False

    imbalance = catalog["class_imbalance"]
    assert imbalance["enable"]["curriculum.warmup.presence_balance"]  is True
    assert imbalance["degrade"]["curriculum.warmup.use_active_normalization"] is False

    warmup = catalog["warmup_loss"]
    assert warmup["enable"]["curriculum.warmup.use_mse_curve"]   is True
    assert warmup["degrade"]["curriculum.warmup.use_mse_curve"]  is False


def test_ablation_catalog_as_dict_covers_loss_terms():
    catalog = AblationCatalog.as_dict()

    assert "curve_loss_mse_to_l1" in catalog
    assert "spectral_coherence"   in catalog

    swap = catalog["curve_loss_mse_to_l1"]
    assert swap["enable"]["curriculum.warmup.use_mse_curve"]  is True
    assert swap["degrade"]["curriculum.warmup.use_l1_curve"]  is True


def test_ablation_catalog_covers_normalization_and_clamp():
    catalog = AblationCatalog.as_dict()

    norm = catalog["normalization"]
    assert norm["enable"]["normalization.input_strategy"]   == "per_slot"
    assert norm["degrade"]["normalization.input_strategy"]  == "zscore"
    assert norm["degrade"]["normalization.output_strategy"] == "zscore"

    clamp = catalog["output_clamp"]
    assert clamp["enable"]["normalization.clamp_output"]  is True
    assert clamp["degrade"]["normalization.clamp_output"] is False


@pytest.mark.real_data
def test_input_from_dataset_uses_full_stack(test_data_dir):
    from configuration.sar.geometry_config import GeometryConfig

    planner = InputTrialPlanner.from_dataset("resunet", _default_input_trials(), GeometryConfig(), test_data_dir)

    plans = planner.plan()

    assert len(plans) == 1
    name, overrides = plans[0]
    assert name == "resunet_in-amp-allsec-noifg"
    assert overrides["input.use_interferograms"] is False
    assert len(overrides["paths.secondary_labels"]) == len(planner.candidates) >= 1


@pytest.mark.real_data
def test_secondary_from_dataset_loads_candidates(test_data_dir):
    from configuration.sar.geometry_config import GeometryConfig

    trials  = SecondaryTrialsConfig(strategy="consecutive", n_secondaries=2, block_step=1)
    planner = SecondaryTrialPlanner.from_dataset("resunet", trials, GeometryConfig(), test_data_dir)

    plans = planner.plan()

    assert len(plans) >= 1
    assert all("paths.secondary_labels" in overrides for _, overrides in plans)
