from __future__ import annotations

import pytest

from configuration.sar.geometry_config import GeometryConfig
from configuration.training import BackboneEntryConfig
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
from tools.runtime.config_cli import ConfigCli


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
        {"AB": {"use_active_normalization": True, "presence_balance": True}},
    )

    plans = planner.plan()

    assert len(plans) == 1
    assert [name for name, _ in plans] == ["resunet_pr-AB"]
    assert planner.summary()["Total runs"] == 1

    ov = dict(plans)["resunet_pr-AB"]

    assert ov["curriculum.enabled"] is False
    assert "curriculum.warmup.param_match" not in ov
    assert ov["curriculum.warmup.use_active_normalization"]  is True
    assert ov["curriculum.warmup.presence_balance"]          is True


def test_presence_planner_default_matrix():
    planner = SlotPresenceTrialPlanner("resunet", _default_presence_trials())

    plans = dict(planner.plan())

    assert len(plans) == 10
    assert "resunet_pr-none" in plans
    assert plans["resunet_pr-none"] == {"curriculum.enabled": False}

    for name, overrides in plans.items():
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
    {"label": "active_norm", "enable": {"curriculum.warmup.use_active_normalization": True}, "degrade": {"curriculum.warmup.use_active_normalization": False}},
    {"label": "focal",       "enable": {"curriculum.warmup.amp_focal_gamma": 2.0},           "degrade": {"curriculum.warmup.amp_focal_gamma": 0.0}},
    {"label": "balance",     "enable": {"curriculum.warmup.presence_balance": True},         "degrade": {"curriculum.warmup.presence_balance": False}},
]


def test_ablation_planner_cumulative_full_to_baseline():
    planner = AblationTrialPlanner("resunet", ABL_FEATURES, include_full=True)

    plans = planner.plan()
    names = [name for name, _ in plans]

    assert names == [
        "resunet_abl-0-full",
        "resunet_abl-1-no_active_norm",
        "resunet_abl-2-no_focal",
        "resunet_abl-3-baseline",
    ]
    assert planner.summary()["Total runs"] == 4

    full = dict(plans)["resunet_abl-0-full"]
    assert full["curriculum.warmup.use_active_normalization"] is True
    assert full["curriculum.warmup.amp_focal_gamma"]      == 2.0
    assert full["curriculum.warmup.presence_balance"]     is True

    step1 = dict(plans)["resunet_abl-1-no_active_norm"]
    assert step1["curriculum.warmup.use_active_normalization"] is False
    assert step1["curriculum.warmup.amp_focal_gamma"]     == 2.0
    assert step1["curriculum.warmup.presence_balance"]    is True

    baseline = dict(plans)["resunet_abl-3-baseline"]
    assert baseline["curriculum.warmup.use_active_normalization"] is False
    assert baseline["curriculum.warmup.amp_focal_gamma"]  == 0.0
    assert baseline["curriculum.warmup.presence_balance"] is False


def test_ablation_planner_without_full_run():
    planner = AblationTrialPlanner("unet", ABL_FEATURES, include_full=False)

    plans = planner.plan()

    assert [name for name, _ in plans] == [
        "unet_abl-1-no_active_norm",
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
        "covariance_match", "physics_curriculum", "coherence_resyn",
        "cosine_curve", "architecture", "augmentation",
        "active_norm", "lr_per_group", "lr_warmup",
        "out_sigma", "out_amp", "ifg_phase", "pass_mag",
        "output_clamp",
    ]
    assert "out_mu" not in labels
    for feature in features:
        assert set(feature) >= {"label", "enable", "degrade"}


def test_ablation_catalog_standard_categories_present():
    catalog = AblationCatalog.as_dict()

    assert catalog["out_amp"]["degrade"]["normalization.out_amp"]      == "zscore"
    assert catalog["pass_mag"]["degrade"]["normalization.pass_mag"]    == "zscore_log1p"
    assert catalog["ifg_phase"]["degrade"]["normalization.ifg_phase"]  == "fixed_div_pi"
    assert catalog["augmentation"]["degrade"]["augmentation.p_flip_h"] == 0.0
    assert "augmentation.p_noise"     not in catalog["augmentation"]["enable"]
    assert "out_mu"      not in catalog
    assert "total_power" not in catalog
    assert "moments"     not in catalog
    assert "capon_cycle" not in catalog

    covariance = catalog["covariance_match"]
    assert covariance["enable"]["curriculum.complete.use_covariance_match"]  is True
    assert covariance["degrade"]["curriculum.complete.use_covariance_match"] is False

    coherence = catalog["coherence_resyn"]
    assert coherence["enable"]["curriculum.complete.use_coherence_resyn"]     is True
    assert coherence["degrade"]["curriculum.complete.weight_coherence_resyn"] == 0.0
    assert coherence["degrade"]["curriculum.warmup.use_coherence_resyn"]      is False

    physics_curriculum = catalog["physics_curriculum"]
    assert physics_curriculum["enable"]["curriculum.enabled"]                        is True
    assert physics_curriculum["degrade"]["curriculum.enabled"]                       is False
    assert physics_curriculum["degrade"]["curriculum.warmup.use_coherence_resyn"]    is True
    assert physics_curriculum["degrade"]["curriculum.warmup.weight_coherence_resyn"] == 0.05

    cosine = catalog["cosine_curve"]
    assert cosine["enable"]["curriculum.warmup.use_cosine_curve"]    is True
    assert cosine["enable"]["curriculum.complete.use_cosine_curve"]  is True
    assert cosine["degrade"]["curriculum.warmup.use_cosine_curve"]   is False
    assert cosine["degrade"]["curriculum.complete.use_cosine_curve"] is False

    imbalance = catalog["class_imbalance"]
    assert imbalance["enable"]["curriculum.warmup.use_active_normalization"]  is True
    assert imbalance["degrade"]["curriculum.warmup.use_active_normalization"] is False
    assert "curriculum.warmup.presence_balance" not in imbalance["enable"]

    assert "predict_presence" not in catalog

    architecture = catalog["architecture"]
    assert architecture["enable"]["backbone_name"]                       == "resunet"
    assert architecture["enable"]["curriculum.warmup.use_param_l1"]      is True
    assert architecture["enable"]["curriculum.warmup.use_param_mse"]     is False
    assert architecture["degrade"]["backbone_name"]                      == "unet"
    assert architecture["degrade"]["curriculum.warmup.use_param_l1"]     is False
    assert architecture["degrade"]["curriculum.warmup.use_param_mse"]    is True
    assert architecture["degrade"]["curriculum.warmup.weight_param_mse"] == 1.0

    active_norm = catalog["active_norm"]
    assert active_norm["enable"]["curriculum.warmup.use_active_normalization"]  is True
    assert active_norm["degrade"]["curriculum.warmup.use_active_normalization"] is False

    assert "warmup_loss" not in catalog
    assert "curriculum"  not in catalog

    lr_warmup = catalog["lr_warmup"]
    assert lr_warmup["enable"]["training.warmup_enabled"]  is True
    assert lr_warmup["degrade"]["training.warmup_enabled"] is False

    lr_per_group = catalog["lr_per_group"]
    assert lr_per_group["enable"]["model_overrides"]["output_head_lr"]  == 1e-3
    assert lr_per_group["enable"]["model_overrides"]["encoder_lr"]      == 3e-4
    assert set(lr_per_group["degrade"]["model_overrides"].values())     == {3e-4}
    assert set(lr_per_group["degrade"]["model_overrides"])              == set(lr_per_group["enable"]["model_overrides"])


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
    planner = InputTrialPlanner.from_dataset("resunet", _default_input_trials(), GeometryConfig(), test_data_dir)

    plans = planner.plan()

    assert len(plans) == 1
    name, overrides = plans[0]
    assert name == "resunet_in-amp-allsec-noifg"
    assert overrides["input.use_interferograms"] is False
    assert len(overrides["paths.secondary_labels"]) == len(planner.candidates) >= 1


@pytest.mark.real_data
def test_secondary_from_dataset_loads_candidates(test_data_dir):
    trials  = SecondaryTrialsConfig(strategy="consecutive", n_secondaries=2, block_step=1)
    planner = SecondaryTrialPlanner.from_dataset("resunet", trials, GeometryConfig(), test_data_dir)

    plans = planner.plan()

    assert len(plans) >= 1
    assert all("paths.secondary_labels" in overrides for _, overrides in plans)


def test_ablation_catalog_paths_are_entry_config_leaves():
    leaves  = {path for path, _ in ConfigCli._leaves(BackboneEntryConfig())}
    unknown = [
        (feature["label"], side, path)
        for feature in AblationCatalog.features()
        for side in ("enable", "degrade")
        for path in feature.get(side, {})
        if path not in leaves
    ]

    assert unknown == []


def test_ablation_default_plan_round_trips_through_config_cli():
    config  = BackboneEntryConfig()
    planner = AblationTrialPlanner(config.backbone_name, config.ablation_features, config.ablation_include_full)

    plans = planner.plan()
    assert len(plans) == len(config.ablation_features) + 1

    for run_name, overrides in plans:
        argv  = ConfigCli.to_argv({**overrides, "run_name": run_name, "logdir": "/tmp/abl"})
        trial = ConfigCli(BackboneEntryConfig()).apply(argv + ["--trial"])
        assert trial.run_name == run_name

    full     = ConfigCli(BackboneEntryConfig()).apply(ConfigCli.to_argv(dict(plans)[f"{config.backbone_name}_abl-0-full"]) + ["--trial"])
    baseline = ConfigCli(BackboneEntryConfig()).apply(ConfigCli.to_argv(dict(plans)[f"{config.backbone_name}_abl-{len(config.ablation_features)}-baseline"]) + ["--trial"])
    assert full.curriculum.enabled                       is True
    assert full.backbone_name                            == "resunet"
    assert full.curriculum.warmup.use_param_l1           is True
    assert full.curriculum.warmup.use_cosine_curve       is True
    assert full.training.warmup_enabled                  is True
    assert baseline.curriculum.enabled                   is False
    assert baseline.backbone_name                        == "unet"
    assert baseline.curriculum.warmup.use_param_mse      is True
    assert baseline.curriculum.warmup.use_param_l1       is False
    assert baseline.curriculum.warmup.use_active_normalization is False
    assert set(baseline.model_overrides.values())        == {3e-4}
    assert baseline.training.warmup_enabled              is False
    assert baseline.normalization.clamp_output           is False
    assert baseline.normalization.out_amp                == "zscore"
    assert baseline.normalization.ifg_phase              == "fixed_div_pi"
