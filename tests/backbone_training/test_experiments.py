from __future__ import annotations

import pytest

from configuration.sar.geometry_config import GeometryConfig
from configuration.training import BackboneEntryConfig, CurriculumInheritance, default_curriculum
from configuration.training.backbone        import PairTrialsConfig, PatchTrialsConfig, PhysicsTrialsConfig, SecondaryTrialsConfig, _default_input_trials, _default_presence_trials
from configuration.training.general.ablation import AblationCatalog
from pipelines.backbone.training.experiments import (
    AblationTrialPlanner,
    CurriculumTrialPlanner,
    InputTrialPlanner,
    PatchSizeTrialPlanner,
    PairLossTrialPlanner,
    PhysicsLossTrialPlanner,
    SecondaryTrialPlanner,
    SlotPresenceTrialPlanner,
    WarmupTrialPlanner,
)
from tools.runtime.config_cli import ConfigCli


CANDIDATES = ["PS04", "PS06", "PS08", "PS10", "PS12", "PS14"]


def test_curriculum_planner_crosses_warmup_and_complete():
    planner = CurriculumTrialPlanner(
        {"w1": {"use_param_l1": True}, "w2": {"use_l1_curve": True}},
        {"c1": {"use_mse_curve": True}},
    )

    plans = planner.plan()

    assert len(plans) == 2
    assert planner.summary() == {"Warmup losses": 2, "Complete losses": 1}

    run_name, overrides = plans[0]
    assert run_name == "w-w1_c-c1"
    assert overrides["curriculum.enabled"] is True
    assert overrides["curriculum.warmup.use_param_l1"] is True
    assert overrides["curriculum.complete.use_mse_curve"] is True


def test_warmup_planner_disables_curriculum():
    planner = WarmupTrialPlanner({"a": {"use_param_l1": True}, "b": {"use_mse_curve": True}})

    plans = planner.plan()

    assert len(plans) == 2
    assert all(overrides["curriculum.enabled"] is False for _, overrides in plans)
    assert plans[0][0] == "nc-a"
    assert plans[0][1]["curriculum.complete.use_param_l1"] is True


def test_presence_planner_disables_curriculum():
    planner = SlotPresenceTrialPlanner(
        {"AB": {"use_active_normalization": True, "presence_balance": True}},
    )

    plans = planner.plan()

    assert len(plans) == 1
    assert [name for name, _ in plans] == ["pr-AB"]
    assert planner.summary()["Total runs"] == 1

    ov = dict(plans)["pr-AB"]

    assert ov["curriculum.enabled"] is False
    assert "curriculum.complete.param_match" not in ov
    assert ov["curriculum.complete.use_active_normalization"]  is True
    assert ov["curriculum.complete.presence_balance"]          is True


def test_presence_planner_default_matrix():
    planner = SlotPresenceTrialPlanner(_default_presence_trials())

    plans = dict(planner.plan())

    assert len(plans) == 10
    assert "pr-none" in plans
    assert plans["pr-none"] == {"curriculum.enabled": False}

    for name, overrides in plans.items():
        if overrides.get("curriculum.warmup.amp_focal_gamma"):
            assert overrides["curriculum.warmup.amp_focal_gamma"] > 0.0


def test_patch_planner_emits_size_and_stride():
    planner = PatchSizeTrialPlanner(PatchTrialsConfig(sizes=[32, 64], stride_ratio=0.5))

    plans = planner.plan()

    assert len(plans) == 2

    name, overrides = plans[0]
    assert name == "p-32"
    assert overrides["training.patch_size"]   == (32, 32)
    assert overrides["training.patch_stride"] == 16

    assert plans[1][1]["training.patch_stride"] == 32


def test_patch_planner_probes_max_batch_and_scales_lr_by_default():
    planner = PatchSizeTrialPlanner(PatchTrialsConfig(sizes=[32, 64], stride_ratio=0.5))

    for _name, overrides in planner.plan():
        assert overrides["pretrain.find_batch_size"]     is True
        assert overrides["training.scale_lr_with_batch"] is True

    assert planner.summary()["Max-batch probe"] is True
    assert planner.summary()["Scale LR"]        is True


def test_patch_planner_can_disable_probe_and_lr_scaling():
    planner = PatchSizeTrialPlanner(PatchTrialsConfig(sizes=[32], stride_ratio=0.5, find_max_batch=False, scale_lr=False))

    overrides = planner.plan()[0][1]

    assert overrides["pretrain.find_batch_size"]     is False
    assert overrides["training.scale_lr_with_batch"] is False


def test_patch_planner_rejects_empty_sizes():
    with pytest.raises(ValueError):
        PatchSizeTrialPlanner(PatchTrialsConfig(sizes=[], stride_ratio=0.5))


def test_patch_planner_rejects_bad_stride_ratio():
    with pytest.raises(ValueError):
        PatchSizeTrialPlanner(PatchTrialsConfig(sizes=[32], stride_ratio=1.5))


def test_secondary_consecutive_blocks():
    trials  = SecondaryTrialsConfig(strategy="consecutive", n_secondaries=2, block_step=1)
    planner = SecondaryTrialPlanner(trials, CANDIDATES)

    plans = planner.plan()

    assert len(plans) == len(CANDIDATES) - 1

    name, overrides = plans[0]
    assert overrides["paths.secondary_labels"] == ("PS04", "PS06")
    assert name.startswith("sec-consecutive-t00")


def test_secondary_spaced_picks_strided_labels():
    trials  = SecondaryTrialsConfig(strategy="spaced", n_secondaries=2, block_step=1, spacing=2)
    planner = SecondaryTrialPlanner(trials, CANDIDATES)

    plans = planner.plan()

    assert plans[0][1]["paths.secondary_labels"] == ("PS04", "PS08")


def test_secondary_uniform_produces_distinct_trials():
    trials  = SecondaryTrialsConfig(strategy="uniform", n_secondaries=2, n_trials=4, seed=0)
    planner = SecondaryTrialPlanner(trials, CANDIDATES)

    plans = planner.plan()

    selections = [overrides["paths.secondary_labels"] for _, overrides in plans]

    assert len(plans) == 4
    assert len(set(selections)) == 4


def test_secondary_gaussian_requires_mean_and_sigma():
    trials = SecondaryTrialsConfig(strategy="gaussian", n_secondaries=2, n_trials=2)

    with pytest.raises(ValueError):
        SecondaryTrialPlanner(trials, CANDIDATES)


def test_secondary_rejects_unknown_strategy():
    trials = SecondaryTrialsConfig(strategy="bogus", n_secondaries=2)

    with pytest.raises(ValueError):
        SecondaryTrialPlanner(trials, CANDIDATES)


def test_secondary_rejects_too_many_secondaries():
    trials = SecondaryTrialsConfig(strategy="consecutive", n_secondaries=99)

    with pytest.raises(ValueError):
        SecondaryTrialPlanner(trials, CANDIDATES)


def test_input_planner_default_catalog_covers_the_stack_grid():
    planner = InputTrialPlanner(_default_input_trials(), CANDIDATES)

    plans = dict(planner.plan())

    assert list(plans) == ["in-amp-allsec-noifg", "in-noamp-allsec-ifg", "in-amp-redsec-ifg", "in-amp-redsec-noifg", "in-noamp-redsec-ifg"]

    amp_all = plans["in-amp-allsec-noifg"]
    assert amp_all["input.use_primary"]        is True
    assert amp_all["input.use_secondaries"]    is True
    assert amp_all["input.use_interferograms"] is False
    assert amp_all["paths.secondary_labels"]   == tuple(CANDIDATES)

    ifg_all = plans["in-noamp-allsec-ifg"]
    assert ifg_all["input.use_primary"]        is False
    assert ifg_all["input.use_secondaries"]    is False
    assert ifg_all["input.use_interferograms"] is True
    assert ifg_all["paths.secondary_labels"]   == tuple(CANDIDATES)

    reduced = plans["in-amp-redsec-ifg"]
    assert reduced["input.use_primary"]        is True
    assert reduced["input.use_secondaries"]    is True
    assert reduced["input.use_interferograms"] is True

    amp_reduced = plans["in-amp-redsec-noifg"]
    assert amp_reduced["input.use_primary"]        is True
    assert amp_reduced["input.use_secondaries"]    is True
    assert amp_reduced["input.use_interferograms"] is False

    ifg_reduced = plans["in-noamp-redsec-ifg"]
    assert ifg_reduced["input.use_primary"]        is False
    assert ifg_reduced["input.use_secondaries"]    is False
    assert ifg_reduced["input.use_interferograms"] is True


def test_input_planner_scopes_tracks_per_variant():
    trials  = {"a": {"tracks": "all", "use_interferograms": False}, "b": {"tracks": "reduced", "use_dem": True}}
    planner = InputTrialPlanner(trials, CANDIDATES)

    plans = dict(planner.plan())

    assert set(plans) == {"in-a", "in-b"}
    assert plans["in-a"]["paths.secondary_labels"] == tuple(CANDIDATES)
    assert "paths.secondary_labels" not in plans["in-b"]
    assert "input.tracks"           not in plans["in-a"]
    assert "input.tracks"           not in plans["in-b"]
    assert planner.summary() == {"Input variants": 2, "Tracks": f"1 all ({len(CANDIDATES)} secondaries), 1 reduced (configured selection)"}


def test_input_planner_rejects_unknown_keys():
    with pytest.raises(ValueError):
        InputTrialPlanner({"bad": {"tracks": "all", "use_phase": True}}, CANDIDATES)


def test_input_planner_rejects_missing_or_invalid_track_scope():
    with pytest.raises(ValueError):
        InputTrialPlanner({"bad": {"use_primary": True}}, CANDIDATES)

    with pytest.raises(ValueError):
        InputTrialPlanner({"bad": {"tracks": "some", "use_primary": True}}, CANDIDATES)


def test_input_planner_rejects_empty_trials():
    with pytest.raises(ValueError):
        InputTrialPlanner({}, CANDIDATES)


def test_physics_planner_crosses_components_weights_and_curriculum():
    trials  = PhysicsTrialsConfig(components=["coherence_resyn", "capon_cycle"], weights=[0.01, 0.05], curriculum_states=[True, False], include_baseline=False)
    planner = PhysicsLossTrialPlanner(trials)

    plans = planner.plan()
    names = [name for name, _ in plans]

    assert len(plans) == 8
    assert names == [
        "phys-coherence_resyn-w0.01-cur",
        "phys-coherence_resyn-w0.01-nc",
        "phys-coherence_resyn-w0.05-cur",
        "phys-coherence_resyn-w0.05-nc",
        "phys-capon_cycle-w0.01-cur",
        "phys-capon_cycle-w0.01-nc",
        "phys-capon_cycle-w0.05-cur",
        "phys-capon_cycle-w0.05-nc",
    ]
    assert planner.summary()["Total runs"] == 8
    assert planner.summary()["Curriculum"] == ["on", "off"]

    overrides = dict(plans)["phys-capon_cycle-w0.05-cur"]
    assert overrides["curriculum.enabled"]                     is True
    assert overrides["curriculum.complete.use_capon_cycle"]    is True
    assert overrides["curriculum.complete.weight_capon_cycle"] == 0.05

    overrides = dict(plans)["phys-capon_cycle-w0.05-nc"]
    assert overrides["curriculum.enabled"]                     is False
    assert overrides["curriculum.complete.use_capon_cycle"]    is True
    assert overrides["curriculum.complete.weight_capon_cycle"] == 0.05


def test_physics_planner_neutralizes_untested_terms():
    trials  = PhysicsTrialsConfig(components=["coherence_resyn"], weights=[0.1], curriculum_states=[True], include_baseline=False)
    planner = PhysicsLossTrialPlanner(trials)

    overrides = planner.plan()[0][1]

    assert overrides["curriculum.inherit"] is False

    for component in PhysicsLossTrialPlanner.COMPONENTS:
        assert overrides[f"curriculum.warmup.use_{component}"]    is False
        assert overrides[f"curriculum.warmup.weight_{component}"] == 0.0

    for component in set(PhysicsLossTrialPlanner.COMPONENTS) - {"coherence_resyn"}:
        assert overrides[f"curriculum.complete.use_{component}"]    is False
        assert overrides[f"curriculum.complete.weight_{component}"] == 0.0


def test_physics_planner_prepends_baseline_run_per_curriculum_state():
    trials  = PhysicsTrialsConfig(components=["total_power"], weights=[0.01], curriculum_states=[True, False], include_baseline=True)
    planner = PhysicsLossTrialPlanner(trials)

    plans = planner.plan()

    assert [name for name, _ in plans] == [
        "phys-baseline-cur",
        "phys-baseline-nc",
        "phys-total_power-w0.01-cur",
        "phys-total_power-w0.01-nc",
    ]
    assert planner.summary()["Total runs"] == 4

    baseline_cur = dict(plans)["phys-baseline-cur"]
    baseline_nc  = dict(plans)["phys-baseline-nc"]
    assert baseline_cur["curriculum.enabled"] is True
    assert baseline_nc["curriculum.enabled"]  is False
    assert not any(key.startswith("curriculum.complete.use_") and value is True for key, value in baseline_cur.items())
    assert not any(key.startswith("curriculum.complete.use_") and value is True for key, value in baseline_nc.items())


def test_physics_planner_default_config_covers_recommended_terms():
    planner = PhysicsLossTrialPlanner(PhysicsTrialsConfig())

    plans = planner.plan()

    assert len(plans) == 14
    assert "phys-baseline-cur" in dict(plans)
    assert "phys-baseline-nc"  in dict(plans)
    assert f"phys-coherence_resyn-w{AblationCatalog.PHYSICS_WEIGHT:g}-cur" in dict(plans)
    assert f"phys-covariance_match-w{AblationCatalog.PHYSICS_WEIGHT:g}-nc" in dict(plans)


def test_physics_planner_rejects_empty_components():
    with pytest.raises(ValueError, match="at least one"):
        PhysicsLossTrialPlanner(PhysicsTrialsConfig(components=[], weights=[0.1]))


def test_physics_planner_rejects_unknown_component():
    with pytest.raises(ValueError, match="smoothness_tv"):
        PhysicsLossTrialPlanner(PhysicsTrialsConfig(components=["smoothness_tv"], weights=[0.1]))


def test_physics_planner_rejects_duplicate_components():
    with pytest.raises(ValueError, match="unique"):
        PhysicsLossTrialPlanner(PhysicsTrialsConfig(components=["moments", "moments"], weights=[0.1]))


def test_physics_planner_rejects_empty_weights():
    with pytest.raises(ValueError, match="weight"):
        PhysicsLossTrialPlanner(PhysicsTrialsConfig(components=["moments"], weights=[]))


def test_physics_planner_rejects_non_positive_weights():
    with pytest.raises(ValueError, match="positive"):
        PhysicsLossTrialPlanner(PhysicsTrialsConfig(components=["moments"], weights=[0.1, 0.0]))


def test_physics_planner_rejects_duplicate_weights():
    with pytest.raises(ValueError, match="unique"):
        PhysicsLossTrialPlanner(PhysicsTrialsConfig(components=["moments"], weights=[0.1, 0.1]))


def test_physics_planner_rejects_empty_curriculum_states():
    with pytest.raises(ValueError, match="curriculum state"):
        PhysicsLossTrialPlanner(PhysicsTrialsConfig(components=["moments"], weights=[0.1], curriculum_states=[]))


def test_physics_planner_rejects_non_boolean_curriculum_states():
    with pytest.raises(ValueError, match="boolean"):
        PhysicsLossTrialPlanner(PhysicsTrialsConfig(components=["moments"], weights=[0.1], curriculum_states=[1]))


def test_physics_planner_rejects_duplicate_curriculum_states():
    with pytest.raises(ValueError, match="unique"):
        PhysicsLossTrialPlanner(PhysicsTrialsConfig(components=["moments"], weights=[0.1], curriculum_states=[True, True]))


def test_physics_plan_round_trips_through_config_cli():
    trials  = PhysicsTrialsConfig(components=["coherence_resyn", "covariance_match"], weights=[0.05], curriculum_states=[True, False], include_baseline=True)
    planner = PhysicsLossTrialPlanner(trials)

    def _apply(overrides: dict):
        cli   = ConfigCli(BackboneEntryConfig())
        trial = cli.apply(ConfigCli.to_argv(overrides) + ["--trial"])
        CurriculumInheritance(trial.curriculum, default_curriculum(), cli.overrides).apply()
        return trial

    plans = dict(planner.plan())

    baseline = _apply({**plans["phys-baseline-cur"], "run_name": "phys-baseline-cur", "logdir": "/tmp/phys"})
    assert baseline.curriculum.enabled                       is True
    assert baseline.curriculum.complete.use_coherence_resyn  is False
    assert baseline.curriculum.complete.use_covariance_match is False

    tested = _apply({**plans["phys-coherence_resyn-w0.05-cur"], "run_name": "phys-coherence_resyn-w0.05-cur", "logdir": "/tmp/phys"})
    assert tested.curriculum.enabled                           is True
    assert tested.curriculum.complete.use_coherence_resyn      is True
    assert tested.curriculum.complete.weight_coherence_resyn   == 0.05
    assert tested.curriculum.complete.use_covariance_match     is False
    assert tested.curriculum.warmup.use_coherence_resyn        is False
    assert tested.curriculum.warmup.weight_coherence_resyn     == 0.0

    immediate = _apply({**plans["phys-coherence_resyn-w0.05-nc"], "run_name": "phys-coherence_resyn-w0.05-nc", "logdir": "/tmp/phys"})
    assert immediate.curriculum.enabled                          is False
    assert immediate.curriculum.complete.use_coherence_resyn     is True
    assert immediate.curriculum.complete.weight_coherence_resyn  == 0.05
    assert immediate.curriculum.initial_stage.use_coherence_resyn is True


def test_physics_planner_paths_are_entry_config_leaves():
    planner = PhysicsLossTrialPlanner(PhysicsTrialsConfig())
    leaves  = {path for path, _ in ConfigCli._leaves(BackboneEntryConfig())}

    unknown = [path for _, overrides in planner.plan() for path in overrides if path not in leaves]

    assert unknown == []


def test_pair_planner_crosses_components_and_weights():
    trials  = PairTrialsConfig(base_component="param_l1", components=["cosine_curve", "coherence_resyn"], weights=[0.01, 0.05], include_baseline=False)
    planner = PairLossTrialPlanner(trials)

    plans = planner.plan()
    names = [name for name, _ in plans]

    assert len(plans) == 4
    assert names == [
        "pair-cosine_curve-w0.01",
        "pair-cosine_curve-w0.05",
        "pair-coherence_resyn-w0.01",
        "pair-coherence_resyn-w0.05",
    ]
    assert planner.summary()["Total runs"] == 4
    assert planner.summary()["Base"]       == "param_l1 @ 1"

    overrides = dict(plans)["pair-cosine_curve-w0.05"]
    assert overrides["curriculum.enabled"]                    is False
    assert overrides["curriculum.inherit"]                    is False
    assert overrides["curriculum.complete.use_param_l1"]      is True
    assert overrides["curriculum.complete.weight_param_l1"]   == 1.0
    assert overrides["curriculum.complete.use_cosine_curve"]  is True
    assert overrides["curriculum.complete.weight_cosine_curve"] == 0.05


def test_pair_planner_neutralizes_every_other_term():
    trials  = PairTrialsConfig(base_component="param_l1", components=["covariance_match"], weights=[0.1], include_baseline=False)
    planner = PairLossTrialPlanner(trials)

    overrides = planner.plan()[0][1]
    enabled   = [path for path, value in overrides.items() if path.startswith("curriculum.complete.use_") and value is True]

    assert sorted(enabled) == ["curriculum.complete.use_covariance_match", "curriculum.complete.use_param_l1"]
    assert overrides["curriculum.complete.use_cosine_curve"]      is False
    assert overrides["curriculum.complete.weight_cosine_curve"]   == 0.0
    assert overrides["curriculum.complete.use_coherence_resyn"]   is False
    assert overrides["curriculum.complete.weight_coherence_resyn"] == 0.0


def test_pair_planner_uses_catalog_flags_for_irregular_names():
    trials  = PairTrialsConfig(base_component="param_l1", components=["total_power_relerr"], weights=[0.1], include_baseline=False)
    planner = PairLossTrialPlanner(trials)

    name, overrides = planner.plan()[0]

    assert name == "pair-total_power_relerr-w0.1"
    assert overrides["curriculum.complete.use_total_power"]    is True
    assert overrides["curriculum.complete.weight_total_power"] == 0.1


def test_pair_planner_prepends_base_only_baseline():
    trials  = PairTrialsConfig(base_component="cosine_curve", base_weight=0.5, components=["param_l1"], weights=[0.01], include_baseline=True)
    planner = PairLossTrialPlanner(trials)

    plans = planner.plan()

    assert [name for name, _ in plans] == ["pair-baseline", "pair-param_l1-w0.01"]

    baseline = dict(plans)["pair-baseline"]
    enabled  = [path for path, value in baseline.items() if path.startswith("curriculum.complete.use_") and value is True]

    assert enabled == ["curriculum.complete.use_cosine_curve"]
    assert baseline["curriculum.complete.weight_cosine_curve"] == 0.5


def test_pair_planner_default_config_targets_param_l1_partners():
    planner = PairLossTrialPlanner(PairTrialsConfig())

    plans = planner.plan()

    assert len(plans) == 10
    assert "pair-baseline" in dict(plans)
    assert f"pair-coherence_resyn-w{AblationCatalog.PHYSICS_WEIGHT:g}" in dict(plans)


def test_pair_planner_rejects_unknown_base_component():
    with pytest.raises(ValueError, match="base_component"):
        PairLossTrialPlanner(PairTrialsConfig(base_component="bogus", components=["cosine_curve"], weights=[0.1]))


def test_pair_planner_rejects_non_positive_base_weight():
    with pytest.raises(ValueError, match="base_weight"):
        PairLossTrialPlanner(PairTrialsConfig(base_weight=0.0, components=["cosine_curve"], weights=[0.1]))


def test_pair_planner_rejects_empty_components():
    with pytest.raises(ValueError, match="at least one"):
        PairLossTrialPlanner(PairTrialsConfig(components=[], weights=[0.1]))


def test_pair_planner_rejects_unknown_component():
    with pytest.raises(ValueError, match="bogus"):
        PairLossTrialPlanner(PairTrialsConfig(components=["bogus"], weights=[0.1]))


def test_pair_planner_rejects_base_repeated_as_candidate():
    with pytest.raises(ValueError, match="repeat"):
        PairLossTrialPlanner(PairTrialsConfig(base_component="param_l1", components=["param_l1", "cosine_curve"], weights=[0.1]))


def test_pair_planner_rejects_duplicate_components():
    with pytest.raises(ValueError, match="unique"):
        PairLossTrialPlanner(PairTrialsConfig(components=["moments", "moments"], weights=[0.1]))


def test_pair_planner_rejects_bad_weights():
    with pytest.raises(ValueError, match="at least one"):
        PairLossTrialPlanner(PairTrialsConfig(components=["moments"], weights=[]))

    with pytest.raises(ValueError, match="positive"):
        PairLossTrialPlanner(PairTrialsConfig(components=["moments"], weights=[0.1, -0.1]))

    with pytest.raises(ValueError, match="unique"):
        PairLossTrialPlanner(PairTrialsConfig(components=["moments"], weights=[0.1, 0.1]))


def test_pair_plan_round_trips_through_config_cli():
    trials  = PairTrialsConfig(base_component="param_l1", components=["cosine_curve"], weights=[0.05], include_baseline=True)
    planner = PairLossTrialPlanner(trials)

    def _apply(overrides: dict):
        cli   = ConfigCli(BackboneEntryConfig())
        trial = cli.apply(ConfigCli.to_argv(overrides) + ["--trial"])
        CurriculumInheritance(trial.curriculum, default_curriculum(), cli.overrides).apply()
        return trial

    plans = dict(planner.plan())

    baseline = _apply({**plans["pair-baseline"], "run_name": "pair-baseline", "logdir": "/tmp/pair"})
    assert baseline.curriculum.enabled                         is False
    assert baseline.curriculum.initial_stage.use_param_l1      is True
    assert baseline.curriculum.initial_stage.use_cosine_curve  is False
    assert baseline.curriculum.initial_stage.use_coherence_resyn is False

    tested = _apply({**plans["pair-cosine_curve-w0.05"], "run_name": "pair-cosine_curve-w0.05", "logdir": "/tmp/pair"})
    assert tested.curriculum.enabled                              is False
    assert tested.curriculum.initial_stage.use_param_l1           is True
    assert tested.curriculum.initial_stage.weight_param_l1        == 1.0
    assert tested.curriculum.initial_stage.use_cosine_curve       is True
    assert tested.curriculum.initial_stage.weight_cosine_curve    == 0.05
    assert tested.curriculum.initial_stage.use_covariance_match   is False


def test_pair_planner_paths_are_entry_config_leaves():
    planner = PairLossTrialPlanner(PairTrialsConfig())
    leaves  = {path for path, _ in ConfigCli._leaves(BackboneEntryConfig())}

    unknown = [path for _, overrides in planner.plan() for path in overrides if path not in leaves]

    assert unknown == []


ABL_FEATURES = [
    {"label": "active_norm", "enable": {"curriculum.warmup.use_active_normalization": True}, "degrade": {"curriculum.warmup.use_active_normalization": False}},
    {"label": "focal",       "enable": {"curriculum.warmup.amp_focal_gamma": 2.0},           "degrade": {"curriculum.warmup.amp_focal_gamma": 0.0}},
    {"label": "balance",     "enable": {"curriculum.warmup.presence_balance": True},         "degrade": {"curriculum.warmup.presence_balance": False}},
]


def test_ablation_planner_cumulative_full_to_baseline():
    planner = AblationTrialPlanner(ABL_FEATURES, include_full=True)

    plans = planner.plan()
    names = [name for name, _ in plans]

    assert names == [
        "abl-0-full",
        "abl-1-no_active_norm",
        "abl-2-no_focal",
        "abl-3-baseline",
    ]
    assert planner.summary()["Total runs"] == 4

    full = dict(plans)["abl-0-full"]
    assert full["curriculum.warmup.use_active_normalization"] is True
    assert full["curriculum.warmup.amp_focal_gamma"]      == 2.0
    assert full["curriculum.warmup.presence_balance"]     is True

    step1 = dict(plans)["abl-1-no_active_norm"]
    assert step1["curriculum.warmup.use_active_normalization"] is False
    assert step1["curriculum.warmup.amp_focal_gamma"]     == 2.0
    assert step1["curriculum.warmup.presence_balance"]    is True

    baseline = dict(plans)["abl-3-baseline"]
    assert baseline["curriculum.warmup.use_active_normalization"] is False
    assert baseline["curriculum.warmup.amp_focal_gamma"]  == 0.0
    assert baseline["curriculum.warmup.presence_balance"] is False


def test_ablation_planner_without_full_run():
    planner = AblationTrialPlanner(ABL_FEATURES, include_full=False)

    plans = planner.plan()

    assert [name for name, _ in plans] == [
        "abl-1-no_active_norm",
        "abl-2-no_focal",
        "abl-3-baseline",
    ]
    assert planner.summary()["Total runs"] == 3


def test_ablation_planner_rejects_empty_features():
    with pytest.raises(ValueError):
        AblationTrialPlanner([], include_full=True)


def test_ablation_planner_rejects_feature_without_degrade():
    with pytest.raises(ValueError):
        AblationTrialPlanner([{"label": "x"}], include_full=True)


def test_ablation_planner_rejects_feature_without_enable():
    with pytest.raises(ValueError, match="enable"):
        AblationTrialPlanner([{"label": "x", "degrade": {"curriculum.enabled": False}}], include_full=True)


def test_ablation_planner_rejects_duplicate_labels():
    features = [
        {"label": "same", "enable": {"curriculum.warmup.use_cosine_curve": True},         "degrade": {"curriculum.warmup.use_cosine_curve": False}},
        {"label": "same", "enable": {"curriculum.warmup.use_active_normalization": True}, "degrade": {"curriculum.warmup.use_active_normalization": False}},
    ]

    with pytest.raises(ValueError, match="unique"):
        AblationTrialPlanner(features, include_full=True)


def test_ablation_planner_allows_complete_terms_with_disabled_curriculum():
    features = [
        {"label": "curriculum", "enable": {"curriculum.enabled": True},                          "degrade": {"curriculum.enabled": False}},
        {"label": "covariance", "enable": {"curriculum.complete.use_covariance_match": True},    "degrade": {"curriculum.complete.use_covariance_match": False}},
    ]

    planner = AblationTrialPlanner(features, include_full=True)

    names = [name for name, _ in planner.plan()]
    assert names == ["abl-0-full", "abl-1-no_curriculum", "abl-2-baseline"]

    no_curriculum = dict(planner.plan())["abl-1-no_curriculum"]
    assert no_curriculum["curriculum.enabled"]                       is False
    assert no_curriculum["curriculum.complete.use_covariance_match"] is True


def test_ablation_catalog_default_is_the_standard_set():
    features = AblationCatalog.default_features()
    labels   = [feature["label"] for feature in features]

    assert labels == [
        "covariance_match", "physics_curriculum", "coherence_resyn",
        "cosine_curve", "architecture_param_loss", "augmentation",
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
    assert catalog["ifg_phase"]["degrade"]["normalization.ifg_phase"]  == "zscore"
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
    assert "curriculum.warmup.use_coherence_resyn" not in coherence["degrade"]

    physics_curriculum = catalog["physics_curriculum"]
    assert physics_curriculum["enable"]["curriculum.enabled"]  is True
    assert physics_curriculum["degrade"]                       == {"curriculum.enabled": False}

    cosine = catalog["cosine_curve"]
    assert cosine["enable"]["curriculum.complete.use_cosine_curve"]  is True
    assert cosine["degrade"]["curriculum.complete.use_cosine_curve"] is False
    assert "curriculum.warmup.use_cosine_curve" not in cosine["enable"]

    assert "class_imbalance"  not in catalog
    assert "predict_presence" not in catalog

    architecture = catalog["architecture_param_loss"]
    assert architecture["enable"]["backbone_name"]                         == "resunet"
    assert architecture["enable"]["curriculum.complete.use_param_l1"]      is True
    assert architecture["enable"]["curriculum.complete.use_param_mse"]     is False
    assert architecture["degrade"]["backbone_name"]                        == "unet"
    assert architecture["degrade"]["curriculum.complete.use_param_l1"]     is False
    assert architecture["degrade"]["curriculum.complete.use_param_mse"]    is True
    assert architecture["degrade"]["curriculum.complete.weight_param_mse"] == 1.0

    active_norm = catalog["active_norm"]
    assert active_norm["enable"]["curriculum.complete.use_active_normalization"]  is True
    assert active_norm["degrade"]["curriculum.complete.use_active_normalization"] is False

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

    assert "curve_loss_mse_to_l1" not in catalog
    assert "spectral_coherence"   not in catalog
    assert "ssim"                 not in catalog
    assert "smoothness_tv"        in catalog


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
    planner = InputTrialPlanner.from_dataset(_default_input_trials(), GeometryConfig(), test_data_dir)

    plans = dict(planner.plan())

    assert len(plans) == 5

    overrides = plans["in-amp-allsec-noifg"]
    assert overrides["input.use_interferograms"] is False
    assert len(overrides["paths.secondary_labels"]) == len(planner.candidates) >= 1

    assert "paths.secondary_labels" not in plans["in-amp-redsec-ifg"]


@pytest.mark.real_data
def test_secondary_from_dataset_loads_candidates(test_data_dir):
    trials  = SecondaryTrialsConfig(strategy="consecutive", n_secondaries=2, block_step=1)
    planner = SecondaryTrialPlanner.from_dataset(trials, GeometryConfig(), test_data_dir)

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
    planner = AblationTrialPlanner(config.ablation_features, config.ablation_include_full)

    plans = planner.plan()
    assert len(plans) == len(config.ablation_features) + 1

    def _apply(overrides: dict):
        cli   = ConfigCli(BackboneEntryConfig())
        trial = cli.apply(ConfigCli.to_argv(overrides) + ["--trial"])
        CurriculumInheritance(trial.curriculum, default_curriculum(), cli.overrides).apply()
        return trial

    for run_name, overrides in plans:
        trial = _apply({**overrides, "run_name": run_name, "logdir": "/tmp/abl"})
        assert trial.run_name == run_name

    full     = _apply(dict(plans)["abl-0-full"])
    baseline = _apply(dict(plans)[f"abl-{len(config.ablation_features)}-baseline"])
    assert full.curriculum.enabled                         is True
    assert full.backbone_name                              == "resunet"
    assert full.curriculum.complete.use_param_l1           is True
    assert full.curriculum.complete.use_cosine_curve       is True
    assert full.curriculum.warmup.use_param_l1             is True
    assert full.curriculum.warmup.use_cosine_curve         is True
    assert full.curriculum.warmup.use_coherence_resyn      is False
    assert full.training.warmup_enabled                    is True
    assert baseline.curriculum.enabled                     is False
    assert baseline.backbone_name                          == "unet"
    assert baseline.curriculum.complete.use_param_mse      is True
    assert baseline.curriculum.complete.use_param_l1       is False
    assert baseline.curriculum.complete.use_active_normalization is False
    assert baseline.curriculum.warmup.use_param_mse        is True
    assert set(baseline.model_overrides.values())          == {3e-4}
    assert baseline.training.warmup_enabled                is False
    assert baseline.normalization.clamp_output           is False
    assert baseline.normalization.out_amp                == "zscore"
    assert baseline.normalization.ifg_phase              == "zscore"
