from __future__ import annotations

import math

import pytest

from configuration.patch_sweep import PatchSweepConfig
from configuration.sar.geometry_config import GeometryConfig
from pipelines.patch_sweep.planner import ArchitecturePatchStep, PatchSweepPlanner, SecondarySpread
from tools.baselines import TrackBaselines


def make_candidates(n: int = 28) -> list[str]:
    return [f"FL01_PS{i:02d}" for i in range(3, 3 + n)]


def make_planner(track_counts: list[int], **patch_overrides) -> PatchSweepPlanner:
    config = PatchSweepConfig(track_counts=track_counts)
    for key, value in patch_overrides.items():
        setattr(config.patch, key, value)
    return PatchSweepPlanner(config, make_candidates())


def test_step_derives_from_the_resunet_feature_pyramid():
    assert ArchitecturePatchStep("resunet", "conv", {}).resolve() == 16


def test_step_scales_with_the_pyramid_depth():
    assert ArchitecturePatchStep("resunet", "conv", {"features": [32, 64]}).resolve() == 4


def test_sweep_default_narrows_the_pyramid_to_three_levels_for_step_eight():
    config = PatchSweepConfig()

    assert len(config.model_overrides["features"]) == 3
    assert ArchitecturePatchStep(config.backbone_name, config.backbone_head, config.model_overrides).resolve() == 8


def test_step_rejects_backbones_outside_the_verified_unet_family():
    with pytest.raises(ValueError, match="patch.step"):
        ArchitecturePatchStep("pixel_mlp", "conv", {}).resolve()


def test_explicit_step_overrides_the_architecture():
    planner = make_planner([5], step=32, maximum=96)

    assert planner.patch_sizes() == [32, 64, 96]


def test_sizes_run_from_one_step_to_maximum():
    planner = make_planner([5], maximum=128)

    assert planner.patch_sizes() == [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]


def test_minimum_must_be_admissible():
    planner = make_planner([5], minimum=20)

    with pytest.raises(ValueError, match="multiple of the admissible step"):
        planner.patch_sizes()


def test_single_size_grid_is_rejected():
    planner = make_planner([5], maximum=8)

    with pytest.raises(ValueError, match="at least 2"):
        planner.patch_sizes()


def test_track_counts_must_fit_the_dataset():
    with pytest.raises(ValueError, match=r"\[2, 29\]"):
        make_planner([5, 30])


def test_track_counts_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        make_planner([5, 5])


def make_table() -> TrackBaselines:
    labels   = ["REF"] + [f"T{i}" for i in range(6)]
    vertical = [0.0, -0.9, 0.4, -2.1, -0.2, -1.5, 0.1]

    return TrackBaselines(
        labels         = labels,
        vertical       = vertical,
        horizontal     = [0.0] * len(labels),
        vertical_std   = [0.0] * len(labels),
        horizontal_std = [0.0] * len(labels),
    )


def test_candidates_are_ordered_by_the_geometry_baseline_component():
    geometry = GeometryConfig(baseline_component="vertical")
    ordered  = PatchSweepPlanner.baseline_ordered(geometry, make_table())

    assert ordered == ["T2", "T4", "T0", "T3", "T5", "T1"]


def test_baseline_ordering_never_includes_the_reference():
    geometry = GeometryConfig(baseline_component="vertical")

    assert "REF" not in PatchSweepPlanner.baseline_ordered(geometry, make_table())


@pytest.mark.real_data
def test_dataset_selection_spans_the_full_baseline_aperture(test_data_dir):
    config = PatchSweepConfig(track_counts=[5, 9, 29])
    config.paths.dataset_path = test_data_dir

    planner  = PatchSweepPlanner.from_dataset(config)
    geometry = config.geometry
    table    = TrackBaselines.load(geometry.baselines_file(test_data_dir))
    values   = dict(zip(table.labels, table.baselines(geometry.baseline_component, look_angle_deg=geometry.look_angle_deg)))

    candidate_values = [values[label] for label in planner.candidates]

    assert candidate_values == sorted(candidate_values)

    for track_count, labels in planner.selections().items():
        selected = [values[label] for label in labels]

        assert min(selected) == min(candidate_values)
        assert max(selected) == max(candidate_values)


def test_even_spread_keeps_the_endpoints():
    candidates = make_candidates()
    spread     = SecondarySpread.even(candidates, 4)

    assert spread[0]  == candidates[0]
    assert spread[-1] == candidates[-1]
    assert len(set(spread)) == 4


def test_even_spread_is_distinct_for_every_count():
    candidates = make_candidates()

    for n in range(1, len(candidates) + 1):
        spread = SecondarySpread.even(candidates, n)
        assert len(set(spread)) == n


def test_full_count_selects_every_secondary():
    planner = make_planner([29])

    assert planner.selections()[29] == tuple(make_candidates())


def test_units_cover_the_full_cross_product():
    planner = make_planner([5, 9], maximum=64)
    units   = planner.units()

    assert len(units) == 2 * 8
    assert sorted({unit.track_count for unit in units}) == [5, 9]
    assert {unit.name for unit in units} == {f"n{n:02d}-p{s:03d}" for n in (5, 9) for s in range(8, 65, 8)}


def test_constant_pixel_budget_rescales_the_batch():
    planner = make_planner([5], maximum=128)
    by_size = {unit.patch_size: unit.batch_size for unit in planner.units()}

    reference = planner.config.training.batch_size * planner.config.training.patch_size[0] * planner.config.training.patch_size[1]

    assert by_size[64]  == planner.config.training.batch_size
    assert by_size[16]  == reference // (16 * 16)
    assert by_size[128] == reference // (128 * 128)


def test_fixed_batch_when_the_budget_is_disabled():
    planner = make_planner([5], constant_pixel_budget=False)

    assert {unit.batch_size for unit in planner.units()} == {planner.config.training.batch_size}


def test_constant_pixel_budget_keeps_the_lr_scale_constant():
    planner    = make_planner([5], maximum=128)
    configured = planner.config.training.batch_size / planner.config.training.lr_reference_batch_size

    for unit in planner.units():
        assert unit.batch_size / unit.lr_reference_batch_size == pytest.approx(configured)


def test_lr_reference_rescales_with_the_pixel_budget():
    planner = make_planner([5], maximum=128)
    by_size = {unit.patch_size: unit.lr_reference_batch_size for unit in planner.units()}

    reference = planner.config.training.lr_reference_batch_size * planner.config.training.patch_size[0] * planner.config.training.patch_size[1]

    assert by_size[64]  == planner.config.training.lr_reference_batch_size
    assert by_size[16]  == reference // (16 * 16)
    assert by_size[128] == reference // (128 * 128)


def test_lr_reference_untouched_when_the_budget_is_disabled():
    planner = make_planner([5], constant_pixel_budget=False)

    assert {unit.lr_reference_batch_size for unit in planner.units()} == {planner.config.training.lr_reference_batch_size}


def test_predicted_optimum_matches_the_closed_form():
    planner = make_planner([5])

    assert planner.predicted_optimum(5) == pytest.approx(20 * math.sqrt(29 / 5))
    assert planner.predicted_optimum(29) == pytest.approx(20.0)


def test_unit_lookup_is_loud_for_unknown_names():
    planner = make_planner([5])

    with pytest.raises(KeyError, match="n99-p016"):
        planner.unit("n99-p016")
