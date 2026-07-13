from __future__ import annotations

from pathlib import Path

import pytest

from configuration.patch_sweep import PatchSweepConfig
from pipelines.patch_sweep.planner import ArchitecturePatchStep, PatchSweepPlanner


def make_datasets(names: list[str]) -> list[Path]:
    return [Path("/data") / name for name in names]


def make_planner(datasets: list[str], **patch_overrides) -> PatchSweepPlanner:
    config               = PatchSweepConfig()
    config.dataset_paths = make_datasets(datasets)
    for key, value in patch_overrides.items():
        setattr(config.patch, key, value)
    return PatchSweepPlanner(config)


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
    planner = make_planner(["w20_10"], step=32, maximum=96)

    assert planner.patch_sizes() == [32, 64, 96]


def test_sizes_run_from_one_step_to_maximum():
    planner = make_planner(["w20_10"], maximum=128)

    assert planner.patch_sizes() == [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]


def test_minimum_must_be_admissible():
    planner = make_planner(["w20_10"], minimum=20)

    with pytest.raises(ValueError, match="multiple of the admissible step"):
        planner.patch_sizes()


def test_single_size_grid_is_rejected():
    planner = make_planner(["w20_10"], maximum=8)

    with pytest.raises(ValueError, match="at least 2"):
        planner.patch_sizes()


def test_datasets_must_be_present():
    with pytest.raises(ValueError, match="at least one dataset"):
        make_planner([])


def test_dataset_names_must_be_unique():
    config               = PatchSweepConfig()
    config.dataset_paths = [Path("/data/a/w20_10"), Path("/data/b/w20_10")]

    with pytest.raises(ValueError, match="unique"):
        PatchSweepPlanner(config)


def test_parameters_template_must_live_inside_the_dataset():
    config                       = PatchSweepConfig()
    config.dataset_paths         = make_datasets(["w20_10"])
    config.paths.parameters_path = Path("/elsewhere/parameters.npy")

    with pytest.raises(ValueError, match="re-roots"):
        PatchSweepPlanner(config)


def test_parameters_are_rerooted_onto_every_dataset():
    planner  = make_planner(["w20_10", "w20_20"], maximum=16)
    template = planner.parameters_template()

    for unit in planner.units():
        assert unit.parameters_path == unit.dataset_path / template
        assert str(unit.parameters_path).startswith(str(unit.dataset_path))


def test_units_cover_the_full_cross_product():
    planner = make_planner(["w20_10", "w20_20"], maximum=64)
    units   = planner.units()

    assert len(units) == 2 * 8
    assert sorted({unit.dataset for unit in units}) == ["w20_10", "w20_20"]
    assert {unit.name for unit in units} == {f"{d}-p{s:03d}" for d in ("w20_10", "w20_20") for s in range(8, 65, 8)}


def test_units_keep_the_config_dataset_order():
    planner = make_planner(["w20_20", "w20_10"], maximum=16)

    assert [unit.dataset for unit in planner.units()] == ["w20_20", "w20_20", "w20_10", "w20_10"]


def test_constant_pixel_budget_rescales_the_batch():
    planner = make_planner(["w20_10"], maximum=128)
    by_size = {unit.patch_size: unit.batch_size for unit in planner.units()}

    reference = planner.config.training.batch_size * planner.config.training.patch_size[0] * planner.config.training.patch_size[1]

    assert by_size[64]  == planner.config.training.batch_size
    assert by_size[16]  == reference // (16 * 16)
    assert by_size[128] == reference // (128 * 128)


def test_fixed_batch_when_the_budget_is_disabled():
    planner = make_planner(["w20_10"], constant_pixel_budget=False)

    assert {unit.batch_size for unit in planner.units()} == {planner.config.training.batch_size}


def test_constant_pixel_budget_keeps_the_lr_scale_constant():
    planner    = make_planner(["w20_10"], maximum=128)
    configured = planner.config.training.batch_size / planner.config.training.lr_reference_batch_size

    for unit in planner.units():
        assert unit.batch_size / unit.lr_reference_batch_size == pytest.approx(configured)


def test_lr_reference_rescales_with_the_pixel_budget():
    planner = make_planner(["w20_10"], maximum=128)
    by_size = {unit.patch_size: unit.lr_reference_batch_size for unit in planner.units()}

    reference = planner.config.training.lr_reference_batch_size * planner.config.training.patch_size[0] * planner.config.training.patch_size[1]

    assert by_size[64]  == planner.config.training.lr_reference_batch_size
    assert by_size[16]  == reference // (16 * 16)
    assert by_size[128] == reference // (128 * 128)


def test_lr_reference_untouched_when_the_budget_is_disabled():
    planner = make_planner(["w20_10"], constant_pixel_budget=False)

    assert {unit.lr_reference_batch_size for unit in planner.units()} == {planner.config.training.lr_reference_batch_size}


def test_summary_reports_the_seed_axis():
    planner = make_planner(["w20_10", "w20_20"], maximum=16)
    summary = planner.summary()

    assert summary["Datasets"] == ["w20_10", "w20_20"]
    assert summary["Units"]    == 4
    assert summary["Seeds"]    == [0, 1, 2, 3, 4]


def test_unit_lookup_is_loud_for_unknown_names():
    planner = make_planner(["w20_10"])

    with pytest.raises(KeyError, match="w99-p016"):
        planner.unit("w99-p016")
