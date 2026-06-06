from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from configuration.dataset_config             import AugmentationConfig, InputConfig, OutputConfig
from configuration.norm_config                import ChannelStats, ChannelStrategy, NormMethod, Presets
from configuration.processing_config          import CropRegion
from configuration.representation             import Representation
from pipelines.dataset_pipeline.datasets      import Loader, MultiRegionDataset, PatchDataset, SpatialAugmenter
from pipelines.dataset_pipeline.normalization import Normalizer, Stats, StatsComputer
from pipelines.dataset_pipeline.spatial       import Cropper, GridInfo, Layout, Patcher
from tools.logger                             import NullLogger


def _null_logger():
    return NullLogger()


def _input_stats(n_channels: int, method: NormMethod = NormMethod.ZSCORE, log1p: bool = False) -> ChannelStats:
    strat = ChannelStrategy(method, apply_log1p=log1p)
    return ChannelStats(
        loc        = [0.0] * n_channels,
        scale      = [1.0] * n_channels,
        names      = [f"c{i}" for i in range(n_channels)],
        strategies = [strat] * n_channels,
    )


class TestPatcherBuild:
    def test_single_patch_when_spatial_not_larger_than_patch(self):
        patcher = Patcher.build(spatial_size=(10, 10), patch_size=(16, 16), stride=8)

        assert patcher.grid.n_v == 1
        assert patcher.grid.n_h == 1
        assert patcher.grid.number_of_patches == 1

    def test_padding_centered_for_undersized_spatial(self):
        patcher = Patcher.build(spatial_size=(10, 10), patch_size=(16, 16), stride=8)
        grid    = patcher.grid

        assert grid.padding_vertical   == 6
        assert grid.padding_horizontal == 6
        assert grid.pad_top  == 3
        assert grid.pad_bot  == 3
        assert grid.pad_left == 3
        assert grid.pad_right == 3

    def test_grid_count_for_multiple_patches(self):
        patcher = Patcher.build(spatial_size=(64, 64), patch_size=(32, 32), stride=32)

        assert patcher.grid.n_v == 2
        assert patcher.grid.n_h == 2
        assert patcher.grid.number_of_patches == 4

    def test_non_divisible_dimensions_round_up(self):
        patcher = Patcher.build(spatial_size=(70, 50), patch_size=(32, 32), stride=32)

        assert patcher.grid.n_v == 3
        assert patcher.grid.n_h == 2

    def test_padded_size_covers_all_patches(self):
        patcher = Patcher.build(spatial_size=(70, 50), patch_size=(32, 32), stride=32)
        grid    = patcher.grid
        ph, pw  = grid.patch_size

        expected_h = ph + (grid.n_v - 1) * grid.stride
        expected_w = pw + (grid.n_h - 1) * grid.stride

        assert grid.padded_size == (expected_h, expected_w)

    def test_constant_mode_when_padding_not_reflective(self):
        patcher = Patcher.build(spatial_size=(10, 10), patch_size=(16, 16), stride=8, use_reflective_padding=False)
        _, _, _, _, pw_spec = patcher._patch_coords[0]

        assert pw_spec is not None
        assert pw_spec[-1] == "constant"

    def test_symmetric_mode_default(self):
        patcher = Patcher.build(spatial_size=(10, 10), patch_size=(16, 16), stride=8)
        _, _, _, _, pw_spec = patcher._patch_coords[0]

        assert pw_spec[-1] == "symmetric"


class TestPatcherExtract:
    def test_extract_returns_patch_size(self):
        patcher = Patcher.build(spatial_size=(64, 64), patch_size=(32, 32), stride=32)
        array   = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)

        patch = patcher.extract(array, 0)

        assert patch.shape == (32, 32)

    def test_extract_preserves_leading_dimensions(self):
        patcher = Patcher.build(spatial_size=(64, 64), patch_size=(32, 32), stride=32)
        array   = np.zeros((5, 64, 64), dtype=np.float32)

        patch = patcher.extract(array, 0)

        assert patch.shape == (5, 32, 32)

    def test_extract_with_padding_returns_full_patch_size(self):
        patcher = Patcher.build(spatial_size=(10, 10), patch_size=(16, 16), stride=8)
        array   = np.ones((10, 10), dtype=np.float32)

        patch = patcher.extract(array, 0)

        assert patch.shape == (16, 16)

    def test_extract_all_patches_cover_grid(self):
        patcher = Patcher.build(spatial_size=(64, 64), patch_size=(32, 32), stride=32)
        array   = np.zeros((64, 64), dtype=np.float32)

        for idx in range(patcher.grid.number_of_patches):
            patch = patcher.extract(array, idx)
            assert patch.shape == (32, 32)

    def test_extract_values_match_source_no_padding(self):
        patcher = Patcher.build(spatial_size=(32, 32), patch_size=(32, 32), stride=32)
        array   = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)

        patch = patcher.extract(array, 0)

        assert np.array_equal(patch, array)


class TestGridInfo:
    def test_properties(self):
        grid = GridInfo(
            n_v=2, n_h=3, pad_top=1, pad_bot=2, pad_left=3, pad_right=4,
            patch_size=(16, 16), stride=8, spatial_size=(20, 30),
        )

        assert grid.padding_vertical   == 3
        assert grid.padding_horizontal == 7
        assert grid.number_of_patches  == 6
        assert grid.padded_size        == (23, 37)

    def test_as_dict_round_trip_fields(self):
        grid = GridInfo(
            n_v=1, n_h=1, pad_top=0, pad_bot=0, pad_left=0, pad_right=0,
            patch_size=(16, 16), stride=8, spatial_size=(16, 16),
        )

        payload = grid.as_dict()

        assert payload["n_v"] == 1
        assert payload["patch_size"] == [16, 16]
        assert payload["spatial_size"] == [16, 16]
        assert payload["number_of_patches"] == 1
        assert payload["use_reflective_padding"] is True


class TestLayout:
    def _write_dataset_json(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "global_crop"   : [0, 100, 0, 200],
            "dataset_type"  : "FSAR",
            "tomogram_tag"  : "tomo_tag",
            "parameter_tag" : "param_tag",
            "artifacts"     : {"primary_reduced": "primary.npy", "dem_reduced": "dem.npy"},
        }
        with open(data_dir / "dataset.json", "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return data_dir

    def test_layout_parses_fields(self, tmp_path):
        self._write_dataset_json(tmp_path)
        params = tmp_path / "params.npy"

        layout = Layout(tmp_path, logger=_null_logger(), parameters_path=params)

        assert layout.global_crop.as_tuple() == (0, 100, 0, 200)
        assert layout.dataset_type  == "FSAR"
        assert layout.tomogram_tag  == "tomo_tag"
        assert layout.parameter_tag == "param_tag"

    def test_artifact_path_parameters_returns_parameters_path(self, tmp_path):
        self._write_dataset_json(tmp_path)
        params = tmp_path / "params.npy"

        layout = Layout(tmp_path, logger=_null_logger(), parameters_path=params)

        assert layout.artifact_path("parameters") == params

    def test_artifact_path_named_artifact(self, tmp_path):
        data_dir = self._write_dataset_json(tmp_path)
        params   = tmp_path / "params.npy"

        layout = Layout(tmp_path, logger=_null_logger(), parameters_path=params)

        assert layout.artifact_path("primary_reduced") == data_dir / "primary.npy"

    def test_artifact_path_unknown_key_raises(self, tmp_path):
        self._write_dataset_json(tmp_path)
        layout = Layout(tmp_path, logger=_null_logger(), parameters_path=tmp_path / "params.npy")

        with pytest.raises(KeyError):
            layout.artifact_path("does_not_exist")


class TestCropper:
    def _make_layout(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "global_crop"   : [0, 40, 0, 40],
            "dataset_type"  : "FSAR",
            "tomogram_tag"  : "t",
            "parameter_tag" : "p",
            "artifacts"     : {
                "primary_reduced"        : "primary.npy",
                "secondaries_reduced"    : "secondaries.npy",
                "interferograms_reduced" : "interferograms.npy",
                "dem_reduced"            : "dem.npy",
            },
        }
        with open(data_dir / "dataset.json", "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return data_dir

    def _write_artifacts(self, data_dir, n_sec=2, n_ifg=3, az=40, rg=40):
        np.save(data_dir / "primary.npy",        np.ones((1, az, rg), dtype=np.complex64))
        np.save(data_dir / "secondaries.npy",    np.ones((n_sec, az, rg), dtype=np.complex64))
        np.save(data_dir / "interferograms.npy", np.ones((n_ifg, az, rg), dtype=np.complex64))
        np.save(data_dir / "dem.npy",            np.zeros((az, rg), dtype=np.float32))
        params = np.zeros((3, az, rg), dtype=np.float32)
        np.save(data_dir / "params.npy",         params)

    def test_to_local_slices(self, tmp_path):
        from tools.regions import SplitRegions
        self._make_layout(tmp_path)
        layout  = Layout(tmp_path, logger=_null_logger(), parameters_path=tmp_path / "data" / "params.npy")
        region  = CropRegion(10, 30, 5, 25)
        splits  = SplitRegions(train=region, val=region, test=region)
        cropper = Cropper(layout, splits, logger=_null_logger())

        az_slice, rg_slice = cropper.to_local_slices(region)

        assert az_slice == slice(10, 30)
        assert rg_slice == slice(5, 25)

    def test_load_split_stacks_inputs(self, tmp_path):
        from tools.regions import SplitRegions
        data_dir = self._make_layout(tmp_path)
        self._write_artifacts(data_dir, n_sec=2, n_ifg=3)

        layout  = Layout(tmp_path, logger=_null_logger(), parameters_path=data_dir / "params.npy")
        region  = CropRegion(0, 16, 0, 16)
        splits  = SplitRegions(train=region, val=region, test=region)
        cropper = Cropper(layout, splits, logger=_null_logger())

        out = cropper.load_split(region)

        assert out["inputs"].shape == (1 + 2 + 3, 16, 16)
        assert out["n_secondaries"]    == 2
        assert out["n_interferograms"] == 3
        assert out["dem"].shape        == (16, 16)
        assert out["parameters"].shape == (3, 16, 16)


class TestSpatialAugmenter:
    def _config(self, **overrides):
        base = dict(p_flip_h=0.0, p_flip_v=0.0, p_rot90=0.0, p_noise=0.0, noise_std=0.0)
        base.update(overrides)
        return AugmentationConfig(**base)

    def test_identity_when_all_probabilities_zero(self):
        aug   = SpatialAugmenter(self._config(), _null_logger())
        inp   = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
        gt    = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)

        out_inp, out_gt = aug(inp.copy(), gt.copy())

        assert np.array_equal(out_inp, inp)
        assert np.array_equal(out_gt, gt)

    def test_horizontal_flip_when_probability_one(self):
        aug   = SpatialAugmenter(self._config(p_flip_h=1.0), _null_logger())
        inp   = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
        gt    = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)

        out_inp, _ = aug(inp.copy(), gt.copy())

        assert np.array_equal(out_inp, inp[..., ::-1])

    def test_vertical_flip_when_probability_one(self):
        aug   = SpatialAugmenter(self._config(p_flip_v=1.0), _null_logger())
        inp   = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
        gt    = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)

        out_inp, _ = aug(inp.copy(), gt.copy())

        assert np.array_equal(out_inp, inp[..., ::-1, :])

    def test_noise_added_when_probability_one(self):
        aug = SpatialAugmenter(self._config(p_noise=1.0, noise_std=1.0), _null_logger())
        aug._rng = np.random.default_rng(0)
        inp = np.zeros((1, 4, 4), dtype=np.float32)
        gt  = np.zeros((1, 4, 4), dtype=np.float32)

        out_inp, _ = aug(inp.copy(), gt.copy())

        assert not np.array_equal(out_inp, inp)
        assert out_inp.dtype == np.float32

    def test_output_is_contiguous(self):
        aug = SpatialAugmenter(self._config(p_flip_h=1.0, p_flip_v=1.0), _null_logger())
        inp = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
        gt  = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)

        out_inp, out_gt = aug(inp.copy(), gt.copy())

        assert out_inp.flags["C_CONTIGUOUS"]
        assert out_gt.flags["C_CONTIGUOUS"]

    def test_rotation_changes_shape_for_non_square_unaffected(self):
        aug = SpatialAugmenter(self._config(p_rot90=1.0), _null_logger())
        aug._rng = np.random.default_rng(0)
        inp = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
        gt  = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)

        out_inp, out_gt = aug(inp.copy(), gt.copy())

        assert out_inp.shape == (1, 4, 4)
        assert out_gt.shape  == (1, 4, 4)


class TestStats:
    def _stats(self):
        ch_in  = _input_stats(2)
        ch_out = ChannelStats(loc=[0.0], scale=[1.0], names=["G1_amp"], strategies=[Presets.MIN_MAX])
        return Stats(input_stats=ch_in, output_stats=ch_out)

    def test_save_and_load_round_trip(self, tmp_path):
        stats = self._stats()
        out_path = stats.save(tmp_path)

        assert out_path.exists()
        loaded = Stats.load(tmp_path, logger=_null_logger())

        assert loaded.input_stats.n_channels  == 2
        assert loaded.output_stats.n_channels == 1
        assert loaded.input_stats.loc   == stats.input_stats.loc
        assert loaded.input_stats.scale == stats.input_stats.scale

    def test_save_writes_expected_filename(self, tmp_path):
        stats    = self._stats()
        out_path = stats.save(tmp_path)

        assert out_path.name == "normalization_stats.json"

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Stats.load(tmp_path, logger=_null_logger())

    def test_merge_combines_input_and_output(self):
        input_only  = Stats(input_stats=_input_stats(3), output_stats=None)
        out_stats   = ChannelStats(loc=[1.0], scale=[2.0], names=["x"], strategies=[Presets.MIN_MAX])
        output_only = Stats(input_stats=None, output_stats=out_stats)

        merged = Stats.merge(input_only, output_only)

        assert merged.input_stats.n_channels  == 3
        assert merged.output_stats.n_channels == 1


class TestStatsComputerHelpers:
    def test_compact_ranges_single(self):
        assert StatsComputer._compact_ranges([5]) == "5"

    def test_compact_ranges_contiguous(self):
        assert StatsComputer._compact_ranges([0, 1, 2, 3]) == "0-3"

    def test_compact_ranges_mixed(self):
        assert StatsComputer._compact_ranges([0, 1, 2, 5, 7, 8]) == "0-2, 5, 7-8"

    def test_compact_ranges_truncates_with_ellipsis(self):
        indices = [0, 2, 4, 6, 8, 10, 12, 14]
        out     = StatsComputer._compact_ranges(indices, max_items=3)

        assert out.endswith("...")
        assert out.count(",") == 3

    def test_get_subset_uses_all_when_max_zero(self):
        dataset = list(range(10))
        subset, n_use, n_total = StatsComputer._get_subset(dataset, max_samples=0)

        assert n_use   == 10
        assert n_total == 10
        assert subset is dataset

    def test_get_subset_limits_samples(self):
        dataset = list(range(20))
        subset, n_use, n_total = StatsComputer._get_subset(dataset, max_samples=5)

        assert n_use   == 5
        assert n_total == 20
        assert len(subset) == 5

    def test_get_subset_deterministic(self):
        dataset = list(range(20))
        s1, _, _ = StatsComputer._get_subset(dataset, max_samples=5)
        s2, _, _ = StatsComputer._get_subset(dataset, max_samples=5)

        assert [s1[i] for i in range(5)] == [s2[i] for i in range(5)]


class _ArrayDataset:
    def __init__(self, tensors):
        self._tensors = tensors

    def __len__(self):
        return len(self._tensors)

    def __getitem__(self, idx):
        return self._tensors[idx], np.zeros((1,), dtype=np.float32)


class TestStatsComputerCompute:
    def test_compute_input_stats_zscore(self):
        rng     = np.random.default_rng(0)
        tensors = [rng.normal(5.0, 2.0, (1, 4, 4)).astype(np.float32) for _ in range(8)]
        dataset = _ArrayDataset(tensors)

        input_config = InputConfig(
            use_primary=True, primary_representation=Representation.MAG_ONLY,
            use_secondaries=False, use_interferograms=False, use_dem=False,
        )

        stats = StatsComputer.compute_input_stats(
            dataset          = dataset,
            logger           = _null_logger(),
            input_config     = input_config,
            n_secondaries    = 0,
            n_interferograms = 0,
            num_workers      = 0,
            batch_size       = 4,
        )

        assert stats.input_stats is not None
        assert stats.output_stats is None
        assert stats.input_stats.n_channels == 1

    def test_compute_output_stats_from_params(self, tmp_path):
        rng    = np.random.default_rng(1)
        params = np.empty((3, 8, 8), dtype=np.float32)
        params[0] = rng.uniform(0.0, 1.0, (8, 8))
        params[1] = rng.uniform(-10.0, 10.0, (8, 8))
        params[2] = rng.uniform(0.5, 5.0, (8, 8))

        params_path = tmp_path / "params.npy"
        np.save(params_path, params)

        stats = StatsComputer.compute_output_stats(
            params_path   = params_path,
            n_gaussians   = 1,
            output_config = OutputConfig(),
            logger        = None,
        )

        assert stats.output_stats is not None
        assert stats.input_stats is None
        assert stats.output_stats.n_channels == 3

    def test_compute_output_stats_channel_naming(self, tmp_path):
        params = np.ones((6, 4, 4), dtype=np.float32)
        params_path = tmp_path / "p.npy"
        np.save(params_path, params)

        stats = StatsComputer.compute_output_stats(
            params_path   = params_path,
            n_gaussians   = 2,
            output_config = OutputConfig(),
            logger        = None,
        )

        assert stats.output_stats.n_channels == 6
        assert stats.output_stats.names[0].startswith("G1_")
        assert stats.output_stats.names[3].startswith("G2_")


class TestNormalizer:
    def test_zscore_identity_with_unit_scale(self):
        normalizer = Normalizer(Stats(input_stats=_input_stats(2), output_stats=None))
        tensor     = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)

        out = normalizer.normalize_input(tensor)

        assert np.allclose(out, tensor)

    def test_zscore_subtracts_loc_divides_scale(self):
        stats      = ChannelStats(loc=[2.0], scale=[4.0], names=["c"], strategies=[Presets.ZSCORE])
        normalizer = Normalizer(Stats(input_stats=stats, output_stats=None))
        tensor     = np.full((1, 2, 2), 10.0, dtype=np.float32)

        out = normalizer.normalize_input(tensor)

        assert np.allclose(out, (10.0 - 2.0) / 4.0)

    def test_normalize_denormalize_round_trip_numpy(self):
        stats      = ChannelStats(loc=[1.5], scale=[3.0], names=["c"], strategies=[Presets.ZSCORE])
        normalizer = Normalizer(Stats(input_stats=stats, output_stats=None))
        tensor     = np.random.default_rng(0).normal(0.0, 5.0, (1, 4, 4)).astype(np.float32)

        out = normalizer.normalize_input(tensor)
        rec = normalizer.denormalize_input(out)

        assert np.allclose(rec, tensor, atol=1e-4)

    def test_round_trip_with_log1p(self):
        stats      = ChannelStats(loc=[0.0], scale=[1.0], names=["c"], strategies=[Presets.ZSCORE_LOG1P])
        normalizer = Normalizer(Stats(input_stats=stats, output_stats=None))
        tensor     = np.abs(np.random.default_rng(0).normal(2.0, 1.0, (1, 4, 4))).astype(np.float32)

        out = normalizer.normalize_input(tensor)
        rec = normalizer.denormalize_input(out)

        assert np.allclose(rec, tensor, atol=1e-3)

    def test_log1p_clamps_negative_inputs(self):
        stats      = ChannelStats(loc=[0.0], scale=[1.0], names=["c"], strategies=[Presets.ZSCORE_LOG1P])
        normalizer = Normalizer(Stats(input_stats=stats, output_stats=None))
        tensor     = np.full((1, 2, 2), -5.0, dtype=np.float32)

        out = normalizer.normalize_input(tensor)

        assert np.all(np.isfinite(out))
        assert np.allclose(out, 0.0)

    def test_torch_and_numpy_match(self):
        stats        = ChannelStats(loc=[1.0], scale=[2.0], names=["c"], strategies=[Presets.ZSCORE])
        normalizer   = Normalizer(Stats(input_stats=stats, output_stats=None))
        np_tensor    = np.random.default_rng(0).normal(0.0, 3.0, (1, 4, 4)).astype(np.float32)
        torch_tensor = torch.from_numpy(np_tensor.copy()).to("cpu")

        out_np    = normalizer.normalize_input(np_tensor)
        out_torch = normalizer.normalize_input(torch_tensor)

        assert isinstance(out_torch, torch.Tensor)
        assert out_torch.device.type == "cpu"
        assert np.allclose(out_np, out_torch.numpy(), atol=1e-5)

    def test_four_dim_batched_input(self):
        normalizer = Normalizer(Stats(input_stats=_input_stats(2), output_stats=None))
        tensor     = np.zeros((3, 2, 4, 4), dtype=np.float32)

        out = normalizer.normalize_input(tensor)

        assert out.shape == (3, 2, 4, 4)

    def test_normalize_output_uses_output_stats(self):
        out_stats  = ChannelStats(loc=[5.0], scale=[1.0], names=["o"], strategies=[Presets.ZSCORE])
        normalizer = Normalizer(Stats(input_stats=None, output_stats=out_stats))
        tensor     = np.full((1, 2, 2), 5.0, dtype=np.float32)

        out = normalizer.normalize_output(tensor)

        assert np.allclose(out, 0.0)

    def test_channel_vectors_cached(self):
        stats      = _input_stats(2)
        normalizer = Normalizer(Stats(input_stats=stats, output_stats=None))

        v1 = normalizer._channel_vectors(stats)
        v2 = normalizer._channel_vectors(stats)

        assert v1 is v2

    def test_numpy_output_is_float32_contiguous(self):
        normalizer = Normalizer(Stats(input_stats=_input_stats(1), output_stats=None))
        tensor     = np.zeros((1, 4, 4), dtype=np.float64)

        out = normalizer.normalize_input(tensor)

        assert out.dtype == np.float32
        assert out.flags["C_CONTIGUOUS"]


def _make_patch_dataset(n_sec=0, n_ifg=1, normalizer=None, augmenter=None, split_name="val"):
    az, rg = 8, 8
    n_passes = 1 + n_sec + n_ifg
    inputs   = np.ones((n_passes, az, rg), dtype=np.complex64)
    params   = np.zeros((3, az, rg), dtype=np.float32)
    params[0] = 0.5
    params[1] = 1.0
    params[2] = 2.0

    grid = Patcher.build(spatial_size=(az, rg), patch_size=(az, rg), stride=az)

    input_config = InputConfig(
        use_primary=True, primary_representation=Representation.MAG_ONLY,
        use_secondaries=(n_sec > 0), secondaries_representation=Representation.MAG_ONLY,
        use_interferograms=(n_ifg > 0), interferograms_representation=Representation.ANGLE_ONLY,
        use_dem=False,
    )

    return PatchDataset(
        inputs           = inputs,
        gt_parameters    = params,
        grid             = grid,
        input_config     = input_config,
        output_config    = OutputConfig(),
        split_name       = split_name,
        n_secondaries    = n_sec,
        n_interferograms = n_ifg,
        normalizer       = normalizer,
        x_axis           = None,
        n_gaussians      = 1,
        augmenter        = augmenter,
        dem              = None,
    )


class TestPatchDataset:
    def test_len_matches_grid(self):
        ds = _make_patch_dataset()
        assert len(ds) == ds.grid.grid.number_of_patches

    def test_input_channel_count(self):
        ds = _make_patch_dataset(n_sec=0, n_ifg=1)
        assert ds.input_channels == 1 + 1

    def test_gt_channel_count(self):
        ds = _make_patch_dataset()
        assert ds.gt_channels == 3

    def test_getitem_shapes(self):
        ds = _make_patch_dataset(n_sec=0, n_ifg=1)
        inp, gt = ds[0]

        assert inp.shape == (2, 8, 8)
        assert gt.shape  == (3, 8, 8)
        assert inp.dtype == np.float32
        assert gt.dtype  == np.float32

    def test_getitem_no_normalizer_passes_through(self):
        ds = _make_patch_dataset()
        inp, gt = ds[0]

        assert np.all(gt[0] == 0.5)

    def test_getitem_applies_normalizer(self):
        stats      = _input_stats(2, method=NormMethod.ZSCORE)
        stats.loc  = [1.0, 1.0]
        normalizer = Normalizer(Stats(input_stats=stats, output_stats=None))
        ds         = _make_patch_dataset(n_sec=0, n_ifg=1, normalizer=normalizer)

        inp, _ = ds[0]

        assert inp.shape == (2, 8, 8)

    def test_layer_count_mismatch_raises(self):
        inputs = np.ones((2, 8, 8), dtype=np.complex64)
        params = np.zeros((3, 8, 8), dtype=np.float32)
        grid   = Patcher.build(spatial_size=(8, 8), patch_size=(8, 8), stride=8)

        with pytest.raises(ValueError, match="layers"):
            PatchDataset(
                inputs=inputs, gt_parameters=params, grid=grid,
                input_config=InputConfig(), output_config=OutputConfig(),
                split_name="val", n_secondaries=0, n_interferograms=0,
            )

    def test_use_secondaries_without_secondaries_raises(self):
        inputs = np.ones((2, 8, 8), dtype=np.complex64)
        params = np.zeros((3, 8, 8), dtype=np.float32)
        grid   = Patcher.build(spatial_size=(8, 8), patch_size=(8, 8), stride=8)
        cfg    = InputConfig(use_primary=True, use_secondaries=True, use_interferograms=True)

        with pytest.raises(ValueError, match="secondaries"):
            PatchDataset(
                inputs=inputs, gt_parameters=params, grid=grid,
                input_config=cfg, output_config=OutputConfig(),
                split_name="val", n_secondaries=0, n_interferograms=1,
            )

    def test_use_interferograms_without_interferograms_raises(self):
        inputs = np.ones((2, 8, 8), dtype=np.complex64)
        params = np.zeros((3, 8, 8), dtype=np.float32)
        grid   = Patcher.build(spatial_size=(8, 8), patch_size=(8, 8), stride=8)
        cfg    = InputConfig(use_primary=True, use_secondaries=True, use_interferograms=True)

        with pytest.raises(ValueError, match="interferograms"):
            PatchDataset(
                inputs=inputs, gt_parameters=params, grid=grid,
                input_config=cfg, output_config=OutputConfig(),
                split_name="val", n_secondaries=1, n_interferograms=0,
            )

    def test_augmenter_only_applied_on_train_split(self):
        aug = SpatialAugmenter(
            AugmentationConfig(p_flip_h=1.0, p_flip_v=1.0, p_rot90=0.0, p_noise=0.0, noise_std=0.0),
            _null_logger(),
        )
        ds_val   = _make_patch_dataset(augmenter=aug, split_name="val")
        ds_train = _make_patch_dataset(augmenter=aug, split_name="train")

        inp_val, _   = ds_val[0]
        inp_train, _ = ds_train[0]

        assert inp_val.shape == inp_train.shape


class TestMultiRegionDataset:
    def test_requires_at_least_one_part(self):
        with pytest.raises(ValueError, match="at least one"):
            MultiRegionDataset([])

    def test_length_is_sum_of_parts(self):
        p1 = _make_patch_dataset()
        p2 = _make_patch_dataset()
        ds = MultiRegionDataset([p1, p2])

        assert len(ds) == len(p1) + len(p2)

    def test_inherits_metadata_from_first_part(self):
        p1 = _make_patch_dataset(n_sec=0, n_ifg=1)
        p2 = _make_patch_dataset(n_sec=0, n_ifg=1)
        ds = MultiRegionDataset([p1, p2])

        assert ds.n_secondaries    == p1.n_secondaries
        assert ds.n_interferograms == p1.n_interferograms
        assert ds.input_channels   == p1.input_channels
        assert ds.gt_channels      == p1.gt_channels

    def test_getitem_routes_to_correct_part(self):
        p1 = _make_patch_dataset()
        p2 = _make_patch_dataset()
        ds = MultiRegionDataset([p1, p2])

        first = ds[0]
        last  = ds[len(ds) - 1]

        assert first[0].shape == last[0].shape

    def test_negative_index_supported(self):
        p1 = _make_patch_dataset()
        p2 = _make_patch_dataset()
        ds = MultiRegionDataset([p1, p2])

        direct  = ds[len(ds) - 1]
        negative = ds[-1]

        assert np.array_equal(direct[0], negative[0])

    def test_out_of_range_raises(self):
        ds = MultiRegionDataset([_make_patch_dataset()])

        with pytest.raises(IndexError):
            _ = ds[len(ds)]

    def test_normalizer_setter_propagates(self):
        p1 = _make_patch_dataset(n_sec=0, n_ifg=1)
        p2 = _make_patch_dataset(n_sec=0, n_ifg=1)
        ds = MultiRegionDataset([p1, p2])

        normalizer = Normalizer(Stats(input_stats=_input_stats(2), output_stats=None))
        ds.normalizer = normalizer

        assert p1.normalizer is normalizer
        assert p2.normalizer is normalizer
        assert ds.normalizer is normalizer


class TestLoader:
    def test_build_returns_three_loaders(self):
        train = _make_patch_dataset()
        val   = _make_patch_dataset()
        test  = _make_patch_dataset()

        train_loader, val_loader, test_loader = Loader.build(
            train_dataset=train, val_dataset=val, test_dataset=test,
            batch_size=1, num_workers=0, logger=_null_logger(),
            pin_memory=False, shuffle_train=False,
        )

        from torch.utils.data import DataLoader
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

    def test_train_loader_drops_last(self):
        train = MultiRegionDataset([_make_patch_dataset() for _ in range(3)])
        val   = _make_patch_dataset()
        test  = _make_patch_dataset()

        train_loader, _, _ = Loader.build(
            train_dataset=train, val_dataset=val, test_dataset=test,
            batch_size=2, num_workers=0, logger=_null_logger(),
            pin_memory=False, shuffle_train=False,
        )

        assert train_loader.drop_last is True

    def test_loader_yields_batches(self):
        train = MultiRegionDataset([_make_patch_dataset(n_sec=0, n_ifg=1) for _ in range(4)])
        val   = _make_patch_dataset(n_sec=0, n_ifg=1)
        test  = _make_patch_dataset(n_sec=0, n_ifg=1)

        train_loader, _, _ = Loader.build(
            train_dataset=train, val_dataset=val, test_dataset=test,
            batch_size=2, num_workers=0, logger=_null_logger(),
            pin_memory=False, shuffle_train=False,
        )

        batch = next(iter(train_loader))
        inp, gt = batch

        assert inp.shape[0] == 2
        assert inp.shape[1] == 2
        assert gt.shape[1]  == 3
