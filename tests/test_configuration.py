from __future__ import annotations

import dataclasses
import inspect
import json

import numpy as np
import pytest

import configuration.models_config as models_config
from configuration.benchmark_config import (
    BenchmarkConfig,
    BenchmarkPathsConfig,
    ComparisonReportConfig,
    InferenceQueueConfig,
    OverfitGateConfig,
    SizeMatchConfig,
    TrainingQueueConfig,
)
from configuration.cross_validation_config import CrossValidationConfig, FoldConfig
from configuration.dataset_config import (
    AugmentationConfig,
    DatasetConfiguration,
    InputConfig,
    OutputConfig,
    PatchConfiguration,
)
from configuration.inference_config import (
    InferenceConfig,
    InferenceEntryConfig,
    InferencePaths,
)
from configuration.norm_config import (
    ChannelStats,
    ChannelStrategy,
    NormMethod,
    Presets,
)
from configuration.param_extraction_config import (
    ExtractionConfig,
    ExtractParamsEntryConfig,
    FitMode,
    FitSettings,
)
from configuration.physics_check_config import PhysicsCheckEntryConfig
from configuration.processing_config import (
    ParallelConfiguration,
    PathConfiguration,
    PreProcessEntryConfig,
    ProcessingConfiguration,
    TomogramConfiguration,
)
from configuration.representation import Representation
from configuration.train_config import TrainEntryConfig
from configuration.training_config import (
    EarlyStoppingConfig,
    EMAConfig,
    GaussianConfig,
    GeometryConfig,
    GradientClipperConfig,
    IOConfig,
    LossConfig,
    LossCurriculumConfig,
    LossNormalizationConfig,
    MemoryConfig,
    OptimizerConfig,
    OverfitConfig,
    PermutationMetricsConfig,
    ResourceConfig,
    SchedulerConfig,
    TrainerConfig,
    TrainingConfigInner,
    WarmupConfig,
)
from configuration.tuning_config import (
    Phase1TuneConfig,
    Phase2TuneConfig,
    TuningConfig,
    TuningEntryConfig,
)
from tools.regions import CropRegion, SplitRegions


def _model_config_classes():
    classes = []
    for name in dir(models_config):
        obj = getattr(models_config, name)
        if inspect.isclass(obj) and dataclasses.is_dataclass(obj) and name.endswith("Config"):
            classes.append(obj)
    return classes


MODEL_CONFIG_CLASSES = _model_config_classes()


class TestBenchmarkConfig:
    def test_default_construction_round_trip(self):
        cfg = BenchmarkConfig()

        assert isinstance(cfg.paths,      BenchmarkPathsConfig)
        assert isinstance(cfg.overfit,    OverfitGateConfig)
        assert isinstance(cfg.size_match, SizeMatchConfig)
        assert isinstance(cfg.training,   TrainingQueueConfig)
        assert isinstance(cfg.inference,  InferenceQueueConfig)
        assert isinstance(cfg.comparison, ComparisonReportConfig)

    def test_default_field_values(self):
        cfg = BenchmarkConfig()

        assert cfg.resume         is True
        assert cfg.seed           == 0
        assert cfg.n_gaussians    == 5
        assert cfg.gpus           == [2, 3]
        assert cfg.skip_models    == []
        assert cfg.run_tag        is None
        assert cfg.poll_interval_s == 5.0

    def test_nested_defaults_are_independent_instances(self):
        a = BenchmarkConfig()
        b = BenchmarkConfig()

        a.gpus.append(99)
        a.skip_models.append("x")

        assert b.gpus        == [2, 3]
        assert b.skip_models == []

    def test_benchmark_paths_are_path_objects(self):
        paths = BenchmarkPathsConfig()

        assert isinstance(paths.dataset_path,    type(paths.dataset_path))
        assert str(paths.parameters_path).endswith(".npy")
        assert "logs" in str(paths.log_base_dir)

    def test_overfit_gate_defaults(self):
        gate = OverfitGateConfig()

        assert gate.max_steps           > 0
        assert gate.stop_threshold      > 0.0
        assert gate.batch_size          > 0
        assert gate.require_convergence is True
        assert gate.abort_on_fail       is True

    def test_size_match_scale_bounds_ordered(self):
        sm = SizeMatchConfig()

        assert sm.scale_low < sm.scale_high
        assert 0.0 < sm.tolerance < 1.0
        assert sm.max_iterations > 0

    def test_training_queue_azimuth_ranges_ordered(self):
        tq = TrainingQueueConfig()

        assert tq.train_azimuth[0] < tq.train_azimuth[1]
        assert tq.val_azimuth[0]   < tq.val_azimuth[1]
        assert tq.test_azimuth[0]  < tq.test_azimuth[1]
        assert tq.train_azimuth[1] == tq.val_azimuth[0]
        assert tq.val_azimuth[1]   == tq.test_azimuth[0]

    def test_training_queue_patch_size_is_pair(self):
        tq = TrainingQueueConfig()

        assert len(tq.patch_size) == 2
        assert tq.patch_stride <= tq.patch_size[0]

    def test_inference_queue_profile_counts_non_negative(self):
        iq = InferenceQueueConfig()

        assert iq.n_best_profiles    >= 0
        assert iq.n_worst_profiles   >= 0
        assert iq.n_random_profiles  >= 0
        assert iq.gif_axes           == ["elevation"]
        assert iq.batch_size         is None


class TestCrossValidationConfig:
    def test_default_construction(self):
        cfg = CrossValidationConfig()

        assert cfg.model_name       == "resunet"
        assert cfg.model_overrides  == {}
        assert isinstance(cfg.folds, FoldConfig)
        assert cfg.inference_splits == ["val", "test"]

    def test_log_dir_override_applied(self):
        cfg = CrossValidationConfig()

        assert "cross_validation" in str(cfg.paths.log_base_dir)

    def test_fold_config_range_ordered(self):
        fold = FoldConfig()

        assert fold.n_folds > 1
        assert fold.azimuth_start < fold.azimuth_end

    def test_nested_lists_independent(self):
        a = CrossValidationConfig()
        b = CrossValidationConfig()

        a.inference_splits.append("train")
        a.gpus.append(7)

        assert b.inference_splits == ["val", "test"]
        assert b.gpus             == [2, 3]


class TestInputConfig:
    def test_default_channels_per_pass(self):
        cfg = InputConfig()

        assert cfg.primary_channels_per_pass        == 1
        assert cfg.secondaries_channels_per_pass    == 0
        assert cfg.interferograms_channels_per_pass == 1

    def test_disabled_inputs_contribute_zero(self):
        cfg = InputConfig(use_primary=False, use_secondaries=False, use_interferograms=False)

        assert cfg.primary_channels_per_pass        == 0
        assert cfg.secondaries_channels_per_pass    == 0
        assert cfg.interferograms_channels_per_pass == 0
        assert cfg.total_channels(3, 4)             == 0

    def test_total_channels_default(self):
        cfg = InputConfig()

        assert cfg.total_channels(2, 3) == 1 + 0 + 3 * 1

    def test_total_channels_with_dem(self):
        cfg = InputConfig(use_dem=True)

        assert cfg.total_channels(0, 0) == 1 + 1

    def test_total_channels_with_secondaries(self):
        cfg = InputConfig(use_secondaries=True)

        assert cfg.total_channels(2, 1) == 1 + 2 * 1 + 1 * 1

    def test_channel_group_keys_default(self):
        cfg  = InputConfig()
        keys = cfg.channel_group_keys(2, 3)

        assert keys == ["pass/mag", "ifg/phase", "ifg/phase", "ifg/phase"]

    def test_channel_group_keys_with_dem(self):
        cfg  = InputConfig(use_dem=True)
        keys = cfg.channel_group_keys(0, 0)

        assert keys[-1] == "dem/elevation"

    def test_channel_group_keys_length_matches_total_channels(self):
        cfg = InputConfig(use_secondaries=True, use_dem=True)

        assert len(cfg.channel_group_keys(2, 3)) == cfg.total_channels(2, 3)

    def test_as_dict_from_dict_round_trip_flat(self):
        cfg = InputConfig(use_secondaries=True, use_dem=True)

        assert InputConfig.from_dict(cfg.as_dict()) == cfg

    def test_from_dict_nested_layout(self):
        payload = {
            "primary"        : {"use": True,  "representation": Representation.MAG_ANGLE.value},
            "secondaries"    : {"use": False, "representation": Representation.MAG_ONLY.value},
            "interferograms" : {"use": True,  "representation": Representation.ANGLE_ONLY.value},
            "dem"            : {"use": True},
        }
        cfg = InputConfig.from_dict(payload)

        assert cfg.use_primary             is True
        assert cfg.primary_representation  is Representation.MAG_ANGLE
        assert cfg.use_dem                 is True


class TestOutputConfig:
    def test_role_names_default(self):
        cfg = OutputConfig()

        assert cfg.role_names          == ["a", "mu", "sig"]
        assert cfg.params_per_gaussian == 3

    def test_role_names_subset(self):
        cfg = OutputConfig(use_mu=False)

        assert cfg.role_names          == ["a", "sig"]
        assert cfg.params_per_gaussian == 2

    def test_selected_indices_full(self):
        cfg = OutputConfig()

        assert cfg.selected_indices(2) == [0, 1, 2, 3, 4, 5]

    def test_selected_indices_subset(self):
        cfg = OutputConfig(use_mu=False)

        assert cfg.selected_indices(2) == [0, 2, 3, 5]

    def test_selected_indices_empty_for_zero_gaussians(self):
        cfg = OutputConfig()

        assert cfg.selected_indices(0) == []

    def test_total_channels(self):
        cfg = OutputConfig()

        assert cfg.total_channels(5) == 15

    def test_strategy_for_known_and_unknown(self):
        cfg = OutputConfig()

        known   = cfg.strategy_for("out/amp")
        unknown = cfg.strategy_for("out/mu")

        assert isinstance(known,   ChannelStrategy)
        assert isinstance(unknown, ChannelStrategy)

    def test_as_dict_from_dict_round_trip(self):
        cfg = OutputConfig(use_amplitude=True, use_mu=False, use_sigma=True)
        rt  = OutputConfig.from_dict(cfg.as_dict())

        assert rt.use_amplitude == cfg.use_amplitude
        assert rt.use_mu        == cfg.use_mu
        assert rt.use_sigma     == cfg.use_sigma

    def test_from_dict_nested_layout(self):
        payload = {
            "amplitude" : {"use": True},
            "mu"        : {"use": False},
            "sigma"     : {"use": True},
        }
        cfg = OutputConfig.from_dict(payload)

        assert cfg.use_amplitude is True
        assert cfg.use_mu        is False
        assert cfg.use_sigma     is True

    def test_default_strategies_are_independent(self):
        a = OutputConfig()
        b = OutputConfig()

        assert a.output_strategies is not b.output_strategies


class TestDatasetConfiguration:
    def _split_regions(self):
        crop = CropRegion(1000, 2000, 500, 1000)
        return SplitRegions(train=crop, val=crop, test=crop)

    def test_minimal_construction_defaults(self):
        cfg = DatasetConfiguration(
            preprocessing_run_directory = "/tmp/run",
            split_regions               = self._split_regions(),
        )

        assert isinstance(cfg.patch,         PatchConfiguration)
        assert isinstance(cfg.input_config,  InputConfig)
        assert isinstance(cfg.output_config, OutputConfig)
        assert isinstance(cfg.augmentation,  AugmentationConfig)
        assert cfg.batch_size  == 8
        assert cfg.n_gaussians == 1
        assert cfg.x_axis      is None

    def test_patch_defaults(self):
        patch = PatchConfiguration()

        assert patch.size                   == (64, 64)
        assert patch.stride                 == 32
        assert patch.use_reflective_padding is True

    def test_augmentation_probabilities_in_unit_interval(self):
        aug = AugmentationConfig()

        for p in (aug.p_flip_h, aug.p_flip_v, aug.p_rot90, aug.p_amp_scale, aug.p_noise):
            assert 0.0 <= p <= 1.0
        assert aug.amp_scale_range[0] < aug.amp_scale_range[1]
        assert aug.noise_std >= 0.0


class TestInferenceConfig:
    def test_default_construction(self):
        cfg = InferenceConfig(run_directory="/tmp/run")

        assert cfg.device          == "cuda"
        assert cfg.split           == "test"
        assert cfg.use_ema         is True
        assert cfg.checkpoint_name == "best_model.pt"
        assert isinstance(cfg.paths, InferencePaths)

    def test_paths_filenames(self):
        paths = InferencePaths()

        assert paths.metrics_filename.endswith(".json")
        assert paths.report_filename.endswith(".md")

    def test_dpi_values_positive(self):
        cfg = InferenceConfig(run_directory="/tmp/run")

        assert cfg.fig_dpi   > 0
        assert cfg.save_dpi  > 0
        assert cfg.gif_dpi   > 0
        assert cfg.gif_fps   > 0

    def test_gif_axes_default_independent(self):
        a = InferenceConfig(run_directory="/tmp/a")
        b = InferenceConfig(run_directory="/tmp/b")

        a.gif_axes.append("range")

        assert b.gif_axes == ["elevation"]

    def test_entry_config_defaults(self):
        cfg = InferenceEntryConfig()

        assert isinstance(cfg.inference, InferenceConfig)
        assert cfg.inference.gif_axes == ["elevation", "range", "azimuth"]
        assert cfg.inference.cpu_workers == 16
        assert cfg.gpu == 0
        assert cfg.run_filter == []


class TestNormConfig:
    def test_norm_method_values(self):
        assert NormMethod.MIN_MAX_P999.value == "min_max_p999"
        assert NormMethod.ROBUST_IQR.value   == "robust_iqr"
        assert NormMethod.FIXED_DIV_PI.value == "fixed_div_pi"
        assert NormMethod.ZSCORE.value       == "zscore"

    def test_fit_fixed_div_pi_ignores_data(self):
        cs = ChannelStrategy(NormMethod.FIXED_DIV_PI)

        lo, scale = cs.fit(np.array([100.0, -50.0, 0.0]))

        assert lo    == 0.0
        assert np.isclose(scale, np.pi)

    def test_fit_zscore_matches_mean_std(self):
        cs   = ChannelStrategy(NormMethod.ZSCORE)
        data = np.array([0.0, 2.0, 4.0])

        loc, scale = cs.fit(data)

        assert np.isclose(loc,   data.mean())
        assert np.isclose(scale, data.std())

    def test_fit_robust_iqr_matches_median_iqr(self):
        cs   = ChannelStrategy(NormMethod.ROBUST_IQR)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        loc, scale = cs.fit(data)

        assert np.isclose(loc, 3.0)
        assert scale > 0.0

    def test_fit_min_max_returns_positive_scale(self):
        cs = ChannelStrategy(NormMethod.MIN_MAX_P999)

        loc, scale = cs.fit(np.array([1.0, 2.0, 3.0, 4.0]))

        assert scale >= 1e-8

    def test_fit_constant_data_scale_floored(self):
        for method in (NormMethod.MIN_MAX_P999, NormMethod.ROBUST_IQR, NormMethod.ZSCORE):
            cs         = ChannelStrategy(method)
            loc, scale = cs.fit(np.array([5.0, 5.0, 5.0, 5.0]))

            assert scale >= 1e-8

    def test_fit_log1p_applied(self):
        cs   = ChannelStrategy(NormMethod.ZSCORE, apply_log1p=True)
        data = np.array([0.0, 1.0, 3.0])

        loc, scale = cs.fit(data)
        expected   = np.log1p(np.maximum(data, 0.0))

        assert np.isclose(loc, expected.mean())

    def test_as_dict_from_dict_round_trip(self):
        cs = ChannelStrategy(NormMethod.ROBUST_IQR, apply_log1p=True)
        rt = ChannelStrategy.from_dict(cs.as_dict())

        assert rt.norm_method is cs.norm_method
        assert rt.apply_log1p == cs.apply_log1p

    def test_from_dict_legacy_strategy_key(self):
        cs = ChannelStrategy.from_dict({"strategy": NormMethod.MIN_MAX_P999.value})

        assert cs.norm_method is NormMethod.MIN_MAX_P999

    def test_from_dict_default_is_zscore(self):
        cs = ChannelStrategy.from_dict({})

        assert cs.norm_method is NormMethod.ZSCORE
        assert cs.apply_log1p is False

    def test_from_slot_known_keys(self):
        assert ChannelStrategy.from_slot("pass/phase").norm_method is NormMethod.FIXED_DIV_PI
        assert ChannelStrategy.from_slot("dem/elevation").norm_method is NormMethod.ZSCORE

    def test_from_slot_unknown_raises(self):
        with pytest.raises(KeyError):
            ChannelStrategy.from_slot("nonexistent/slot")

    def test_presets_distinct_instances(self):
        assert Presets.MIN_MAX.apply_log1p       is False
        assert Presets.MIN_MAX_LOG1P.apply_log1p is True
        assert Presets.ZSCORE.norm_method        is NormMethod.ZSCORE


class TestChannelStats:
    def test_n_channels(self):
        stats = ChannelStats(loc=[1.0, 2.0, 3.0], scale=[0.5, 0.5, 0.5])

        assert stats.n_channels == 3

    def test_as_dict_from_dict_round_trip(self):
        stats = ChannelStats(
            loc        = [1.0, 2.0],
            scale      = [0.5, 0.25],
            names      = ["a", "b"],
            strategies = [Presets.ZSCORE, Presets.MIN_MAX],
        )
        rt = ChannelStats.from_dict(stats.as_dict())

        assert rt.loc       == stats.loc
        assert rt.scale     == stats.scale
        assert rt.names     == stats.names
        assert rt.n_channels == stats.n_channels

    def test_from_dict_legacy_mean_std_keys(self):
        payload = {
            "channels": [
                {"name": "c0", "mean": 1.0, "std": 2.0, "norm_method": "zscore"},
            ],
            "log1p_channels": [0],
        }
        stats = ChannelStats.from_dict(payload)

        assert stats.loc            == [1.0]
        assert stats.scale          == [2.0]
        assert stats.log1p_channels == [0]

    def test_log1p_channels_default_empty(self):
        stats = ChannelStats(loc=[1.0], scale=[1.0])

        assert stats.log1p_channels == []


class TestRepresentation:
    @pytest.mark.parametrize("rep", list(Representation))
    def test_channels_per_pass_matches_slot_kinds(self, rep):
        assert rep.channels_per_pass == len(rep.slot_kinds)

    def test_value_membership(self):
        assert Representation("mag_only") is Representation.MAG_ONLY
        assert Representation("angle_only") is Representation.ANGLE_ONLY

    def test_convert_shape_and_dtype(self):
        data = (np.ones((2, 3, 4, 5)) + 1j * np.ones((2, 3, 4, 5))).astype(np.complex64)

        out = Representation.MAG_REAL_IMAG.convert(data)

        assert out.shape == (2, 3 * 3, 4, 5)
        assert out.dtype == np.float32

    def test_convert_mag_only_is_abs(self):
        data = (np.arange(2 * 1 * 2 * 2).reshape(2, 1, 2, 2) + 1j).astype(np.complex64)

        out = Representation.MAG_ONLY.convert(data)

        assert out.shape == (2, 1, 2, 2)
        assert np.allclose(out, np.abs(data))

    def test_convert_angle_only_is_angle(self):
        data = (np.ones((1, 1, 2, 2)) * (1 + 1j)).astype(np.complex64)

        out = Representation.ANGLE_ONLY.convert(data)

        assert np.allclose(out, np.angle(data), atol=1e-5)

    def test_convert_into_matches_convert(self):
        data    = (np.random.default_rng(0).standard_normal((3, 4, 5)) + 1j).astype(np.complex64)
        rep     = Representation.MAG_REAL_IMAG
        buffer  = np.empty((3 * rep.channels_per_pass, 4, 5), dtype=np.float32)

        rep.convert_into(buffer, data)
        reference = rep.convert(data[None])[0]

        assert np.allclose(buffer, reference)

    def test_convert_mag_real_imag_normalization_safe_at_zero(self):
        data = np.zeros((1, 1, 2, 2), dtype=np.complex64)

        out = Representation.MAG_REAL_IMAG.convert(data)

        assert np.all(np.isfinite(out))


class TestParamExtractionConfig:
    def test_fit_settings_derived_properties(self):
        fs = FitSettings()

        assert fs.number_of_gaussians    == fs.fit_config.k_max
        assert fs.parameters_per_profile == 3 * fs.fit_config.k_max
        assert fs.fitting_method         == "sigma_only_adam"

    def test_fit_mode_sigma_only_defaults(self):
        sigma = FitMode.SigmaOnly()

        assert sigma.k_max            == 5
        assert sigma.lambda_k         > 0.0
        assert 0.0 < sigma.threshold_factor < 1.0

    def test_output_suffix_value_derived(self):
        cfg = ExtractionConfig(processed_data_path="/tmp/data")

        assert cfg.output_suffix_value == "Ng5_sigonly_k5"
        assert cfg.output_subdir_name  == "params_Ng5_sigonly_k5"

    def test_output_suffix_value_explicit_override(self):
        cfg = ExtractionConfig(processed_data_path="/tmp/data", output_suffix="custom")

        assert cfg.output_suffix_value == "custom"

    def test_directory_properties(self, tmp_path):
        cfg = ExtractionConfig(processed_data_path=tmp_path)

        assert cfg.data_directory     == tmp_path / "data"
        assert cfg.metadata_directory == tmp_path / "meta"
        assert cfg.output_directory   == tmp_path / "params" / "params_Ng5_sigonly_k5"
        assert cfg.parameters_npy_path.name == "parameters_Ng5_sigonly_k5.npy"

    def test_discover_tomogram_path_none_when_missing(self, tmp_path):
        (tmp_path / "data").mkdir()
        cfg = ExtractionConfig(processed_data_path=tmp_path)

        assert cfg.discover_tomogram_path() is None

    def test_discover_tomogram_path_explicit_filename(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        target = data_dir / "tomogram_full_explicit.npy"
        target.write_bytes(b"x")

        cfg = ExtractionConfig(processed_data_path=tmp_path, tomogram_filename="tomogram_full_explicit.npy")

        assert cfg.discover_tomogram_path() == target

    def test_discover_tomogram_path_glob_fallback(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        match = data_dir / "tomogram_full_001.npy"
        match.write_bytes(b"x")

        cfg = ExtractionConfig(processed_data_path=tmp_path)

        assert cfg.discover_tomogram_path() == match

    def test_discover_height_range_explicit(self):
        cfg = ExtractionConfig(processed_data_path="/tmp/data", height_range=(-10.0, 50.0))

        assert cfg.discover_height_range() == (-10.0, 50.0)

    def test_discover_height_range_from_meta(self, tmp_path):
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "config_state_001.json").write_text(
            json.dumps({"output_configs": {"height_range": [-5.0, 80.0]}})
        )
        cfg = ExtractionConfig(processed_data_path=tmp_path)

        assert cfg.discover_height_range() == (-5.0, 80.0)

    def test_discover_height_range_none_when_unavailable(self, tmp_path):
        cfg = ExtractionConfig(processed_data_path=tmp_path)

        assert cfg.discover_height_range() is None

    def test_entry_config_defaults(self):
        cfg = ExtractParamsEntryConfig()

        assert cfg.fit_k_max         == 5
        assert cfg.fit_lambda_k      > 0.0
        assert cfg.parameter_workers > 0
        assert cfg.dataset_filter    == []

    def test_gpu_device_ids_default_independent(self):
        a = ExtractionConfig(processed_data_path="/tmp/a")
        b = ExtractionConfig(processed_data_path="/tmp/b")

        a.gpu_device_ids.append(9)

        assert b.gpu_device_ids == [0, 1, 3]


class TestPhysicsCheckConfig:
    def test_default_construction(self):
        cfg = PhysicsCheckEntryConfig()

        assert cfg.fit_k_max     == 5
        assert cfg.n_pixels      > 0
        assert cfg.physics_floor > 0.0
        assert cfg.capon_loading > 0.0
        assert isinstance(cfg.geometry, GeometryConfig)

    def test_moments_weights_length(self):
        cfg = PhysicsCheckEntryConfig()

        assert len(cfg.moments_weights) == 3


class TestProcessingConfig:
    def _crop(self):
        return CropRegion(1000, 2000, 500, 1000)

    def test_tomogram_configuration_defaults(self):
        cfg = TomogramConfiguration()

        assert cfg.height_range[0] < cfg.height_range[1]
        assert cfg.beamforming_method == "Capon"
        assert cfg.filter_arguments == {"win": [20, 10]}

    def test_tomogram_filter_arguments_independent(self):
        a = TomogramConfiguration()
        b = TomogramConfiguration()

        a.filter_arguments["win"].append(99)

        assert b.filter_arguments == {"win": [20, 10]}

    def test_parallel_resolve_fixed_workers_clamped(self):
        par = ParallelConfiguration(tomogram_workers=4)

        assert par.resolve_workers(10) == 4
        assert par.resolve_workers(2)  == 2

    def test_parallel_resolve_auto_at_least_one(self):
        par = ParallelConfiguration()

        assert par.resolve_workers(100) >= 1
        assert par.resolve_workers(1)   >= 1

    def test_parallel_available_cores_positive(self):
        assert ParallelConfiguration.available_cores() >= 1

    def test_path_configuration_run_directory_default(self):
        path = PathConfiguration()

        assert path.run_directory == path.main_directory

    def test_path_configuration_with_run_subdir(self):
        path = PathConfiguration(run_subdirectory="run_x")

        assert path.run_directory      == path.main_directory / "run_x"
        assert path.data_directory     == path.main_directory / "run_x" / "data"
        assert path.metadata_directory == path.main_directory / "run_x" / "meta"
        assert path.temporary_directory == path.main_directory / "run_x" / "tmp"

    def test_processing_configuration_post_init_sets_run_subdir(self):
        cfg = ProcessingConfiguration(crop=self._crop())

        assert cfg.paths.run_subdirectory is not None
        assert cfg.paths.run_subdirectory.startswith("run_")

    def test_processing_tomogram_tag(self):
        cfg = ProcessingConfiguration(crop=self._crop())

        assert cfg.tomogram_tag.startswith("1000a2000a500a1000")
        assert cfg.parameter_tag.startswith("1000a2000a500a1000")

    def test_processing_output_config_defaults_to_input(self):
        cfg = ProcessingConfiguration(crop=self._crop())

        assert cfg.has_split_configs is False
        assert cfg.output_config is cfg.input_configs

    def test_processing_output_config_split(self):
        out = TomogramConfiguration(polarisation="vv")
        cfg = ProcessingConfiguration(crop=self._crop(), output_configs=out)

        assert cfg.has_split_configs is True
        assert cfg.output_config is out

    def test_preprocess_entry_defaults(self):
        cfg = PreProcessEntryConfig()

        assert cfg.azimuth_start < cfg.azimuth_end
        assert cfg.range_start   < cfg.range_end
        assert len(cfg.win_list) == 4

    def test_preprocess_win_list_independent(self):
        a = PreProcessEntryConfig()
        b = PreProcessEntryConfig()

        a.win_list.append([1, 1])

        assert len(b.win_list) == 4

    def test_preprocess_dataset_name_default_composition(self):
        cfg  = PreProcessEntryConfig()
        name = cfg.resolve_dataset_name([20, 10], "20260606_120000")

        assert name == "17sartom-traun_L_1000a16000a500a4000_w20_10_hv_1_dtmf_20260606_120000"

    def test_preprocess_dataset_name_provided_single_win(self):
        cfg = PreProcessEntryConfig(dataset_name="traun_test", win_list=[[20, 10]])

        assert cfg.resolve_dataset_name([20, 10], "20260606_120000") == "traun_test"

    def test_preprocess_dataset_name_provided_multiple_wins(self):
        cfg = PreProcessEntryConfig(dataset_name="traun_test")

        assert cfg.resolve_dataset_name([20, 10], "20260606_120000") == "traun_test_w20_10"


class TestTrainingConfig:
    def test_geometry_defaults(self):
        geo = GeometryConfig()

        assert geo.wavelength  > 0.0
        assert geo.slant_range > 0.0
        assert len(geo.baselines) == 9
        assert geo.baselines[0] == 0.0

    def test_loss_normalization_defaults_present(self):
        norm = LossNormalizationConfig()

        assert norm.param_l1   == 1.0
        assert norm.mse_curve  > 0.0

    def test_loss_config_eff_multiplies_norm(self):
        cfg = LossConfig(weight_param_l1=0.5)

        assert np.isclose(cfg.eff("weight_param_l1"), 0.5 * cfg.norm.param_l1)

    def test_loss_config_eff_uses_norm_factor(self):
        cfg = LossConfig(weight_mse_curve=2.0)

        assert np.isclose(cfg.eff("weight_mse_curve"), 2.0 * cfg.norm.mse_curve)

    def test_loss_config_param_weights_triplet(self):
        cfg = LossConfig()

        assert len(cfg.param_weights)   == 3
        assert len(cfg.moments_weights) == 3

    def test_loss_curriculum_defaults(self):
        cur = LossCurriculumConfig()

        assert cur.enabled is False
        assert isinstance(cur.warmup,   LossConfig)
        assert isinstance(cur.complete, LossConfig)
        assert cur.warmup is not cur.complete

    def test_gaussian_config_make_param_names(self):
        cfg = GaussianConfig(n_default_gaussians=2, x_min=-10.0, x_max=50.0)

        assert cfg.make_param_names(2) == ["a1", "mu1", "sig1", "a2", "mu2", "sig2"]
        assert cfg.default_param_names == cfg.make_param_names(2)

    def test_gaussian_config_make_param_names_zero(self):
        cfg = GaussianConfig(n_default_gaussians=0, x_min=0.0, x_max=1.0)

        assert cfg.make_param_names(0) == []
        assert cfg.default_param_names == []

    def test_gaussian_config_defaults(self):
        cfg = GaussianConfig(n_default_gaussians=3, x_min=-5.0, x_max=80.0)

        assert cfg.amp_max             == 1000
        assert cfg.params_per_gaussian == 3
        assert cfg.x_min < cfg.x_max

    def test_gaussian_config_from_dataset(self, tmp_path):
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "config_state_001.json").write_text(
            json.dumps({"output_configs": {"height_range": [-5.0, 80.0]}})
        )
        cfg = GaussianConfig.from_dataset(tmp_path, n_gaussians=4)

        assert cfg.n_default_gaussians == 4
        assert cfg.x_min == -5.0
        assert cfg.x_max == 80.0

    def test_early_stopping_defaults(self):
        es = EarlyStoppingConfig()

        assert es.patience     > 0
        assert es.min_delta    > 0.0
        assert es.restore_best is True

    def test_warmup_defaults(self):
        wu = WarmupConfig()

        assert wu.warmup_steps        > 0
        assert 0.0 < wu.warmup_start_factor <= 1.0
        assert wu.warmup_mode == "linear"

    def test_scheduler_milestones_independent(self):
        a = SchedulerConfig()
        b = SchedulerConfig()

        a.milestones.append(120)

        assert b.milestones == [30, 60, 90]

    def test_ema_defaults(self):
        ema = EMAConfig()

        assert ema.use_ema is True
        assert 0.0 < ema.ema_decay < 1.0
        assert ema.update_every_n_steps > 0

    def test_optimizer_defaults(self):
        opt = OptimizerConfig()

        assert opt.lr  > 0.0
        assert opt.eps > 0.0
        assert len(opt.betas) == 2
        assert opt.betas[0] < opt.betas[1]

    def test_overfit_defaults(self):
        of = OverfitConfig()

        assert of.enabled    is False
        assert of.max_steps  > 0
        assert of.batch_size > 0

    def test_training_inner_device_valid(self):
        inner = TrainingConfigInner()

        assert inner.device in {"cuda", "cpu"}
        assert inner.epochs > 0
        assert inner.gradient_accumulation_steps >= 1

    def test_memory_config_defaults(self):
        mem = MemoryConfig()

        assert mem.streaming_eval is True
        assert mem.eval_pixel_subsample >= 0

    def test_resource_config_warn_thresholds(self):
        res = ResourceConfig()

        assert 0.0 < res.warn_ram_pct  <= 100.0
        assert 0.0 < res.warn_vram_pct <= 100.0
        assert res.poll_interval_sec > 0.0

    def test_gradient_clipper_defaults(self):
        gc = GradientClipperConfig()

        assert gc.clip_mode in {"disabled", "fixed", "adaptive_percentile", "adaptive_mean_std"}
        assert gc.max_grad_norm > 0.0
        assert 0.0 < gc.adaptive_percentile <= 100.0

    def test_permutation_metrics_defaults(self):
        pm = PermutationMetricsConfig()

        assert pm.enabled is True
        assert pm.amp_threshold > 0.0
        assert pm.max_G_for_margin > 0

    def test_io_config_defaults(self):
        io = IOConfig()

        assert isinstance(io.logdir, str)
        assert io.writer is None

    def test_trainer_config_nested_defaults(self):
        gaussian = GaussianConfig(n_default_gaussians=3, x_min=-5.0, x_max=80.0)
        cfg      = TrainerConfig(gaussian=gaussian)

        assert isinstance(cfg.geometry,            GeometryConfig)
        assert isinstance(cfg.early_stopping,      EarlyStoppingConfig)
        assert isinstance(cfg.warmup,              WarmupConfig)
        assert isinstance(cfg.scheduler,           SchedulerConfig)
        assert isinstance(cfg.ema,                 EMAConfig)
        assert isinstance(cfg.optimizer,           OptimizerConfig)
        assert isinstance(cfg.training,            TrainingConfigInner)
        assert isinstance(cfg.overfit,             OverfitConfig)
        assert isinstance(cfg.curriculum,          LossCurriculumConfig)
        assert isinstance(cfg.resources,           ResourceConfig)
        assert isinstance(cfg.memory,              MemoryConfig)
        assert isinstance(cfg.gradient_clipper,    GradientClipperConfig)
        assert isinstance(cfg.permutation_metrics, PermutationMetricsConfig)


class TestTrainEntryConfig:
    def test_default_construction(self):
        cfg = TrainEntryConfig()

        assert cfg.model_name  == "resunet"
        assert cfg.n_gaussians == 5
        assert isinstance(cfg.training,   TrainingQueueConfig)
        assert isinstance(cfg.curriculum, LossCurriculumConfig)
        assert isinstance(cfg.overfit,    OverfitConfig)
        assert isinstance(cfg.geometry,   GeometryConfig)
        assert isinstance(cfg.inference,  InferenceConfig)

    def test_default_curriculum_uses_param_l1(self):
        cfg = TrainEntryConfig()

        assert cfg.curriculum.warmup.use_param_l1   is True
        assert cfg.curriculum.complete.use_param_l1 is True

    def test_warmup_and_complete_losses_dicts(self):
        cfg = TrainEntryConfig()

        assert "pL11" in cfg.warmup_losses
        assert len(cfg.complete_losses) > 0
        for spec in cfg.complete_losses.values():
            assert spec["use_param_l1"] is True

    def test_gpus_and_overrides_independent(self):
        a = TrainEntryConfig()
        b = TrainEntryConfig()

        a.gpus.append(9)
        a.model_overrides["x"] = 1

        assert b.gpus            == [0, 1, 3]
        assert b.model_overrides == {}


class TestTuningConfig:
    def test_phase1_bounds_ordered(self):
        p1 = Phase1TuneConfig()

        assert p1.lr_low < p1.lr_high
        assert p1.wd_low < p1.wd_high
        assert p1.n_trials > 0

    def test_phase2_defaults(self):
        p2 = Phase2TuneConfig()

        assert p2.n_trials > 0
        assert p2.early_stop_patience > 0

    def test_tuning_config_nested(self):
        cfg = TuningConfig()

        assert isinstance(cfg.phase1, Phase1TuneConfig)
        assert isinstance(cfg.phase2, Phase2TuneConfig)
        assert cfg.n_gpus > 0

    def test_tuning_entry_log_dir_override(self):
        cfg = TuningEntryConfig()

        assert "tuning" in str(cfg.paths.log_base_dir)
        assert cfg.gpus == [0, 1, 2, 3]

    def test_tuning_entry_lists_independent(self):
        a = TuningEntryConfig()
        b = TuningEntryConfig()

        a.gpus.append(9)
        a.skip_models.append("x")

        assert b.gpus        == [0, 1, 2, 3]
        assert b.skip_models == []


class TestModelsConfig:
    def test_at_least_one_model_config_discovered(self):
        assert len(MODEL_CONFIG_CLASSES) > 0

    @pytest.mark.parametrize("config_cls", MODEL_CONFIG_CLASSES, ids=lambda c: c.__name__)
    def test_default_construction(self, config_cls):
        inst = config_cls()

        assert dataclasses.is_dataclass(inst)

    @pytest.mark.parametrize("config_cls", MODEL_CONFIG_CLASSES, ids=lambda c: c.__name__)
    def test_common_channel_fields(self, config_cls):
        inst = config_cls()

        assert inst.in_channels         == 1
        assert inst.out_channels        == 6
        assert inst.params_per_gaussian == 3

    @pytest.mark.parametrize("config_cls", MODEL_CONFIG_CLASSES, ids=lambda c: c.__name__)
    def test_learning_rate_and_weight_decay_positive(self, config_cls):
        inst        = config_cls()
        field_names = {f.name for f in dataclasses.fields(config_cls)}

        assert inst.encoder_lr > 0.0
        assert inst.decoder_lr > 0.0

        lr_fields = [n for n in field_names if n.endswith("_lr")]
        wd_fields = [n for n in field_names if n.endswith("_wd")]

        assert len(lr_fields) > 0
        for name in lr_fields:
            assert getattr(inst, name) > 0.0
        for name in wd_fields:
            assert getattr(inst, name) >= 0.0

    @pytest.mark.parametrize("config_cls", MODEL_CONFIG_CLASSES, ids=lambda c: c.__name__)
    def test_tunable_param_schemas_well_formed(self, config_cls):
        lr_params   = config_cls.tunable_lr_params()
        arch_params = config_cls.tunable_arch_params()

        assert isinstance(lr_params,   dict)
        assert isinstance(arch_params, dict)

        for spec in lr_params.values():
            assert spec["type"] in {"float", "int", "categorical", "indexed_categorical"}

        for spec in arch_params.values():
            assert "type" in spec
            assert ("choices" in spec) or ("low" in spec and "high" in spec)

    @pytest.mark.parametrize("config_cls", MODEL_CONFIG_CLASSES, ids=lambda c: c.__name__)
    def test_features_default_is_independent_list(self, config_cls):
        if "features" not in {f.name for f in dataclasses.fields(config_cls)}:
            pytest.skip("config has no features field")

        a = config_cls()
        b = config_cls()

        assert isinstance(a.features, list)
        a.features.append(9999)

        assert 9999 not in b.features

    @pytest.mark.parametrize("config_cls", MODEL_CONFIG_CLASSES, ids=lambda c: c.__name__)
    def test_dropout_in_unit_interval(self, config_cls):
        if "dropout" not in {f.name for f in dataclasses.fields(config_cls)}:
            pytest.skip("config has no dropout field")

        inst = config_cls()

        assert 0.0 <= inst.dropout <= 1.0
