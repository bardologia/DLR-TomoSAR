from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from configuration.benchmark_config import BenchmarkConfig
from configuration.cross_validation_config import CrossValidationConfig, FoldConfig
from tools.logger import Logger
from tools.regions import CropRegion, SplitRegions

from pipelines.benchmark_pipeline.sizing import (
    SizeMatcher,
    SizeMatchResult,
    WidthRule,
    WidthScaler,
)
from pipelines.benchmark_pipeline.config_factory import ConfigFactory
from pipelines.benchmark_pipeline.results import (
    ComparisonReport,
    TrialCollector,
    TrialRecord,
    _HEADLINE_METRICS,
)
from pipelines.benchmark_pipeline.workers import (
    BenchmarkWorker,
    InferenceWorker,
    OverfitWorker,
    TrainingWorker,
)
from pipelines.cross_validation_pipeline.folds import (
    FoldCollector,
    FoldConfigFactory,
    FoldPlan,
    FoldPlanner,
    CrossValidationWorker,
    FoldTrainingWorker,
    FoldInferenceWorker,
)
from pipelines.cross_validation_pipeline.cv_report import CrossValidationReport


@pytest.fixture
def logger(tmp_path):
    return Logger(log_dir=str(tmp_path / "logs"), name="benchmark_cv_test", level="ERROR")


@pytest.fixture
def dataset_dir(tmp_path):
    root = tmp_path / "dataset"

    data_dir = root / "data"
    meta_dir = root / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    layout = {"global_crop": [1000, 16000, 0, 500]}
    (data_dir / "dataset.json").write_text(json.dumps(layout), encoding="utf-8")

    config_state = {"tomogram_config": {"height_range": [-30.0, 30.0]}}
    (meta_dir / "config_state_0.json").write_text(json.dumps(config_state), encoding="utf-8")

    return root


@pytest.fixture
def benchmark_config(dataset_dir):
    config = BenchmarkConfig()
    config.paths.dataset_path    = dataset_dir
    config.paths.parameters_path = dataset_dir / "params.npy"
    config.paths.log_base_dir    = dataset_dir / "logs"
    config.n_gaussians           = 3
    return config


class TestWidthRule:
    def test_fields_are_stored(self):
        rule = WidthRule(attribute="features", divisor=8)

        assert rule.attribute == "features"
        assert rule.divisor   == 8


class TestWidthScaler:
    def test_known_models_have_rules(self):
        scaler = WidthScaler()

        for model in ("unet", "resunet", "swin_unet", "transunet", "unetr", "segformer", "fpn"):
            assert model in scaler.rules
            assert len(scaler.rules[model]) >= 1

    def test_overrides_unknown_model_raises(self):
        scaler = WidthScaler()

        with pytest.raises(ValueError):
            scaler.overrides("not_a_model", 1.0)

    def test_overrides_scale_one_matches_divisor_rounding(self):
        scaler    = WidthScaler()
        overrides = scaler.overrides("unet", 1.0)

        assert "features" in overrides
        assert isinstance(overrides["features"], list)
        for value in overrides["features"]:
            assert value % 8 == 0

    def test_overrides_smaller_scale_reduces_width(self):
        scaler = WidthScaler()

        big   = scaler.overrides("unet", 1.0)["features"]
        small = scaler.overrides("unet", 0.25)["features"]

        assert sum(small) < sum(big)

    def test_round_floor_is_divisor(self):
        scaler = WidthScaler()

        assert scaler._round(64, 0.0, 8) == 8
        assert scaler._round(64, 1e-9, 8) == 8

    def test_round_multiple_of_divisor(self):
        scaler = WidthScaler()

        result = scaler._round(100, 1.0, 8)
        assert result % 8 == 0
        assert result >= 8

    def test_round_scales_proportionally(self):
        scaler = WidthScaler()

        assert scaler._round(80, 2.0, 8) == 160

    def test_scaled_config_applies_overrides(self):
        scaler = WidthScaler()

        config    = scaler.scaled_config("unet", 0.5)
        overrides = scaler.overrides("unet", 0.5)

        assert getattr(config, "features") == overrides["features"]

    def test_scaled_config_returns_fresh_object(self):
        scaler = WidthScaler()

        first  = scaler.scaled_config("unet", 0.5)
        second = scaler.scaled_config("unet", 1.0)

        assert first.features != second.features


class TestSizeMatchResult:
    def test_default_history_is_empty_list(self):
        result = SizeMatchResult(
            model="unet", scale=1.0, overrides={}, parameters=10, target=10, deviation_pct=0.0, iterations=1,
        )

        assert result.history == []


class TestSizeMatcher:
    def test_init_derives_channels_and_image_size(self, benchmark_config, logger):
        matcher = SizeMatcher(benchmark_config, logger)

        assert matcher.in_channels  == benchmark_config.size_match.in_channels
        assert matcher.out_channels == benchmark_config.n_gaussians * 3
        assert matcher.image_size   == benchmark_config.training.patch_size[0]

    def test_reference_count_positive(self, benchmark_config, logger):
        matcher = SizeMatcher(benchmark_config, logger)

        count = matcher.reference_count()

        assert isinstance(count, int)
        assert count > 0

    def test_count_at_monotonic_in_scale(self, benchmark_config, logger):
        matcher = SizeMatcher(benchmark_config, logger)

        small = matcher._count_at("unet", 0.25)
        large = matcher._count_at("unet", 1.0)

        assert small < large

    def test_match_hits_self_target_with_low_deviation(self, benchmark_config, logger):
        benchmark_config.size_match.max_iterations = 8
        benchmark_config.size_match.scale_low      = 0.25
        benchmark_config.size_match.scale_high     = 2.0
        matcher = SizeMatcher(benchmark_config, logger)

        target = matcher._count_at("unet", 1.0)
        result = matcher.match("unet", target)

        assert isinstance(result, SizeMatchResult)
        assert result.model  == "unet"
        assert result.target == target
        assert abs(result.deviation_pct) <= 100.0
        assert result.iterations == len(result.history)
        assert result.overrides == matcher.scaler.overrides("unet", result.scale)

    def test_match_history_entries_have_expected_keys(self, benchmark_config, logger):
        benchmark_config.size_match.max_iterations = 6
        benchmark_config.size_match.scale_low      = 0.25
        benchmark_config.size_match.scale_high     = 1.0
        matcher = SizeMatcher(benchmark_config, logger)

        target = matcher._count_at("unet", 0.5)
        result = matcher.match("unet", target)

        assert len(result.history) >= 1
        for entry in result.history:
            assert set(entry.keys()) == {"iteration", "scale", "parameters", "deviation_pct"}

    def test_match_respects_max_iterations(self, benchmark_config, logger):
        benchmark_config.size_match.max_iterations = 3
        benchmark_config.size_match.tolerance      = 0.0
        benchmark_config.size_match.scale_low      = 0.25
        benchmark_config.size_match.scale_high     = 1.0
        matcher = SizeMatcher(benchmark_config, logger)

        result = matcher.match("unet", 1)

        assert result.iterations <= 3


class TestConfigFactoryFilesystem:
    def test_global_crop_reads_layout(self, benchmark_config):
        factory = ConfigFactory(benchmark_config)

        crop = factory.global_crop()

        assert isinstance(crop, CropRegion)
        assert crop.azimuth_start == 1000
        assert crop.azimuth_end   == 16000

    def test_benchmark_input_config_flags(self, benchmark_config):
        factory = ConfigFactory(benchmark_config)

        input_config = factory.benchmark_input_config()

        assert input_config.use_primary        is True
        assert input_config.use_secondaries    is True
        assert input_config.use_interferograms is True

    def test_training_dataset_config_split_regions(self, benchmark_config):
        factory = ConfigFactory(benchmark_config)

        dataset_config = factory.training_dataset_config()
        training       = benchmark_config.training

        train_region = dataset_config.split_regions.train
        assert train_region.azimuth_start == training.train_azimuth[0]
        assert train_region.azimuth_end   == training.train_azimuth[1]
        assert dataset_config.batch_size   == training.batch_size
        assert dataset_config.shuffle_train is True

    def test_overfit_dataset_config_uses_overfit_crop(self, benchmark_config):
        factory = ConfigFactory(benchmark_config)
        overfit = benchmark_config.overfit

        dataset_config = factory.overfit_dataset_config()
        region         = dataset_config.split_regions.train

        assert region.azimuth_start == overfit.azimuth_start
        assert region.azimuth_end   == overfit.azimuth_start + overfit.azimuth_lines
        assert dataset_config.split_regions.train is dataset_config.split_regions.train

    def test_overfit_dataset_config_disables_augmentation(self, benchmark_config):
        factory = ConfigFactory(benchmark_config)

        dataset_config = factory.overfit_dataset_config()
        aug            = dataset_config.augmentation

        assert aug.p_flip_h    == 0.0
        assert aug.p_flip_v    == 0.0
        assert aug.p_rot90     == 0.0
        assert aug.p_amp_scale == 0.0
        assert aug.p_noise     == 0.0

    def test_training_trainer_config_logdir_and_epochs(self, benchmark_config, tmp_path):
        factory = ConfigFactory(benchmark_config)
        logdir  = tmp_path / "trainlogs"

        trainer_config = factory.training_trainer_config(logdir)

        assert trainer_config.io.logdir == str(logdir)
        assert trainer_config.training.epochs == benchmark_config.training.epochs
        assert trainer_config.overfit.enabled is False
        assert trainer_config.gaussian.x_min == -30.0
        assert trainer_config.gaussian.x_max == 30.0

    def test_training_trainer_config_scheduler_epochs_fallback(self, benchmark_config, tmp_path):
        benchmark_config.training.scheduler_epochs = None
        benchmark_config.training.epochs            = 77
        factory = ConfigFactory(benchmark_config)

        trainer_config = factory.training_trainer_config(tmp_path / "logdir")

        assert trainer_config.scheduler.epochs == 77

    def test_overfit_trainer_config_enables_overfit(self, benchmark_config, tmp_path):
        factory = ConfigFactory(benchmark_config)

        trainer_config = factory.overfit_trainer_config(tmp_path / "logdir")

        assert trainer_config.overfit.enabled is True
        assert trainer_config.overfit.max_steps  == benchmark_config.overfit.max_steps
        assert trainer_config.overfit.batch_size == benchmark_config.overfit.batch_size
        assert trainer_config.scheduler.type == "constant"
        assert trainer_config.warmup.warmup_enabled is False

    def test_inference_config_maps_fields(self, benchmark_config, tmp_path):
        factory = ConfigFactory(benchmark_config)
        run_dir = tmp_path / "run"

        inference_config = factory.inference_config(run_dir)

        assert inference_config.run_directory == run_dir
        assert inference_config.output_subdir is None
        assert inference_config.split        == benchmark_config.inference.split
        assert inference_config.use_ema      == benchmark_config.inference.use_ema
        assert inference_config.gif_axes     == list(benchmark_config.inference.gif_axes)


@dataclass
class _DummyModelConfig:
    dropout                : float = 0.5
    attention_dropout      : float = 0.3
    stochastic_depth_rate  : float = 0.2
    features_wd            : float = 0.01
    encoder_lr             : float = 3e-4
    keep_me               : int   = 7


class TestConfigFactoryPrepareOverfit:
    def test_prepare_overfit_zeroes_regularization(self, benchmark_config):
        factory = ConfigFactory(benchmark_config)

        prepared = factory.prepare_overfit_model_config(_DummyModelConfig())

        assert prepared.dropout               == 0.0
        assert prepared.attention_dropout     == 0.0
        assert prepared.stochastic_depth_rate == 0.0
        assert prepared.features_wd           == 0.0
        assert prepared.keep_me               == 7

    def test_prepare_overfit_scales_learning_rates(self, benchmark_config):
        factory = ConfigFactory(benchmark_config)

        prepared = factory.prepare_overfit_model_config(_DummyModelConfig())

        assert prepared.encoder_lr == pytest.approx(3e-3)

    def test_prepare_overfit_returns_same_object(self, benchmark_config):
        factory  = ConfigFactory(benchmark_config)
        original = _DummyModelConfig()

        prepared = factory.prepare_overfit_model_config(original)

        assert prepared is original

    def test_prepare_overfit_tolerates_missing_attributes(self, benchmark_config):
        @dataclass
        class _Bare:
            channels: int = 16

        factory  = ConfigFactory(benchmark_config)
        prepared = factory.prepare_overfit_model_config(_Bare())

        assert prepared.channels == 16


class TestTrialRecord:
    def test_has_inference_false_by_default(self, tmp_path):
        record = TrialRecord(name="unet", run_dir=tmp_path)

        assert record.has_inference is False

    def test_has_inference_true_when_set(self, tmp_path):
        record = TrialRecord(name="unet", run_dir=tmp_path, inference_dir=tmp_path / "inf")

        assert record.has_inference is True

    def test_default_collections_are_independent(self, tmp_path):
        first  = TrialRecord(name="a", run_dir=tmp_path)
        second = TrialRecord(name="b", run_dir=tmp_path)

        first.metrics["x"] = 1
        first.figures.append(tmp_path / "f.png")

        assert second.metrics == {}
        assert second.figures == []


def _make_trial_tree(run_dir, trial_names, metrics_by_trial=None):
    metrics_by_trial = metrics_by_trial or {}

    training_dir = run_dir / "training"
    training_dir.mkdir(parents=True)

    for name in trial_names:
        trial_dir = training_dir / name
        docs      = trial_dir / "docs"
        docs.mkdir(parents=True)

        (docs / "model_doc.md").write_text(
            f"# {name}\n**Total Parameters:** `1,234,567`\n", encoding="utf-8",
        )

        metrics = metrics_by_trial.get(name)
        if metrics is not None:
            inference_dir = trial_dir / "inference" / "run_0"
            inference_dir.mkdir(parents=True)
            (inference_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    return training_dir


class TestTrialCollector:
    def test_collect_missing_training_dir_returns_empty(self, tmp_path, logger):
        collector = TrialCollector(run_dir=tmp_path / "run", logger=logger)

        records = collector.collect()

        assert records == []

    def test_collect_parses_parameters_and_sorts(self, tmp_path, logger):
        run_dir = tmp_path / "run"
        _make_trial_tree(run_dir, ["resunet", "unet"])

        collector = TrialCollector(run_dir=run_dir, logger=logger)
        records   = collector.collect()

        assert [r.name for r in records] == ["resunet", "unet"]
        assert all(r.parameters == 1234567 for r in records)
        assert all(not r.has_inference for r in records)

    def test_collect_attaches_inference(self, tmp_path, logger):
        run_dir = tmp_path / "run"
        _make_trial_tree(run_dir, ["unet"], metrics_by_trial={"unet": {"curve_rmse_gt": 0.5}})

        collector = TrialCollector(run_dir=run_dir, logger=logger)
        records   = collector.collect()

        assert records[0].has_inference is True
        assert records[0].metrics == {"curve_rmse_gt": 0.5}

    def test_load_json_returns_none_for_missing(self, tmp_path, logger):
        collector = TrialCollector(run_dir=tmp_path, logger=logger)

        assert collector._load_json(tmp_path / "missing.json") is None

    def test_load_json_returns_none_for_invalid(self, tmp_path, logger):
        bad = tmp_path / "bad.json"
        bad.write_text("{ not valid json", encoding="utf-8")
        collector = TrialCollector(run_dir=tmp_path, logger=logger)

        assert collector._load_json(bad) is None

    def test_parse_parameters_falls_back_to_size_match(self, tmp_path, logger):
        trial_dir = tmp_path / "trial"
        (trial_dir / "docs").mkdir(parents=True)
        collector = TrialCollector(run_dir=tmp_path, logger=logger)

        result = collector._parse_parameters(trial_dir, {"parameters": 999})

        assert result == 999

    def test_read_checkpoint_missing_returns_empty(self, tmp_path, logger):
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        collector = TrialCollector(run_dir=tmp_path, logger=logger)

        assert collector._read_checkpoint(trial_dir) == {}

    def test_read_checkpoint_extracts_known_keys(self, tmp_path, logger):
        import torch

        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        checkpoint = {
            "best_val_loss": 0.1,
            "best_epoch":    5,
            "epoch":         9,
            "global_step":   100,
            "train_losses":  [1.0, 0.5, 0.1],
            "val_losses":    [1.1, 0.6],
        }
        torch.save(checkpoint, trial_dir / "best_model.pt")

        collector = TrialCollector(run_dir=tmp_path, logger=logger)
        info      = collector._read_checkpoint(trial_dir)

        assert info["best_val_loss"]  == 0.1
        assert info["best_epoch"]     == 5
        assert info["n_train_epochs"] == 3
        assert info["n_val_epochs"]   == 2


def _record_with_metrics(name, run_dir, metrics):
    return TrialRecord(name=name, run_dir=run_dir, parameters=1000, metrics=dict(metrics), inference_dir=run_dir / "inf")


class TestComparisonReport:
    def test_write_all_creates_core_files(self, tmp_path, logger):
        records = [
            _record_with_metrics("unet", tmp_path / "u", {"curve_rmse_gt": 0.5, "overall_r2_gt": 0.9}),
            _record_with_metrics("resunet", tmp_path / "r", {"curve_rmse_gt": 0.4, "overall_r2_gt": 0.95}),
        ]
        report = ComparisonReport(records, tmp_path / "out", reference_model="unet", embed_images=False, logger=logger)

        written = report.write_all()
        names   = {p.name for p in written}

        assert "benchmark_overview.md" in names
        assert "metrics_comparison.md" in names
        assert "comparison_summary.json" in names
        for path in written:
            assert path.exists()

    def test_fmt_formats_floats_and_none(self, tmp_path, logger):
        report = ComparisonReport([], tmp_path, reference_model="unet", embed_images=False, logger=logger)

        assert report._fmt(1.23456789) == "1.2346"
        assert report._fmt(None)        == "—"
        assert report._fmt(7)           == "7"

    def test_natural_key_sorts_numerically(self, tmp_path, logger):
        report = ComparisonReport([], tmp_path, reference_model="unet", embed_images=False, logger=logger)

        names  = ["slice_10", "slice_2", "slice_1"]
        ordered = sorted(names, key=report._natural_key)

        assert ordered == ["slice_1", "slice_2", "slice_10"]

    def test_direction_classification(self, tmp_path, logger):
        report = ComparisonReport([], tmp_path, reference_model="unet", embed_images=False, logger=logger)

        assert report._direction("curve_rmse_gt")   == "lower"
        assert report._direction("overall_r2_gt")   == "higher"
        assert report._direction("pixel_ssim_mean") == "higher"
        assert report._direction("n_pixels")        is None

    def test_leaderboard_ranks_best_first(self, tmp_path, logger):
        records = [
            _record_with_metrics("good", tmp_path / "g", {"curve_rmse_gt": 0.1, "overall_r2_gt": 0.99}),
            _record_with_metrics("bad",  tmp_path / "b", {"curve_rmse_gt": 0.9, "overall_r2_gt": 0.10}),
        ]
        report = ComparisonReport(records, tmp_path / "out", reference_model="good", embed_images=False, logger=logger)

        lines = report._leaderboard()
        text  = "\n".join(lines)

        good_pos = text.index("`good`")
        bad_pos  = text.index("`bad`")
        assert good_pos < bad_pos

    def test_leaderboard_no_metrics(self, tmp_path, logger):
        records = [TrialRecord(name="unet", run_dir=tmp_path)]
        report  = ComparisonReport(records, tmp_path, reference_model="unet", embed_images=False, logger=logger)

        lines = report._leaderboard()

        assert any("No inference metrics" in line for line in lines)

    def test_metric_table_marks_best_in_bold(self, tmp_path, logger):
        records = [
            _record_with_metrics("a", tmp_path / "a", {"curve_rmse_gt": 0.1}),
            _record_with_metrics("b", tmp_path / "b", {"curve_rmse_gt": 0.9}),
        ]
        report = ComparisonReport(records, tmp_path, reference_model="a", embed_images=False, logger=logger)

        rows = report._metric_table(["curve_rmse_gt"], records)
        text = "\n".join(rows)

        assert "**0.1**" in text
        assert "↓" in text

    def test_img_src_embeds_when_requested(self, tmp_path, logger):
        png = tmp_path / "fig.png"
        png.write_bytes(b"\x89PNG\r\n")
        report = ComparisonReport([], tmp_path, reference_model="unet", embed_images=True, logger=logger)

        src = report._img_src(png)

        assert src.startswith("data:image/png;base64,")

    def test_img_src_relative_when_not_embedding(self, tmp_path, logger):
        png = tmp_path / "figures" / "fig.png"
        png.parent.mkdir()
        png.write_bytes(b"data")
        report = ComparisonReport([], tmp_path / "out", reference_model="unet", embed_images=False, logger=logger)

        src = report._img_src(png)

        assert not src.startswith("data:")
        assert src.endswith("fig.png")

    def test_summary_json_round_trips(self, tmp_path, logger):
        records = [_record_with_metrics("unet", tmp_path / "u", {"curve_rmse_gt": 0.5})]
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        report  = ComparisonReport(records, out_dir, reference_model="unet", embed_images=False, logger=logger)

        out     = report._write_summary_json()
        payload = json.loads(out.read_text(encoding="utf-8"))

        assert payload[0]["name"]    == "unet"
        assert payload[0]["metrics"] == {"curve_rmse_gt": 0.5}

    def test_write_figures_groups_by_pattern(self, tmp_path, logger):
        inf_dir = tmp_path / "u" / "inference" / "run_0"
        figures = inf_dir / "figures"
        figures.mkdir(parents=True)
        (figures / "profiles_best.png").write_bytes(b"x")
        (figures / "param_mu.png").write_bytes(b"x")

        record = TrialRecord(
            name="unet", run_dir=tmp_path / "u", inference_dir=inf_dir,
            figures=[figures / "profiles_best.png", figures / "param_mu.png"],
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        report = ComparisonReport([record], out_dir, reference_model="unet", embed_images=False, logger=logger)

        written = report._write_figures()

        assert len(written) >= 1
        assert all(p.exists() for p in written)


class TestBenchmarkWorker:
    def test_init_sets_run_dir(self, benchmark_config):
        worker = BenchmarkWorker(benchmark_config, run_tag="run_a")

        assert worker.run_dir == Path(benchmark_config.paths.log_base_dir) / "run_a"
        assert isinstance(worker.factory, ConfigFactory)

    def test_probe_config_disabled(self, benchmark_config):
        worker = BenchmarkWorker(benchmark_config, run_tag="run_a")

        probe = worker._probe_config()

        assert probe.enabled is False
        assert probe.exit_after is True


class TestOverfitWorkerHelpers:
    def test_final_loss_from_outputs(self, benchmark_config):
        worker = OverfitWorker(benchmark_config, run_tag="run_a")

        assert worker._final_loss(([1.0, 0.5, 0.25], [])) == 0.25

    def test_final_loss_empty_outputs(self, benchmark_config):
        worker = OverfitWorker(benchmark_config, run_tag="run_a")

        assert worker._final_loss(()) is None
        assert worker._final_loss(None) is None
        assert worker._final_loss(([], [])) is None


class TestTrainingWorkerHelpers:
    def test_size_overrides_missing_file(self, benchmark_config):
        worker = TrainingWorker(benchmark_config, run_tag="run_a")

        assert worker._size_overrides("unet") == {}

    def test_size_overrides_reads_entry(self, benchmark_config):
        worker = TrainingWorker(benchmark_config, run_tag="run_a")
        pipeline_dir = worker.run_dir / "pipeline"
        pipeline_dir.mkdir(parents=True)

        records = {"unet": {"overrides": {"features": [16, 32]}}}
        (pipeline_dir / "size_match.json").write_text(json.dumps(records), encoding="utf-8")

        assert worker._size_overrides("unet") == {"features": [16, 32]}
        assert worker._size_overrides("resunet") == {}


class TestInferenceWorker:
    def test_inference_worker_constructs(self, benchmark_config):
        worker = InferenceWorker(benchmark_config, run_tag="run_a")

        assert worker.run_dir.name == "run_a"


@pytest.fixture
def cv_config(dataset_dir):
    config = CrossValidationConfig()
    config.paths.dataset_path    = dataset_dir
    config.paths.parameters_path = dataset_dir / "params.npy"
    config.paths.log_base_dir    = dataset_dir / "logs"
    config.n_gaussians           = 3
    config.folds                 = FoldConfig(n_folds=5, azimuth_start=0, azimuth_end=100)
    return config


class TestFoldPlanner:
    def test_requires_minimum_three_folds(self):
        config = CrossValidationConfig()
        config.folds = FoldConfig(n_folds=2, azimuth_start=0, azimuth_end=100)

        with pytest.raises(ValueError):
            FoldPlanner(config, range_start=0, range_end=10)

    def test_partition_too_small_extent_raises(self):
        config = CrossValidationConfig()
        config.folds = FoldConfig(n_folds=10, azimuth_start=0, azimuth_end=5)

        with pytest.raises(ValueError):
            FoldPlanner(config, range_start=0, range_end=10)

    def test_partition_covers_full_extent(self):
        config = CrossValidationConfig()
        config.folds = FoldConfig(n_folds=4, azimuth_start=0, azimuth_end=100)
        planner = FoldPlanner(config, range_start=0, range_end=10)

        assert planner.blocks[0][0]  == 0
        assert planner.blocks[-1][1] == 100
        for index in range(len(planner.blocks) - 1):
            assert planner.blocks[index][1] == planner.blocks[index + 1][0]

    def test_plan_disjoint_splits(self):
        config = CrossValidationConfig()
        config.folds = FoldConfig(n_folds=5, azimuth_start=0, azimuth_end=100)
        planner = FoldPlanner(config, range_start=0, range_end=10)

        plan = planner.plan(0)

        assert plan.test_block == 0
        assert plan.val_block  == 1
        assert set(plan.train_blocks).isdisjoint({plan.test_block, plan.val_block})
        assert plan.train_blocks == [2, 3, 4]

    def test_plan_val_wraps_around(self):
        config = CrossValidationConfig()
        config.folds = FoldConfig(n_folds=4, azimuth_start=0, azimuth_end=100)
        planner = FoldPlanner(config, range_start=0, range_end=10)

        plan = planner.plan(3)

        assert plan.test_block == 3
        assert plan.val_block  == 0

    def test_plan_index_out_of_range_raises(self):
        config = CrossValidationConfig()
        config.folds = FoldConfig(n_folds=4, azimuth_start=0, azimuth_end=100)
        planner = FoldPlanner(config, range_start=0, range_end=10)

        with pytest.raises(ValueError):
            planner.plan(4)
        with pytest.raises(ValueError):
            planner.plan(-1)

    def test_plans_returns_all_folds(self):
        config = CrossValidationConfig()
        config.folds = FoldConfig(n_folds=5, azimuth_start=0, azimuth_end=100)
        planner = FoldPlanner(config, range_start=0, range_end=10)

        plans = planner.plans()

        assert len(plans) == 5
        assert [p.fold_index for p in plans] == [0, 1, 2, 3, 4]

    def test_merge_adjacent_groups_runs(self):
        config = CrossValidationConfig()
        config.folds = FoldConfig(n_folds=5, azimuth_start=0, azimuth_end=100)
        planner = FoldPlanner(config, range_start=0, range_end=10)

        assert planner._merge_adjacent([2, 3, 4]) == [(2, 4)]
        assert planner._merge_adjacent([0, 2, 3]) == [(0, 0), (2, 3)]
        assert planner._merge_adjacent([1])       == [(1, 1)]

    def test_split_regions_are_crop_regions(self):
        config = CrossValidationConfig()
        config.folds = FoldConfig(n_folds=5, azimuth_start=0, azimuth_end=100)
        planner = FoldPlanner(config, range_start=2, range_end=8)

        plan        = planner.plan(0)
        test_region = plan.split_regions.regions("test")[0]

        assert isinstance(test_region, CropRegion)
        assert test_region.range_start == 2
        assert test_region.range_end   == 8

    def test_train_single_run_is_not_a_list(self):
        config = CrossValidationConfig()
        config.folds = FoldConfig(n_folds=3, azimuth_start=0, azimuth_end=99)
        planner = FoldPlanner(config, range_start=0, range_end=10)

        plan = planner.plan(0)

        assert isinstance(plan.split_regions.train, CropRegion)


class TestFoldPlan:
    def test_dataclass_fields(self):
        regions = SplitRegions(
            train=CropRegion(0, 10, 0, 10), val=CropRegion(10, 20, 0, 10), test=CropRegion(20, 30, 0, 10),
        )
        plan = FoldPlan(fold_index=0, test_block=0, val_block=1, train_blocks=[2], split_regions=regions)

        assert plan.fold_index   == 0
        assert plan.train_blocks == [2]


class TestFoldConfigFactory:
    def test_planner_is_cached(self, cv_config):
        factory = FoldConfigFactory(cv_config)

        first  = factory.planner()
        second = factory.planner()

        assert first is second
        assert isinstance(first, FoldPlanner)

    def test_fold_dataset_config_uses_fold_split(self, cv_config):
        factory = FoldConfigFactory(cv_config)

        dataset_config = factory.fold_dataset_config(0)
        plan           = factory.planner().plan(0)

        assert dataset_config.split_regions == plan.split_regions

    def test_fold_inference_config_sets_split_and_subdir(self, cv_config, tmp_path):
        factory = FoldConfigFactory(cv_config)

        inference_config = factory.fold_inference_config(tmp_path / "run", split="val")

        assert inference_config.split         == "val"
        assert inference_config.output_subdir == "val"


class TestFoldCollector:
    def test_training_dir_is_folds(self, tmp_path, logger):
        collector = FoldCollector(run_dir=tmp_path, splits=["val", "test"], logger=logger)

        assert collector.training_dir == tmp_path / "folds"
        assert collector.splits == ["val", "test"]

    def test_collect_by_split_views(self, tmp_path, logger):
        run_dir = tmp_path / "run"
        folds_dir = run_dir / "folds"

        fold_dir = folds_dir / "fold_0"
        (fold_dir / "docs").mkdir(parents=True)
        (fold_dir / "docs" / "model_doc.md").write_text("**Total Parameters:** `100`\n", encoding="utf-8")

        for split, value in (("val", 0.2), ("test", 0.3)):
            split_dir = fold_dir / "inference" / split
            split_dir.mkdir(parents=True)
            (split_dir / "metrics.json").write_text(json.dumps({"curve_rmse_gt": value}), encoding="utf-8")

        collector = FoldCollector(run_dir=run_dir, splits=["val", "test"], logger=logger)
        base, by_split = collector.collect_by_split()

        assert len(base) == 1
        assert by_split["val"][0].metrics  == {"curve_rmse_gt": 0.2}
        assert by_split["test"][0].metrics == {"curve_rmse_gt": 0.3}

    def test_split_view_no_metrics_clears_record(self, tmp_path, logger):
        run_dir = tmp_path / "run"
        record  = TrialRecord(name="fold_0", run_dir=run_dir / "folds" / "fold_0")

        collector = FoldCollector(run_dir=run_dir, splits=["val"], logger=logger)
        view      = collector._split_view(record, "val")

        assert view.inference_dir is None
        assert view.metrics == {}
        assert view.figures == []


class TestCrossValidationWorkers:
    def test_fold_name(self, cv_config):
        worker = CrossValidationWorker(cv_config, run_tag="cv")

        assert worker.fold_name(3) == "fold_3"
        assert isinstance(worker.factory, FoldConfigFactory)

    def test_fold_training_worker_uses_fold_factory(self, cv_config):
        worker = FoldTrainingWorker(cv_config, run_tag="cv")

        assert isinstance(worker.factory, FoldConfigFactory)

    def test_fold_inference_worker_constructs(self, cv_config):
        worker = FoldInferenceWorker(cv_config, run_tag="cv")

        assert worker.run_dir.name == "cv"


class TestCrossValidationReport:
    def _planner(self, cv_config):
        return FoldConfigFactory(cv_config).planner()

    def test_mean_std_single_value(self, cv_config, tmp_path, logger):
        report = CrossValidationReport(
            base_records=[], records_by_split={}, planner=self._planner(cv_config),
            out_dir=tmp_path, model_name="resunet", embed_images=False, logger=logger,
        )

        mean, std = report._mean_std([5.0])

        assert mean == 5.0
        assert std  == 0.0

    def test_mean_std_sample_std(self, cv_config, tmp_path, logger):
        report = CrossValidationReport(
            base_records=[], records_by_split={}, planner=self._planner(cv_config),
            out_dir=tmp_path, model_name="resunet", embed_images=False, logger=logger,
        )

        mean, std = report._mean_std([2.0, 4.0])

        assert mean == 3.0
        assert std  == pytest.approx(1.41421356, abs=1e-6)

    def test_scalar_keys_excludes_per_bin(self, cv_config, tmp_path, logger):
        record = TrialRecord(
            name="fold_0", run_dir=tmp_path, metrics={"curve_rmse_gt": 0.5, "elev_metric_3": 0.1, "text": "x"},
        )
        report = CrossValidationReport(
            base_records=[record], records_by_split={"test": [record]}, planner=self._planner(cv_config),
            out_dir=tmp_path, model_name="resunet", embed_images=False, logger=logger,
        )

        keys = report._scalar_keys([record])

        assert "curve_rmse_gt" in keys
        assert "elev_metric_3" not in keys
        assert "text" not in keys

    def test_write_all_creates_aggregate_and_summary(self, cv_config, tmp_path, logger):
        records_by_split = {
            "val":  [TrialRecord(name="fold_0", run_dir=tmp_path / "f0", metrics={"curve_rmse_gt": 0.2}, inference_dir=tmp_path / "f0")],
            "test": [TrialRecord(name="fold_0", run_dir=tmp_path / "f0", metrics={"curve_rmse_gt": 0.3}, inference_dir=tmp_path / "f0")],
        }
        base_records = [TrialRecord(name="fold_0", run_dir=tmp_path / "f0", checkpoint={"best_val_loss": 0.2, "best_epoch": 4})]

        report = CrossValidationReport(
            base_records=base_records, records_by_split=records_by_split, planner=self._planner(cv_config),
            out_dir=tmp_path / "cv_out", model_name="resunet", embed_images=False, logger=logger,
        )

        written = report.write_all()
        names   = {p.name for p in written}

        assert "cv_aggregate_report.md" in names
        assert "cv_summary.json" in names
        for path in written:
            assert path.exists()

    def test_summary_json_aggregates_per_split(self, cv_config, tmp_path, logger):
        records_by_split = {
            "test": [
                TrialRecord(name="fold_0", run_dir=tmp_path / "f0", metrics={"curve_rmse_gt": 0.2}, inference_dir=tmp_path / "f0"),
                TrialRecord(name="fold_1", run_dir=tmp_path / "f1", metrics={"curve_rmse_gt": 0.4}, inference_dir=tmp_path / "f1"),
            ],
        }
        base_records = [
            TrialRecord(name="fold_0", run_dir=tmp_path / "f0", checkpoint={"best_val_loss": 0.2}),
            TrialRecord(name="fold_1", run_dir=tmp_path / "f1", checkpoint={"best_val_loss": 0.4}),
        ]
        out_dir = tmp_path / "cv_out"
        out_dir.mkdir()
        report = CrossValidationReport(
            base_records=base_records, records_by_split=records_by_split, planner=self._planner(cv_config),
            out_dir=out_dir, model_name="resunet", embed_images=False, logger=logger,
        )

        out     = report._write_summary_json()
        payload = json.loads(out.read_text(encoding="utf-8"))

        assert payload["model"] == "resunet"
        assert payload["folds"] == ["fold_0", "fold_1"]
        agg = payload["splits"]["test"]["curve_rmse_gt"]
        assert agg["mean"] == pytest.approx(0.3)
        assert agg["per_fold"] == {"fold_0": 0.2, "fold_1": 0.4}
        assert payload["best_val_loss"]["mean"] == pytest.approx(0.3)

    def test_aggregate_table_skips_non_numeric_only_keys(self, cv_config, tmp_path, logger):
        record = TrialRecord(name="fold_0", run_dir=tmp_path, metrics={"label": "x"})
        report = CrossValidationReport(
            base_records=[record], records_by_split={"test": [record]}, planner=self._planner(cv_config),
            out_dir=tmp_path, model_name="resunet", embed_images=False, logger=logger,
        )

        rows = report._aggregate_table(["label"], [record])
        text = "\n".join(rows)

        assert "`label`" not in text


class TestHeadlineMetricsConstant:
    def test_headline_metrics_pairs(self):
        assert all(isinstance(key, str) and isinstance(label, str) for key, label in _HEADLINE_METRICS)
        keys = [key for key, _ in _HEADLINE_METRICS]
        assert "curve_rmse_gt" in keys
