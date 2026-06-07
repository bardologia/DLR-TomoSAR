from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pipelines.inference_pipeline.baseline import CaponSpectrum, ClassicalBaseline, CovarianceEstimator, GeometryLoader, PreprocessConfigReader
from pipelines.inference_pipeline.metrics  import Metrics, Result
from tools.logger import NullLogger


def _write_trainer_config(run_dir: Path, baselines, wavelength=0.23, slant_range=5000.0, kz_values=()):
    docs = run_dir / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    payload = {
        "geometry": {
            "wavelength"       : wavelength,
            "slant_range"      : slant_range,
            "look_angle_deg"   : 45.0,
            "baselines"        : list(baselines),
            "kz_values"        : list(kz_values),
            "baselines_origin" : "test",
        },
    }
    (docs / "trainer_config.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_config_state(dataset_dir: Path, win=(20, 10), method="Capon", key="tomogram_config", beamforming_arguments=()):
    meta = dataset_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    payload = {
        key: {
            "filter_method"         : "Boxcar",
            "filter_arguments"      : {"win": list(win)},
            "beamforming_method"    : method,
            "beamforming_arguments" : list(beamforming_arguments),
            "height_range"          : [-20.0, 80.0],
        },
    }
    (meta / "config_state_test.json").write_text(json.dumps(payload), encoding="utf-8")


def _synthetic_inputs(kz, height, n_az=24, n_rg=24, amplitude=2.0):
    n_sec  = len(kz) - 1
    inputs = np.empty((1 + 2 * n_sec, n_az, n_rg), dtype=np.complex64)

    inputs[0] = amplitude
    inputs[1:1 + n_sec] = amplitude
    for i, kz_i in enumerate(kz[1:]):
        inputs[1 + n_sec + i] = amplitude * np.exp(1j * kz_i * height)

    return inputs


class TestGeometryLoader:
    def test_kz_from_baselines(self, tmp_path):
        _write_trainer_config(tmp_path, baselines=(0.0, 10.0, 20.0))

        kz = GeometryLoader(tmp_path, NullLogger()).load_kz()

        scale = 4.0 * math.pi / (0.23 * 5000.0)
        assert kz == pytest.approx([0.0, scale * 10.0, scale * 20.0])

    def test_explicit_kz_values_take_priority(self, tmp_path):
        _write_trainer_config(tmp_path, baselines=(0.0, 10.0), kz_values=(0.0, 0.1, 0.2))

        kz = GeometryLoader(tmp_path, NullLogger()).load_kz()

        assert kz == pytest.approx([0.0, 0.1, 0.2])

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            GeometryLoader(tmp_path, NullLogger()).load_kz()


class TestCovarianceEstimator:
    def test_constant_stack_gives_outer_product(self):
        stack = np.stack([
            np.full((8, 8), 2.0 + 0.0j, dtype=np.complex64),
            np.full((8, 8), 1.0 + 1.0j, dtype=np.complex64),
        ])

        cov = CovarianceEstimator((3, 3)).estimate(stack)

        assert cov.shape == (2, 2, 8, 8)
        assert cov[0, 0, 4, 4] == pytest.approx(4.0)
        assert cov[1, 1, 4, 4] == pytest.approx(2.0)
        assert cov[0, 1, 4, 4] == pytest.approx(2.0 * (1.0 - 1.0j))

    def test_hermitian(self):
        rng   = np.random.default_rng(0)
        stack = (rng.normal(size=(3, 10, 10)) + 1j * rng.normal(size=(3, 10, 10))).astype(np.complex64)

        cov = CovarianceEstimator((3, 3)).estimate(stack)

        assert np.allclose(cov[0, 2], np.conj(cov[2, 0]), atol=1e-5)


class TestCaponSpectrum:
    def test_peak_at_scatterer_height(self):
        kz     = np.array([0.0, 0.05, 0.10, 0.15, 0.20])
        x_axis = np.linspace(-20.0, 80.0, 101)
        height = 30.0

        n_tracks = kz.size
        stack    = np.empty((n_tracks, 6, 6), dtype=np.complex64)
        for i, kz_i in enumerate(kz):
            stack[i] = 2.0 * np.exp(1j * kz_i * height)

        cov  = CovarianceEstimator((3, 3)).estimate(stack)
        spec = CaponSpectrum(kz, x_axis, loading=1e-2, phase_sign=1.0).compute(cov)

        peak_height = x_axis[spec[:, 3, 3].argmax()]
        assert peak_height == pytest.approx(height, abs=float(x_axis[1] - x_axis[0]))

    def test_phase_sign_mirrors_peak(self):
        kz     = np.array([0.0, 0.05, 0.10, 0.15, 0.20])
        x_axis = np.linspace(-80.0, 80.0, 161)
        height = 30.0

        stack = np.empty((kz.size, 6, 6), dtype=np.complex64)
        for i, kz_i in enumerate(kz):
            stack[i] = np.exp(1j * kz_i * height)

        cov  = CovarianceEstimator((3, 3)).estimate(stack)
        spec = CaponSpectrum(kz, x_axis, loading=1e-2, phase_sign=-1.0).compute(cov)

        peak_height = x_axis[spec[:, 3, 3].argmax()]
        assert peak_height == pytest.approx(-height, abs=float(x_axis[1] - x_axis[0]))

    def test_chunking_matches_single_pass(self):
        rng    = np.random.default_rng(1)
        kz     = np.array([0.0, 0.08, 0.16])
        x_axis = np.linspace(-10.0, 10.0, 21)
        stack  = (rng.normal(size=(3, 9, 5)) + 1j * rng.normal(size=(3, 9, 5))).astype(np.complex64)
        cov    = CovarianceEstimator((3, 3)).estimate(stack)

        full    = CaponSpectrum(kz, x_axis, loading=1e-2, phase_sign=1.0, chunk_rows=64).compute(cov)
        chunked = CaponSpectrum(kz, x_axis, loading=1e-2, phase_sign=1.0, chunk_rows=2).compute(cov)

        assert full == pytest.approx(chunked, rel=1e-4)


class TestPreprocessConfigReader:
    def test_reads_window_and_method(self, tmp_path):
        _write_config_state(tmp_path, win=(40, 20), method="Capon")

        config = PreprocessConfigReader(tmp_path).read()

        assert config["window"] == (40, 20)
        assert config["beamforming_method"] == "Capon"

    def test_unknown_tomogram_key_raises(self, tmp_path):
        _write_config_state(tmp_path, win=(10, 10), key="output_configs")

        with pytest.raises(KeyError):
            PreprocessConfigReader(tmp_path).read()

    def test_missing_state_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PreprocessConfigReader(tmp_path).read()


class TestWindowResolution:
    def _baseline(self, tmp_path, **kwargs):
        _write_trainer_config(tmp_path, baselines=(0.0, 10.0))
        return ClassicalBaseline(tmp_path, NullLogger(), **kwargs)

    def test_window_from_preprocessing_config(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        _write_config_state(dataset_dir, win=(40, 20))
        baseline = self._baseline(tmp_path, preprocessing_dir=dataset_dir)

        assert baseline._resolve_window() == (40, 20)

    def test_explicit_window_overrides_preprocessing(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        _write_config_state(dataset_dir, win=(40, 20))
        baseline = self._baseline(tmp_path, preprocessing_dir=dataset_dir, window=(5, 5))

        assert baseline._resolve_window() == (5, 5)

    def test_no_window_and_no_dataset_raises(self, tmp_path):
        baseline = self._baseline(tmp_path)

        with pytest.raises(ValueError, match="capon_window"):
            baseline._resolve_window()

    def test_missing_config_state_raises(self, tmp_path):
        baseline = self._baseline(tmp_path, preprocessing_dir=tmp_path / "dataset")

        with pytest.raises(FileNotFoundError):
            baseline._resolve_window()


class TestClassicalBaseline:
    def _baseline(self, tmp_path, kz):
        scale     = 4.0 * math.pi / (0.23 * 5000.0)
        baselines = [float(k) / scale for k in kz]
        _write_trainer_config(tmp_path, baselines=baselines)
        return ClassicalBaseline(tmp_path, NullLogger(), window=(3, 3), loading=1e-2)

    def test_recovers_scatterer_height_normalized(self, tmp_path):
        kz       = [0.0, 0.05, 0.10, 0.15, 0.20]
        height   = 25.0
        x_axis   = np.linspace(-20.0, 80.0, 101)
        inputs   = _synthetic_inputs(kz, height)
        baseline = self._baseline(tmp_path, kz)

        reduced = baseline.compute(inputs, n_secondaries=4, x_axis=x_axis, secondary_labels=["PS04", "PS06", "PS08", "PS26"])

        assert reduced.shape == (101, 24, 24)
        assert reduced.max() == pytest.approx(1.0)
        peak_height = x_axis[reduced[:, 12, 12].argmax()]
        assert peak_height == pytest.approx(height, abs=float(x_axis[1] - x_axis[0]))

    def test_track_count_mismatch_raises(self, tmp_path):
        kz       = [0.0, 0.05, 0.10]
        inputs   = _synthetic_inputs([0.0, 0.05, 0.10, 0.15], 10.0)
        baseline = self._baseline(tmp_path, kz)

        with pytest.raises(ValueError, match="kz values"):
            baseline.compute(inputs, n_secondaries=3, x_axis=np.linspace(-20.0, 80.0, 11))

    def test_stack_uses_interferogram_channels(self, tmp_path):
        inputs       = _synthetic_inputs([0.0, 0.1], 5.0)
        inputs[1]    = 99.0 + 0.0j
        baseline     = self._baseline(tmp_path, [0.0, 0.1])
        stack        = baseline._build_stack(inputs, n_secondaries=1)

        assert stack.shape[0] == 2
        assert stack[0, 0, 0]  == pytest.approx(np.abs(inputs[0, 0, 0]))
        assert stack[1, 0, 0]  == pytest.approx(inputs[2, 0, 0])


class TestResultReduced:
    def _result(self, with_reduced=True):
        rng  = np.random.default_rng(0)
        gt   = rng.random((11, 6, 5)).astype(np.float32)
        pred = gt + 0.01 * rng.standard_normal((11, 6, 5)).astype(np.float32)

        maps   = Metrics.curve_pixel_metrics(pred, gt)
        result = Result(
            pred_curves        = pred,
            gt_curves          = gt,
            params_pred        = np.zeros((3, 6, 5), dtype=np.float32),
            params_gt          = np.zeros((3, 6, 5), dtype=np.float32),
            pixel_mse          = maps["mse"],
            pixel_mae          = maps["mae"],
            pixel_r2           = maps["r2"],
            pixel_cosine       = maps["cos"],
            pixel_peak_err_idx = maps["peak"],
            cube_directory     = Path("."),
            azimuth_offset     = 0,
            range_offset       = 0,
        )

        if with_reduced:
            reduced = gt + 0.1 * rng.standard_normal((11, 6, 5)).astype(np.float32)
            result.attach_reduced(reduced)

        return result

    def test_curve_pixel_metrics_match_definitions(self):
        rng  = np.random.default_rng(2)
        gt   = rng.random((7, 4, 3)).astype(np.float32)
        pred = rng.random((7, 4, 3)).astype(np.float32)

        maps = Metrics.curve_pixel_metrics(pred, gt)

        assert maps["mse"][1, 2] == pytest.approx(((pred[:, 1, 2] - gt[:, 1, 2]) ** 2).mean(), rel=1e-5)
        assert maps["mae"][0, 0] == pytest.approx(np.abs(pred[:, 0, 0] - gt[:, 0, 0]).mean(), rel=1e-5)
        assert maps["peak"][2, 1] == abs(int(pred[:, 2, 1].argmax()) - int(gt[:, 2, 1].argmax()))

    def test_attach_reduced_populates_maps(self):
        result = self._result()

        assert result.has_reduced
        assert result.pixel_mse_red.shape == (6, 5)
        assert result.pixel_improvement.shape == (6, 5)

    def test_improvement_positive_when_model_better(self):
        result = self._result()
        finite = result.pixel_improvement[np.isfinite(result.pixel_improvement)]

        assert (finite > 0.0).mean() > 0.9

    def test_metrics_include_reduced_suite(self):
        result  = self._result()
        metrics = Metrics(result, np.linspace(-20.0, 80.0, 11), n_gaussians=1).compute()

        assert "curve_mse_red" in metrics
        assert "psnr_db_red" in metrics
        assert "improvement_mse_rel" in metrics
        assert "pixel_improvement_positive_frac" in metrics
        assert "elev_mae_red_mean" in metrics
        assert metrics["improvement_mse_rel"] > 0.0
        assert metrics["curve_mse_red"] > metrics["curve_mse_gt"]

    def test_metrics_without_reduced_have_no_baseline_keys(self):
        result  = self._result(with_reduced=False)
        metrics = Metrics(result, np.linspace(-20.0, 80.0, 11), n_gaussians=1).compute()

        assert not result.has_reduced
        assert "curve_mse_red" not in metrics
        assert "improvement_mse_rel" not in metrics
