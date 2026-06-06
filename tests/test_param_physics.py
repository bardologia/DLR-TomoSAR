from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch

from configuration.training_config import GeometryConfig
from pipelines.param_pipeline.metrics import FittingMetricsCalculator
from pipelines.physics_pipeline.check import PhysicsQuantitiesCheck
from tools.gaussians import GaussianMixture
from tools.logger import NullLogger


def _make_params(n_gaussians: int, az: int, r: int, fill) -> np.ndarray:
    arr = np.zeros((3 * n_gaussians, az, r), dtype=np.float32)
    fill(arr)
    return arr


class TestFittingMetricsActivityMap:
    def test_all_active_when_amplitudes_above_threshold(self):
        calc   = FittingMetricsCalculator(n_gaussians=2, logger=NullLogger(), amp_threshold=1e-3)
        params = np.zeros((6, 2, 2), dtype=np.float32)
        params[0] = 1.0
        params[3] = 0.5

        active = calc._compute_activity_map(params)

        assert active.shape == (2, 2)
        assert active.dtype == np.int32
        assert np.all(active == 2)

    def test_no_active_when_amplitudes_below_threshold(self):
        calc   = FittingMetricsCalculator(n_gaussians=3, logger=NullLogger(), amp_threshold=1e-3)
        params = np.zeros((9, 2, 2), dtype=np.float32)

        active = calc._compute_activity_map(params)

        assert np.all(active == 0)

    def test_threshold_is_inclusive(self):
        calc   = FittingMetricsCalculator(n_gaussians=1, logger=NullLogger(), amp_threshold=1e-3)
        params = np.zeros((3, 1, 1), dtype=np.float32)
        params[0, 0, 0] = 1e-3

        active = calc._compute_activity_map(params)

        assert int(active[0, 0]) == 1

    def test_partial_activity_counts_per_pixel(self):
        calc   = FittingMetricsCalculator(n_gaussians=2, logger=NullLogger(), amp_threshold=1e-3)
        params = np.zeros((6, 1, 2), dtype=np.float32)
        params[0, 0, 0] = 1.0
        params[3, 0, 1] = 1.0
        params[0, 0, 1] = 1.0

        active = calc._compute_activity_map(params)

        assert int(active[0, 0]) == 1
        assert int(active[0, 1]) == 2


class TestFittingMetricsHeightAxis:
    def test_build_height_axis_endpoints_and_length(self):
        axis = FittingMetricsCalculator._build_height_axis((-30.0, 30.0), 61)

        assert axis.shape == (61,)
        assert axis.dtype == np.float32
        assert np.isclose(axis[0], -30.0)
        assert np.isclose(axis[-1], 30.0)

    def test_build_height_axis_single_point(self):
        axis = FittingMetricsCalculator._build_height_axis((5.0, 5.0), 1)

        assert axis.shape == (1,)
        assert np.isclose(axis[0], 5.0)


class TestFittingMetricsR2Map:
    def test_perfect_reconstruction_gives_r2_close_to_one(self):
        n_gaussians = 1
        az, r       = 2, 2
        height_axis = np.linspace(-20.0, 20.0, 41, dtype=np.float32)

        params       = np.zeros((3, az, r), dtype=np.float32)
        params[0]    = 1.0
        params[1]    = 0.0
        params[2]    = 4.0

        tomogram = np.zeros((height_axis.size, az, r), dtype=np.float32)
        for j in range(height_axis.size):
            tomogram[j] = GaussianMixture.evaluate_slice(params, float(height_axis[j]), n_gaussians)

        calc   = FittingMetricsCalculator(n_gaussians=n_gaussians, logger=NullLogger())
        r2_map = calc._compute_r2_map(tomogram, params, height_axis)

        assert r2_map.shape == (az, r)
        assert r2_map.dtype == np.float32
        assert np.all(r2_map > 0.999)

    def test_zero_residual_constant_tomogram_is_finite(self):
        az, r       = 1, 1
        height_axis = np.linspace(-5.0, 5.0, 11, dtype=np.float32)
        params      = np.zeros((3, az, r), dtype=np.float32)
        tomogram    = np.zeros((height_axis.size, az, r), dtype=np.float32)

        calc   = FittingMetricsCalculator(n_gaussians=1, logger=NullLogger())
        r2_map = calc._compute_r2_map(tomogram, params, height_axis)

        assert np.all(np.isfinite(r2_map))

    def test_poor_fit_gives_lower_r2_than_good_fit(self):
        az, r       = 1, 1
        height_axis = np.linspace(-20.0, 20.0, 81, dtype=np.float32)

        params_good      = np.zeros((3, az, r), dtype=np.float32)
        params_good[0]   = 1.0
        params_good[2]   = 4.0

        tomogram = np.zeros((height_axis.size, az, r), dtype=np.float32)
        for j in range(height_axis.size):
            tomogram[j] = GaussianMixture.evaluate_slice(params_good, float(height_axis[j]), 1)

        params_bad     = params_good.copy()
        params_bad[1]  = 15.0

        calc    = FittingMetricsCalculator(n_gaussians=1, logger=NullLogger())
        r2_good = calc._compute_r2_map(tomogram, params_good, height_axis)
        r2_bad  = calc._compute_r2_map(tomogram, params_bad,  height_axis)

        assert float(r2_good[0, 0]) > float(r2_bad[0, 0])


class TestFittingMetricsPerGaussianMaps:
    def test_inactive_entries_are_nan(self):
        calc   = FittingMetricsCalculator(n_gaussians=2, logger=NullLogger(), amp_threshold=1e-3)
        params = np.zeros((6, 1, 2), dtype=np.float32)
        params[0, 0, 0] = 2.0
        params[1, 0, 0] = 3.0
        params[2, 0, 0] = 1.5

        maps = calc._compute_per_gaussian_maps(params)

        assert set(maps.keys()) == {"amp_0", "mu_0", "sigma_0", "amp_1", "mu_1", "sigma_1"}
        assert np.isclose(maps["amp_0"][0, 0], 2.0)
        assert np.isclose(maps["mu_0"][0, 0], 3.0)
        assert np.isclose(maps["sigma_0"][0, 0], 1.5)
        assert np.isnan(maps["amp_0"][0, 1])
        assert np.all(np.isnan(maps["amp_1"]))

    def test_all_maps_float32(self):
        calc   = FittingMetricsCalculator(n_gaussians=1, logger=NullLogger())
        params = np.ones((3, 2, 2), dtype=np.float32)

        maps = calc._compute_per_gaussian_maps(params)

        for value in maps.values():
            assert value.dtype == np.float32


class TestFittingMetricsMuSeparation:
    def test_no_separation_maps_for_single_gaussian(self):
        calc   = FittingMetricsCalculator(n_gaussians=1, logger=NullLogger())
        params = np.ones((3, 1, 1), dtype=np.float32)

        maps = calc._compute_mu_separation_maps(params)

        assert maps == {}

    def test_separation_is_absolute_mu_difference_when_both_active(self):
        calc   = FittingMetricsCalculator(n_gaussians=2, logger=NullLogger(), amp_threshold=1e-3)
        params = np.zeros((6, 1, 1), dtype=np.float32)
        params[0, 0, 0] = 1.0
        params[1, 0, 0] = 2.0
        params[3, 0, 0] = 1.0
        params[4, 0, 0] = 9.0

        maps = calc._compute_mu_separation_maps(params)

        assert "mu_sep_0_1" in maps
        assert np.isclose(maps["mu_sep_0_1"][0, 0], 7.0)

    def test_separation_nan_when_one_inactive(self):
        calc   = FittingMetricsCalculator(n_gaussians=2, logger=NullLogger(), amp_threshold=1e-3)
        params = np.zeros((6, 1, 1), dtype=np.float32)
        params[0, 0, 0] = 1.0
        params[1, 0, 0] = 2.0
        params[4, 0, 0] = 9.0

        maps = calc._compute_mu_separation_maps(params)

        assert np.isnan(maps["mu_sep_0_1"][0, 0])


class TestFittingMetricsGlobalSummary:
    def test_summary_keys_and_active_fractions(self):
        calc         = FittingMetricsCalculator(n_gaussians=2, logger=NullLogger())
        r2_map       = np.array([[0.5, 0.9], [0.1, -0.2]], dtype=np.float32)
        activity_map = np.array([[0, 1], [2, 2]], dtype=np.int32)

        summary = calc._compute_global_summary(r2_map, activity_map)

        assert summary["n_pixels"] == 4.0
        assert summary["n_gaussians"] == 2.0
        assert np.isclose(summary["frac_0_active"], 0.25)
        assert np.isclose(summary["frac_1_active"], 0.25)
        assert np.isclose(summary["frac_2_active"], 0.5)
        assert np.isclose(summary["r2_neg_frac"], 0.25)

    def test_summary_handles_all_nan_r2(self):
        calc         = FittingMetricsCalculator(n_gaussians=1, logger=NullLogger())
        r2_map       = np.full((2, 2), np.nan, dtype=np.float32)
        activity_map = np.zeros((2, 2), dtype=np.int32)

        summary = calc._compute_global_summary(r2_map, activity_map)

        assert np.isnan(summary["r2_mean"])
        assert np.isnan(summary["r2_median"])
        assert np.isnan(summary["r2_p10"])
        assert np.isnan(summary["r2_neg_frac"])

    def test_summary_mean_median_match_numpy(self):
        calc         = FittingMetricsCalculator(n_gaussians=1, logger=NullLogger())
        r2_map       = np.array([[0.2, 0.4, 0.6, 0.8]], dtype=np.float32)
        activity_map = np.zeros((1, 4), dtype=np.int32)

        summary = calc._compute_global_summary(r2_map, activity_map)

        valid = r2_map.reshape(-1).astype(np.float64)
        assert np.isclose(summary["r2_mean"], valid.mean())
        assert np.isclose(summary["r2_median"], np.median(valid))


class TestFittingMetricsRun:
    def _write_dataset(self, tmp_path: Path, n_gaussians: int, height_range, n_elev: int, az: int, r: int):
        height_axis = np.linspace(height_range[0], height_range[1], n_elev, dtype=np.float32)

        params    = np.zeros((3 * n_gaussians, az, r), dtype=np.float32)
        params[0] = 1.0
        params[1] = 0.0
        params[2] = 4.0

        tomogram = np.zeros((n_elev, az, r), dtype=np.float32)
        for j in range(n_elev):
            tomogram[j] = GaussianMixture.evaluate_slice(params, float(height_axis[j]), n_gaussians)

        tomo_path = tmp_path / "tomogram.npy"
        np.save(tomo_path, tomogram)
        return params, tomo_path

    def test_run_returns_expected_keys_and_shapes(self, tmp_path):
        n_gaussians  = 1
        height_range = (-20.0, 20.0)
        az, r        = 3, 4
        params, tomo_path = self._write_dataset(tmp_path, n_gaussians, height_range, 41, az, r)

        calc   = FittingMetricsCalculator(n_gaussians=n_gaussians, logger=NullLogger())
        result = calc.run(params, {"height_range": list(height_range)}, tomo_path)

        assert "r2_map" in result
        assert "activity_map" in result
        assert "height_axis" in result
        assert "global_summary" in result
        assert result["r2_map"].shape == (az, r)
        assert result["activity_map"].shape == (az, r)
        assert result["height_axis"].shape == (41,)
        assert np.all(result["r2_map"] > 0.999)
        assert np.all(result["activity_map"] == 1)

    def test_run_two_gaussians_includes_separation_map(self, tmp_path):
        n_gaussians  = 2
        height_range = (-30.0, 30.0)
        az, r        = 2, 2
        n_elev       = 61

        height_axis = np.linspace(height_range[0], height_range[1], n_elev, dtype=np.float32)
        params      = np.zeros((6, az, r), dtype=np.float32)
        params[0] = 1.0
        params[1] = -8.0
        params[2] = 3.0
        params[3] = 0.8
        params[4] = 8.0
        params[5] = 3.0

        tomogram = np.zeros((n_elev, az, r), dtype=np.float32)
        for j in range(n_elev):
            tomogram[j] = GaussianMixture.evaluate_slice(params, float(height_axis[j]), n_gaussians)

        tomo_path = tmp_path / "tomo2.npy"
        np.save(tomo_path, tomogram)

        calc   = FittingMetricsCalculator(n_gaussians=n_gaussians, logger=NullLogger())
        result = calc.run(params, {"height_range": list(height_range)}, tomo_path)

        assert "mu_sep_0_1" in result
        assert np.allclose(result["mu_sep_0_1"], 16.0)
        assert np.all(result["activity_map"] == 2)


@dataclass
class _PhysicsCfg:
    dataset_path      : Path
    tomogram_filename : str
    height_range      : tuple
    fit_k_max         : int   = 1
    output_prefix     : str   = "params"
    output_suffix     : str | None = None
    n_pixels          : int   = 8
    seed              : int   = 0
    device            : str   = "cpu"
    physics_floor     : float = 1e-3
    capon_loading     : float = 1e-2
    moments_weights   : tuple = (1.0, 1.0, 1.0)
    geometry          : GeometryConfig = field(default_factory=lambda: GeometryConfig(baselines=(0.0, 20.0, 40.0)))


def _build_check(tmp_path: Path, n_gaussians: int = 1, height_range=(-20.0, 20.0), n_elev: int = 41, az: int = 4, r: int = 4, n_pixels: int = 6):
    data_dir = tmp_path / "data"
    meta_dir = tmp_path / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    height_axis = np.linspace(height_range[0], height_range[1], n_elev, dtype=np.float32)

    params      = np.zeros((3 * n_gaussians, az, r), dtype=np.float32)
    params[0]   = 0.5
    params[1]   = 0.0
    params[2]   = 5.0

    tomogram = np.zeros((n_elev, az, r), dtype=np.float32)
    for j in range(n_elev):
        tomogram[j] = GaussianMixture.evaluate_slice(params, float(height_axis[j]), n_gaussians)

    tomo_name = "tomogram_full_test.npy"
    np.save(data_dir / tomo_name, tomogram)

    cfg = _PhysicsCfg(
        dataset_path      = tmp_path,
        tomogram_filename = tomo_name,
        height_range      = height_range,
        fit_k_max         = n_gaussians,
        n_pixels          = n_pixels,
    )

    from configuration.param_extraction_config import ExtractionConfig, FitMode, FitSettings

    extraction = ExtractionConfig(
        processed_data_path = cfg.dataset_path,
        tomogram_filename   = cfg.tomogram_filename,
        height_range        = cfg.height_range,
        output_prefix       = cfg.output_prefix,
        output_suffix       = cfg.output_suffix,
        fit_settings        = FitSettings(fit_config=FitMode.SigmaOnly(k_max=cfg.fit_k_max)),
    )
    params_path = extraction.parameters_npy_path
    params_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(params_path, params)

    check = PhysicsQuantitiesCheck(cfg, NullLogger())
    return check, params, tomogram, height_axis


class TestPhysicsCheckConstruction:
    def test_device_is_cpu_without_cuda(self):
        cfg   = _PhysicsCfg(dataset_path=Path("/tmp"), tomogram_filename="x.npy", height_range=(-1.0, 1.0))
        check = PhysicsQuantitiesCheck(cfg, NullLogger())

        assert check.device.type == "cpu"


class TestPhysicsCheckLoadProfiles:
    def test_load_profiles_shapes(self, tmp_path):
        check, params, tomogram, height_axis = _build_check(tmp_path, n_pixels=6, az=4, r=4)

        capon, gauss, x_axis = check._load_profiles()

        n_bins = tomogram.shape[0]
        assert x_axis.shape == (n_bins,)
        assert capon.shape[1] == n_bins
        assert gauss.shape[1] == n_bins
        assert capon.shape[0] == gauss.shape[0]
        assert capon.shape[0] <= 6

    def test_load_profiles_gauss_matches_mixture(self, tmp_path):
        check, params, tomogram, height_axis = _build_check(tmp_path, n_pixels=4, az=2, r=2)

        capon, gauss, x_axis = check._load_profiles()

        assert np.all(np.isfinite(gauss))
        assert np.all(gauss >= 0.0)

    def test_load_profiles_deterministic_with_seed(self, tmp_path):
        check, _, _, _ = _build_check(tmp_path, n_pixels=4, az=3, r=3)

        c1, g1, x1 = check._load_profiles()
        c2, g2, x2 = check._load_profiles()

        assert np.allclose(c1, c2)
        assert np.allclose(g1, g2)


class TestPhysicsCheckTensorReshape:
    def test_to_tensor_layout(self, tmp_path):
        check, _, _, _ = _build_check(tmp_path)
        profiles = np.random.default_rng(0).random((5, 7)).astype(np.float32)

        t = check._to_tensor(profiles)

        assert t.shape == (1, 7, 1, 5)
        assert t.device.type == "cpu"
        assert np.isclose(float(t[0, 3, 0, 2]), profiles[2, 3])


class TestPhysicsCheckLossTerms:
    def test_loss_terms_rows_and_keys(self, tmp_path):
        check, _, _, _ = _build_check(tmp_path)
        capon, gauss, x_axis = check._load_profiles()

        x_t     = torch.tensor(x_axis, dtype=torch.float32)
        dx      = float(x_axis[1] - x_axis[0])
        capon_t = check._to_tensor(capon)
        gauss_t = check._to_tensor(gauss)

        from tools.tomo_geometry import TomoGeometry
        geometry = TomoGeometry(check.config.geometry, x_t)

        rows = check._loss_terms(gauss_t, capon_t, x_t, dx, geometry)

        names = [row["Term"] for row in rows]
        assert names == ["total_power", "moments", "coherence_resyn", "covariance_match", "capon_cycle"]
        for row in rows:
            assert "Gauss vs Capon" in row
            assert np.isfinite(float(row["Gauss vs Capon"]))


class TestPhysicsCheckMomentAgreement:
    def test_moment_rows_for_identical_profiles(self, tmp_path):
        check, _, _, _ = _build_check(tmp_path)
        capon, gauss, x_axis = check._load_profiles()

        x_t     = torch.tensor(x_axis, dtype=torch.float32)
        dx      = float(x_axis[1] - x_axis[0])
        capon_t = check._to_tensor(capon)

        rows = check._moment_agreement(capon_t, capon_t, x_t, dx)

        quantities = [row["Quantity"] for row in rows]
        assert quantities == ["mass m0", "mean elevation", "vertical spread"]
        for row in rows:
            assert np.isclose(float(row["MAE"]), 0.0, atol=1e-4)

    def test_moment_rows_have_all_columns(self, tmp_path):
        check, _, _, _ = _build_check(tmp_path)
        capon, gauss, x_axis = check._load_profiles()

        x_t     = torch.tensor(x_axis, dtype=torch.float32)
        dx      = float(x_axis[1] - x_axis[0])
        capon_t = check._to_tensor(capon)
        gauss_t = check._to_tensor(gauss)

        rows = check._moment_agreement(gauss_t, capon_t, x_t, dx)

        for row in rows:
            assert set(row.keys()) == {"Quantity", "MAE", "Pearson r", "Signed mean rel"}


class TestPhysicsCheckCoherenceAgreement:
    def test_coherence_rows_one_per_track(self, tmp_path):
        check, _, _, _ = _build_check(tmp_path)
        capon, gauss, x_axis = check._load_profiles()

        x_t     = torch.tensor(x_axis, dtype=torch.float32)
        dx      = float(x_axis[1] - x_axis[0])
        capon_t = check._to_tensor(capon)
        gauss_t = check._to_tensor(gauss)

        from tools.tomo_geometry import TomoGeometry
        geometry = TomoGeometry(check.config.geometry, x_t)

        rows = check._coherence_agreement(gauss_t, capon_t, dx, geometry)

        assert len(rows) == geometry.n_tracks
        for n, row in enumerate(rows):
            assert row["Track"] == str(n)
            assert np.isfinite(float(row["kz [rad/m]"]))
            assert np.isfinite(float(row["Mean |dGamma|"]))

    def test_coherence_zero_for_identical_profiles(self, tmp_path):
        check, _, _, _ = _build_check(tmp_path)
        capon, gauss, x_axis = check._load_profiles()

        x_t     = torch.tensor(x_axis, dtype=torch.float32)
        dx      = float(x_axis[1] - x_axis[0])
        capon_t = check._to_tensor(capon)

        from tools.tomo_geometry import TomoGeometry
        geometry = TomoGeometry(check.config.geometry, x_t)

        rows = check._coherence_agreement(capon_t, capon_t, dx, geometry)

        for row in rows:
            assert np.isclose(float(row["Mean |dGamma|"]), 0.0, atol=1e-5)


class TestPhysicsCheckRun:
    def test_run_returns_three_sections(self, tmp_path):
        check, _, _, _ = _build_check(tmp_path, n_pixels=6, az=4, r=4)

        result = check.run()

        assert set(result.keys()) == {"losses", "moments", "coherence"}
        assert len(result["losses"]) == 5
        assert len(result["moments"]) == 3
        assert len(result["coherence"]) == len(check.config.geometry.baselines)

    def test_run_loss_values_finite(self, tmp_path):
        check, _, _, _ = _build_check(tmp_path, n_pixels=6, az=4, r=4)

        result = check.run()

        for row in result["losses"]:
            assert np.isfinite(float(row["Gauss vs Capon"]))
        for row in result["moments"]:
            assert np.isfinite(float(row["MAE"]))


try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except Exception:
    jax  = None
    jnp  = None
    _HAS_JAX = False

requires_jax = pytest.mark.skipif(not _HAS_JAX, reason="jax is not installed")


@requires_jax
class TestSigmaScanPerPixelLoss:
    def test_zero_loss_for_matching_profile(self):
        from pipelines.param_pipeline.sigma import SigmaScan

        height_axis = jnp.linspace(-20.0, 20.0, 41)
        amps        = jnp.array([1.0])
        mus         = jnp.array([0.0])
        sigmas      = jnp.array([4.0])

        diff    = height_axis[None, :] - mus[:, None]
        safe_s2 = 2.0 * jnp.maximum(sigmas, 1e-6) ** 2
        profile = (amps[:, None] * jnp.exp(jnp.clip(-(diff ** 2) / safe_s2[:, None], -100.0, 0.0))).sum(axis=0)

        loss = SigmaScan.per_pixel_loss(sigmas, height_axis, profile, amps, mus)

        assert float(loss) < 1e-10

    def test_loss_is_nonnegative(self):
        from pipelines.param_pipeline.sigma import SigmaScan

        height_axis = jnp.linspace(-10.0, 10.0, 21)
        profile     = jnp.ones(21)
        amps        = jnp.array([0.5])
        mus         = jnp.array([1.0])
        sigmas      = jnp.array([2.0])

        loss = SigmaScan.per_pixel_loss(sigmas, height_axis, profile, amps, mus)

        assert float(loss) >= 0.0


@requires_jax
class TestSigmaAdamKernel:
    def test_adam_recovers_known_sigma(self):
        import numpy as _np

        from pipelines.param_pipeline.sigma import SigmaAdamKernel

        height_axis = jnp.linspace(-20.0, 20.0, 81)
        true_sigma  = 5.0
        amps        = jnp.array([[1.0]])
        mus         = jnp.array([[0.0]])

        diff    = height_axis[None, :] - mus[0][:, None]
        safe_s2 = 2.0 * true_sigma ** 2
        profile = (amps[0][:, None] * jnp.exp(-(diff ** 2) / safe_s2)).sum(axis=0)[None, :]

        kernel = SigmaAdamKernel()
        out    = kernel(
            jnp.array([[2.0]], dtype=jnp.float32),
            height_axis,
            profile.astype(jnp.float32),
            amps.astype(jnp.float32),
            mus.astype(jnp.float32),
            jnp.float32(0.5),
            jnp.float32(20.0),
            n_steps = 500,
            lr      = 1e-1,
        )

        assert out.shape == (1, 1)
        assert _np.isclose(float(out[0, 0]), true_sigma, atol=0.5)

    def test_adam_respects_bounds(self):
        from pipelines.param_pipeline.sigma import SigmaAdamKernel

        height_axis = jnp.linspace(-20.0, 20.0, 41)
        amps        = jnp.array([[1.0]])
        mus         = jnp.array([[0.0]])
        profile     = jnp.ones((1, 41), dtype=jnp.float32)

        lower = jnp.float32(3.0)
        upper = jnp.float32(6.0)

        kernel = SigmaAdamKernel()
        out    = kernel(
            jnp.array([[100.0]], dtype=jnp.float32),
            height_axis,
            profile,
            amps.astype(jnp.float32),
            mus.astype(jnp.float32),
            lower,
            upper,
            n_steps = 50,
            lr      = 1e-1,
        )

        assert float(lower) - 1e-4 <= float(out[0, 0]) <= float(upper) + 1e-4


@requires_jax
class TestPeakInitialiser:
    def test_single_peak_recovered(self):
        from pipelines.param_pipeline.sigma import PeakInitialiser

        height_axis = np.linspace(-20.0, 20.0, 81, dtype=np.float32)
        peak_mu     = 5.0
        prof        = np.exp(-((height_axis - peak_mu) ** 2) / (2.0 * 3.0 ** 2)).astype(np.float32)
        prof        = prof[None, :]

        init = PeakInitialiser()
        amps, mus, sigs = init.run(prof, height_axis, K=1, prominence_frac=0.05, n_workers=1)

        assert amps.shape == (1, 1)
        assert mus.shape == (1, 1)
        assert sigs.shape == (1, 1)
        assert np.abs(mus[0, 0] - peak_mu) <= 1.0
        assert amps[0, 0] > 0.0

    def test_two_peaks_recovered(self):
        from pipelines.param_pipeline.sigma import PeakInitialiser

        height_axis = np.linspace(-30.0, 30.0, 121, dtype=np.float32)
        prof        = np.exp(-((height_axis + 10.0) ** 2) / (2.0 * 3.0 ** 2))
        prof       += np.exp(-((height_axis - 10.0) ** 2) / (2.0 * 3.0 ** 2))
        prof        = prof.astype(np.float32)[None, :]

        init = PeakInitialiser()
        amps, mus, sigs = init.run(prof, height_axis, K=2, prominence_frac=0.05, n_workers=1)

        assert amps.shape == (1, 2)
        found = sorted(float(m) for m in mus[0])
        assert np.abs(found[0] - (-10.0)) <= 2.0
        assert np.abs(found[1] - 10.0) <= 2.0

    def test_flat_profile_falls_back_to_linspace(self):
        from pipelines.param_pipeline.sigma import PeakInitialiser

        height_axis = np.linspace(-10.0, 10.0, 41, dtype=np.float32)
        prof        = np.zeros((1, 41), dtype=np.float32)[None, :].reshape(1, 41)

        init = PeakInitialiser()
        amps, mus, sigs = init.run(prof, height_axis, K=2, prominence_frac=0.05, n_workers=1)

        assert amps.shape == (1, 2)
        assert np.all(amps >= 1e-10)
        assert np.all(np.isfinite(mus))

    def test_sigma_divisor_shrinks_initial_sigma(self):
        from pipelines.param_pipeline.sigma import PeakInitialiser

        height_axis = np.linspace(-20.0, 20.0, 81, dtype=np.float32)
        prof        = np.exp(-(height_axis ** 2) / (2.0 * 3.0 ** 2)).astype(np.float32)[None, :]

        init = PeakInitialiser()
        _, _, sigs_base    = init.run(prof, height_axis, K=1, prominence_frac=0.05, n_workers=1)
        _, mus, sigs_small = init.run(prof, height_axis, K=1, prominence_frac=0.05, n_workers=1, sigma_divisor=4.0)

        assert np.allclose(sigs_small, sigs_base / 4.0)
        assert np.abs(mus[0, 0]) <= 1.0


@requires_jax
class TestBestKSelector:
    def test_select_picks_lower_mse_k(self):
        from pipelines.param_pipeline.sigma import BestKSelector

        height_axis = np.linspace(-20.0, 20.0, 41, dtype=np.float32)

        amps1 = np.array([[1.0]], dtype=np.float32)
        mus1  = np.array([[0.0]], dtype=np.float32)
        sig1  = np.array([[4.0]], dtype=np.float32)

        target = GaussianMixture.evaluate_batch(height_axis, amps1, mus1, sig1).astype(np.float32)

        amps2 = np.array([[1.0, 0.0]], dtype=np.float32)
        mus2  = np.array([[0.0, 0.0]], dtype=np.float32)
        sig2  = np.array([[4.0, 4.0]], dtype=np.float32)

        gpu_results = {
            1: (amps1, mus1, sig1),
            2: (amps2, mus2, sig2),
        }

        scale_all = np.array([1.0], dtype=np.float32)
        selector  = BestKSelector(k_max=2, lambda_k=3e-3, logger=NullLogger())

        best = selector.select(gpu_results, target, scale_all, height_axis, n_params_out=6)

        assert best.shape == (1, 6)
        assert np.isclose(best[0, 0], 1.0, atol=1e-4)
        assert np.isclose(best[0, 2], 4.0, atol=1e-4)

    def test_select_scales_amplitudes(self):
        from pipelines.param_pipeline.sigma import BestKSelector

        height_axis = np.linspace(-20.0, 20.0, 41, dtype=np.float32)

        amps1 = np.array([[1.0]], dtype=np.float32)
        mus1  = np.array([[0.0]], dtype=np.float32)
        sig1  = np.array([[4.0]], dtype=np.float32)

        target = GaussianMixture.evaluate_batch(height_axis, amps1, mus1, sig1).astype(np.float32)

        gpu_results = {1: (amps1, mus1, sig1)}
        scale_all   = np.array([3.0], dtype=np.float32)
        selector    = BestKSelector(k_max=1, lambda_k=3e-3, logger=NullLogger())

        best = selector.select(gpu_results, target, scale_all, height_axis, n_params_out=3)

        assert np.isclose(best[0, 0], 3.0, atol=1e-4)
