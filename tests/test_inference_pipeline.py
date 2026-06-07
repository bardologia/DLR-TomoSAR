from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib     import Path
from types       import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import numpy  as np
import pytest

from pipelines.dataset_pipeline.spatial      import Patcher
from pipelines.inference_pipeline.metrics    import Metrics, Result
from pipelines.inference_pipeline.predictor  import CubeStitcher, Predictor
from pipelines.inference_pipeline.report     import Report, ReportPayloadBuilder
from pipelines.inference_pipeline.plots      import Ploter, PlotTools
from pipelines.inference_pipeline.figures    import Animator
from tools.track_baselines                   import TrackBaselines, TrackProfiles


def _make_result(
    n_elev : int = 5,
    H      : int = 3,
    W      : int = 3,
    n_K    : int = 2,
    *,
    seed   : int = 0,
) -> Result:
    rng = np.random.default_rng(seed)

    pred_curves = rng.random((n_elev, H, W)).astype(np.float32)
    gt_curves   = rng.random((n_elev, H, W)).astype(np.float32)

    params_pred = rng.random((n_K * 3, H, W)).astype(np.float32)
    params_gt   = rng.random((n_K * 3, H, W)).astype(np.float32)

    for k in range(n_K):
        params_pred[3 * k] = np.abs(params_pred[3 * k]) + 0.5
        params_gt  [3 * k] = np.abs(params_gt  [3 * k]) + 0.5

    pixel_mse  = rng.random((H, W)).astype(np.float32)
    pixel_mae  = rng.random((H, W)).astype(np.float32)
    pixel_r2   = rng.random((H, W)).astype(np.float32)
    pixel_cos  = rng.random((H, W)).astype(np.float32)
    pixel_peak = rng.integers(0, n_elev, size=(H, W)).astype(np.int32)

    return Result(
        pred_curves        = pred_curves,
        gt_curves          = gt_curves,
        params_pred        = params_pred,
        params_gt          = params_gt,
        pixel_mse          = pixel_mse,
        pixel_mae          = pixel_mae,
        pixel_r2           = pixel_r2,
        pixel_cosine       = pixel_cos,
        pixel_peak_err_idx = pixel_peak,
        cube_directory     = Path("/tmp"),
        azimuth_offset     = 0,
        range_offset       = 0,
    )


class TestMetricsStaticHelpers:
    def test_percentiles_returns_expected_keys(self):
        x   = np.arange(100, dtype=np.float64)
        out = Metrics._percentiles(x)

        assert set(out.keys()) == {"p1", "p5", "p25", "p50", "p75", "p95", "p99"}
        assert np.isclose(out["p50"], np.percentile(x, 50))

    def test_percentiles_custom_quantiles(self):
        x   = np.linspace(0.0, 1.0, 11, dtype=np.float64)
        out = Metrics._percentiles(x, qs=(0, 100))

        assert np.isclose(out["p0"], 0.0)
        assert np.isclose(out["p100"], 1.0)

    def test_basic_stats_matches_numpy(self):
        x   = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        out = Metrics._basic_stats(x)

        assert np.isclose(out["mean"], 2.5)
        assert np.isclose(out["median"], 2.5)
        assert np.isclose(out["min"], 1.0)
        assert np.isclose(out["max"], 4.0)
        assert np.isclose(out["std"], float(x.std()))

    def test_basic_stats_handles_multidim(self):
        x   = np.ones((3, 4), dtype=np.float64) * 2.0
        out = Metrics._basic_stats(x)

        assert np.isclose(out["mean"], 2.0)
        assert np.isclose(out["std"], 0.0)

    def test_psnr_zero_error_is_inf(self):
        ref = np.array([0.0, 1.0, 2.0], dtype=np.float64)

        assert Metrics._psnr(ref.copy(), ref) == float("inf")

    def test_psnr_zero_range_is_nan(self):
        ref  = np.zeros(4, dtype=np.float64)
        pred = np.ones(4, dtype=np.float64)

        assert np.isnan(Metrics._psnr(pred, ref))

    def test_psnr_known_value(self):
        ref  = np.array([0.0, 1.0], dtype=np.float64)
        pred = np.array([0.0, 0.0], dtype=np.float64)
        mse  = 0.5
        expected = 10.0 * np.log10(1.0 / mse)

        assert np.isclose(Metrics._psnr(pred, ref), expected)

    def test_write_json_creates_file_and_returns_path(self, tmp_path):
        payload = {"a": 1, "b": 2.5, "c": "text"}
        path    = tmp_path / "nested" / "metrics.json"

        returned = Metrics.write_json(payload, path)

        assert returned == path
        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == payload

    def test_write_json_serialises_non_native_with_default_str(self, tmp_path):
        payload = {"path": Path("/tmp/x")}
        path    = tmp_path / "m.json"

        Metrics.write_json(payload, path)
        loaded = json.loads(path.read_text(encoding="utf-8"))

        assert loaded["path"] == "/tmp/x"


class TestMetricsSelectPixels:
    def test_select_pixels_partitions_disjoint(self):
        rng        = np.random.default_rng(1)
        metric_map = rng.random((6, 6)).astype(np.float32)

        sel = Metrics.select_pixels(metric_map, n_best=3, n_worst=3, n_random=4, seed=7)

        best   = {tuple(p) for p in sel["best"]}
        worst  = {tuple(p) for p in sel["worst"]}
        rand   = {tuple(p) for p in sel["random"]}

        assert len(best) == 3
        assert len(worst) == 3
        assert len(rand) == 4
        assert best.isdisjoint(worst)
        assert rand.isdisjoint(best)
        assert rand.isdisjoint(worst)

    def test_select_pixels_best_are_smallest(self):
        metric_map = np.arange(16, dtype=np.float32).reshape(4, 4)

        sel = Metrics.select_pixels(metric_map, n_best=2, n_worst=2, n_random=0, seed=0)

        best_vals  = sorted(metric_map[y, x] for y, x in sel["best"])
        worst_vals = sorted(metric_map[y, x] for y, x in sel["worst"])

        assert best_vals == [0.0, 1.0]
        assert worst_vals == [14.0, 15.0]

    def test_select_pixels_yx_within_bounds(self):
        metric_map = np.random.default_rng(2).random((5, 7)).astype(np.float32)

        sel = Metrics.select_pixels(metric_map, n_best=2, n_worst=2, n_random=3, seed=0)

        for group in sel.values():
            for y, x in group:
                assert 0 <= y < 5
                assert 0 <= x < 7

    def test_select_pixels_ignores_non_finite(self):
        metric_map        = np.arange(16, dtype=np.float32).reshape(4, 4)
        metric_map[0, 0]  = np.nan
        metric_map[3, 3]  = np.inf

        sel = Metrics.select_pixels(metric_map, n_best=2, n_worst=2, n_random=2, seed=0)

        all_pts = [tuple(p) for grp in sel.values() for p in grp]
        assert (0, 0) not in all_pts
        assert (3, 3) not in all_pts

    def test_select_pixels_random_capped_to_pool(self):
        metric_map = np.arange(9, dtype=np.float32).reshape(3, 3)

        sel = Metrics.select_pixels(metric_map, n_best=4, n_worst=4, n_random=100, seed=0)

        assert sel["random"].shape[0] <= 9

    def test_select_pixels_zero_random_returns_empty(self):
        metric_map = np.arange(16, dtype=np.float32).reshape(4, 4)

        sel = Metrics.select_pixels(metric_map, n_best=1, n_worst=1, n_random=0, seed=0)

        assert sel["random"].shape[0] == 0

    def test_select_pixels_deterministic_with_seed(self):
        metric_map = np.random.default_rng(5).random((6, 6)).astype(np.float32)

        a = Metrics.select_pixels(metric_map, n_best=1, n_worst=1, n_random=5, seed=42)
        b = Metrics.select_pixels(metric_map, n_best=1, n_worst=1, n_random=5, seed=42)

        assert np.array_equal(a["random"], b["random"])


class TestMetricsInstanceMethods:
    def _metrics(self, **kw) -> Metrics:
        res    = _make_result(**kw)
        n_elev = res.pred_curves.shape[0]
        x_axis = np.linspace(-10.0, 10.0, n_elev).astype(np.float32)
        n_K    = res.params_pred.shape[0] // 3

        return Metrics(res, x_axis, n_K)

    def test_x_step_computed_from_axis(self):
        m = self._metrics(n_elev=5)

        assert np.isclose(m.x_step, 20.0 / 4.0)

    def test_elev_metrics_keys_and_length(self):
        m   = self._metrics(n_elev=5, H=3, W=3)
        out = m._elev_metrics(m.result.pred_curves, m.result.gt_curves)

        assert set(out.keys()) == {"elev_mae_gt", "elev_rmse_gt", "elev_r2_gt", "elev_ce_gt"}
        for arr in out.values():
            assert arr.shape == (5,)

    def test_elev_metrics_perfect_prediction(self):
        res    = _make_result(n_elev=5, H=3, W=3)
        res.pred_curves[...] = res.gt_curves
        x_axis = np.linspace(-10.0, 10.0, 5).astype(np.float32)
        m      = Metrics(res, x_axis, 2)

        out = m._elev_metrics(res.pred_curves, res.gt_curves)

        assert np.allclose(out["elev_mae_gt"], 0.0)
        assert np.allclose(out["elev_rmse_gt"], 0.0)
        assert np.allclose(out["elev_r2_gt"], 1.0, atol=1e-6)

    def test_mu_ordering_rate_single_gaussian_is_nan(self):
        res    = _make_result(n_elev=4, n_K=1)
        x_axis = np.linspace(-5.0, 5.0, 4).astype(np.float32)
        m      = Metrics(res, x_axis, 1)

        assert np.isnan(m._mu_ordering_rate())

    def test_mu_ordering_rate_fully_ordered_is_one(self):
        res = _make_result(n_elev=4, H=2, W=2, n_K=2)
        res.params_pred[0][...] = 1.0
        res.params_pred[3][...] = 1.0
        res.params_pred[1][...] = -2.0
        res.params_pred[4][...] = 5.0
        x_axis = np.linspace(-5.0, 5.0, 4).astype(np.float32)
        m      = Metrics(res, x_axis, 2)

        assert np.isclose(m._mu_ordering_rate(), 1.0)

    def test_mu_ordering_rate_violation_is_zero(self):
        res = _make_result(n_elev=4, H=2, W=2, n_K=2)
        res.params_pred[0][...] = 1.0
        res.params_pred[3][...] = 1.0
        res.params_pred[1][...] = 5.0
        res.params_pred[4][...] = -2.0
        x_axis = np.linspace(-5.0, 5.0, 4).astype(np.float32)
        m      = Metrics(res, x_axis, 2)

        assert np.isclose(m._mu_ordering_rate(), 0.0)

    def test_slot_mu_stats_keys(self):
        m   = self._metrics(n_K=2)
        out = m._slot_mu_stats()

        for k in range(2):
            assert f"slot_{k}_mu_pred_mean" in out
            assert f"slot_{k}_mu_pred_std" in out
            assert f"slot_{k}_mu_gt_mean" in out
            assert f"slot_{k}_mu_gt_std" in out

    def test_slot_mu_stats_inactive_slot_is_nan(self):
        res = _make_result(n_K=2)
        res.params_gt[0][...] = 0.0
        x_axis = np.linspace(-5.0, 5.0, res.pred_curves.shape[0]).astype(np.float32)
        m      = Metrics(res, x_axis, 2)

        out = m._slot_mu_stats()

        assert np.isnan(out["slot_0_mu_gt_mean"])
        assert np.isnan(out["slot_0_mu_pred_mean"])

    def test_placeholder_detection_keys_present(self):
        m   = self._metrics(n_K=2)
        out = m._placeholder_detection()

        assert "placeholder_precision" in out
        assert "placeholder_recall" in out
        assert "placeholder_f1" in out
        for k in range(2):
            assert f"slot_{k}_placeholder_f1" in out

    def test_placeholder_detection_perfect_match(self):
        res = _make_result(n_K=2)
        res.params_gt[0][...]   = 0.0
        res.params_pred[0][...] = 0.0
        x_axis = np.linspace(-5.0, 5.0, res.pred_curves.shape[0]).astype(np.float32)
        m      = Metrics(res, x_axis, 2)

        out = m._placeholder_detection()

        assert np.isclose(out["slot_0_placeholder_precision"], 1.0, atol=1e-4)
        assert np.isclose(out["slot_0_placeholder_recall"], 1.0, atol=1e-4)

    def test_permutation_consensus_single_gaussian(self):
        res    = _make_result(n_K=1)
        x_axis = np.linspace(-5.0, 5.0, res.pred_curves.shape[0]).astype(np.float32)
        m      = Metrics(res, x_axis, 1)

        out = m._permutation_consensus()

        assert out["permutation_consensus_dominant_frac"] == 1.0
        assert out["permutation_consensus_identity_frac"] == 1.0

    def test_permutation_consensus_fractions_in_unit_interval(self):
        m   = self._metrics(n_K=3, H=4, W=4)
        out = m._permutation_consensus()

        for v in out.values():
            assert 0.0 <= v <= 1.0

    def test_permutation_consensus_identity_when_sorted(self):
        res = _make_result(n_K=2, H=2, W=2)
        res.params_pred[1][...] = 0.0
        res.params_pred[4][...] = 5.0
        res.params_gt[1][...]   = 0.0
        res.params_gt[4][...]   = 5.0
        x_axis = np.linspace(-5.0, 5.0, res.pred_curves.shape[0]).astype(np.float32)
        m      = Metrics(res, x_axis, 2)

        out = m._permutation_consensus()

        assert np.isclose(out["permutation_consensus_identity_frac"], 1.0)

    def test_gaussian_param_metrics_perfect(self):
        res = _make_result(n_K=2)
        res.params_pred[...] = res.params_gt
        x_axis = np.linspace(-5.0, 5.0, res.pred_curves.shape[0]).astype(np.float32)
        m      = Metrics(res, x_axis, 2)

        out = m._gaussian_param_metrics()

        assert np.isclose(out["gauss_all_mu_mae"], 0.0)
        assert np.isclose(out["gauss_all_sig_mae"], 0.0)
        assert out["gauss_0_n_valid"] > 0

    def test_gaussian_param_metrics_all_inactive_is_nan(self):
        res = _make_result(n_K=2)
        res.params_gt[0][...] = 0.0
        res.params_gt[3][...] = 0.0
        x_axis = np.linspace(-5.0, 5.0, res.pred_curves.shape[0]).astype(np.float32)
        m      = Metrics(res, x_axis, 2)

        out = m._gaussian_param_metrics()

        assert np.isnan(out["gauss_all_mu_mae"])
        assert out["gauss_0_n_valid"] == 0

    def test_best_perm_bruteforce_recovers_identity(self):
        cost = np.zeros((1, 2, 2), dtype=np.float64)
        cost[0] = np.array([[0.0, 1.0], [1.0, 0.0]])
        perms  = [(0, 1), (1, 0)]

        idx = Metrics._best_perm_bruteforce(cost, perms, 2)

        assert idx[0] == 0

    def test_best_perm_assignment_recovers_swap(self):
        pytest.importorskip("scipy")
        cost = np.zeros((1, 2, 2), dtype=np.float64)
        cost[0] = np.array([[1.0, 0.0], [0.0, 1.0]])
        perms  = [(0, 1), (1, 0)]

        idx = Metrics._best_perm_assignment(cost, perms)

        assert idx[0] == 1


class TestMetricsCompute:
    def _metrics(self, **kw):
        res    = _make_result(**kw)
        n_elev = res.pred_curves.shape[0]
        x_axis = np.linspace(-10.0, 10.0, n_elev).astype(np.float32)
        n_K    = res.params_pred.shape[0] // 3
        return Metrics(res, x_axis, n_K), n_elev

    def test_compute_with_empty_indices(self):
        m, _   = self._metrics(n_elev=5, H=3, W=3, n_K=2)
        empty  = np.array([], dtype=np.int64)

        out = m.compute(elev_indices=empty, range_indices=empty, az_indices=empty)

        assert out["n_pixels"] == 9
        assert out["n_elevation"] == 5
        assert "curve_mse_gt" in out
        assert "mu_ordering_rate" in out
        assert "overall_r2_gt" in out

    def test_compute_basic_curve_values(self):
        res = _make_result(n_elev=5, H=3, W=3, n_K=2)
        res.pred_curves[...] = res.gt_curves
        x_axis = np.linspace(-10.0, 10.0, 5).astype(np.float32)
        m      = Metrics(res, x_axis, 2)
        empty  = np.array([], dtype=np.int64)

        out = m.compute(elev_indices=empty, range_indices=empty, az_indices=empty)

        assert np.isclose(out["curve_mse_gt"], 0.0)
        assert np.isclose(out["curve_mae_gt"], 0.0)
        assert np.isclose(out["overall_r2_gt"], 1.0, atol=1e-5)
        assert out["psnr_db_gt"] == float("inf")

    def test_compute_runs_ssim_with_indices(self):
        m, n_elev = self._metrics(n_elev=4, H=4, W=4, n_K=2)

        out = m.compute(
            elev_indices  = np.array([0, 2]),
            range_indices = np.array([1]),
            az_indices    = np.array([0]),
        )

        assert "ssim_gt_elev_mean" in out
        assert "ssim_gt_range_mean" in out
        assert "ssim_gt_azimuth_mean" in out
        assert "ssim_gt_elev_0" in out

    def test_compute_includes_x_axis_metadata(self):
        m, _  = self._metrics(n_elev=5)
        empty = np.array([], dtype=np.int64)

        out = m.compute(elev_indices=empty, range_indices=empty, az_indices=empty)

        assert np.isclose(out["x_axis_min"], -10.0)
        assert np.isclose(out["x_axis_max"], 10.0)
        assert np.isclose(out["x_axis_step"], 5.0)

    def test_compute_none_indices_skips_ssim(self):
        m, _ = self._metrics(n_elev=4)

        metrics = m.compute()

        assert "curve_mse_gt" in metrics
        assert not any(key.startswith("ssim_") for key in metrics)


class TestCubeStitcherWindows:
    def test_make_patch_window_uniform(self):
        w = CubeStitcher.make_patch_window((4, 5), kind="uniform")

        assert w.shape == (4, 5)
        assert np.allclose(w, 1.0)

    def test_make_patch_window_hann_shape_and_positive(self):
        w = CubeStitcher.make_patch_window((8, 6), kind="hann")

        assert w.shape == (8, 6)
        assert np.all(w >= 1e-3)

    def test_make_patch_window_triangular_shape(self):
        w = CubeStitcher.make_patch_window((8, 8), kind="triangular")

        assert w.shape == (8, 8)
        assert np.all(w >= 1e-3)

    def test_make_patch_window_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            CubeStitcher.make_patch_window((4, 4), kind="does_not_exist")

    def test_hann_window_peaks_in_centre(self):
        w   = CubeStitcher.make_patch_window((16, 16), kind="hann")
        cy  = w[8, 8]
        edge = w[0, 0]

        assert cy > edge


class TestCubeStitcher:
    def _grid(self, spatial=(8, 8), patch=(4, 4), stride=4):
        return Patcher.build(spatial_size=spatial, patch_size=patch, stride=stride, use_reflective_padding=False).grid

    def test_constructor_allocates_accumulators(self):
        grid = self._grid()
        st   = CubeStitcher(grid, n_channels=3, window_kind="uniform")

        assert st.n_channels == 3
        assert st.number_of_patches == grid.number_of_patches

    def test_uniform_window_constant_field_recovers_value(self):
        grid = self._grid(spatial=(8, 8), patch=(4, 4), stride=4)
        st   = CubeStitcher(grid, n_channels=2, window_kind="uniform")

        for idx in range(grid.number_of_patches):
            patch = np.full((2, 4, 4), 3.0, dtype=np.float32)
            st.add_patch(idx, patch)

        cube = st.finalize_cube()

        assert cube.shape == (2, 8, 8)
        assert np.allclose(cube, 3.0)

    def test_add_patch_batch_matches_individual(self):
        grid = self._grid()
        rng  = np.random.default_rng(0)
        patches = rng.random((grid.number_of_patches, 2, 4, 4)).astype(np.float32)

        st_a = CubeStitcher(grid, n_channels=2, window_kind="hann")
        st_a.add_patch_batch(np.arange(grid.number_of_patches), patches)
        cube_a = st_a.finalize_cube()

        st_b = CubeStitcher(grid, n_channels=2, window_kind="hann")
        for idx in range(grid.number_of_patches):
            st_b.add_patch(idx, patches[idx])
        cube_b = st_b.finalize_cube()

        assert np.allclose(cube_a, cube_b)

    def test_overlapping_constant_field_recovers_value(self):
        grid = self._grid(spatial=(8, 8), patch=(4, 4), stride=2)
        st   = CubeStitcher(grid, n_channels=1, window_kind="hann")

        for idx in range(grid.number_of_patches):
            st.add_patch(idx, np.full((1, 4, 4), 5.0, dtype=np.float32))

        cube = st.finalize_cube()

        assert np.allclose(cube, 5.0, atol=1e-4)

    def test_finalize_dtype_is_configured_dtype(self):
        grid = self._grid()
        st   = CubeStitcher(grid, n_channels=1, window_kind="uniform", dtype="float32")
        st.add_patch(0, np.ones((1, 4, 4), dtype=np.float32))

        cube = st.finalize_cube()

        assert cube.dtype == np.float32

    def test_memmap_path_creates_file(self, tmp_path):
        grid = self._grid()
        mm   = tmp_path / "accum.npy"
        st   = CubeStitcher(grid, n_channels=2, window_kind="uniform", memmap_path=str(mm))

        assert mm.exists()
        st.add_patch(0, np.ones((2, 4, 4), dtype=np.float32))
        cube = st.finalize_cube()
        assert cube.shape == (2, 8, 8)


class TestPredictorCpuWorker:
    def _args(self, B=2, H=2, W=2, n_K=1, n_elev=7, seed=0):
        rng    = np.random.default_rng(seed)
        out_ch = n_K * 3
        pred   = rng.random((B, out_ch, H, W)).astype(np.float32)
        gt     = rng.random((B, out_ch, H, W)).astype(np.float32)

        for k in range(n_K):
            pred[:, 3 * k] = np.abs(pred[:, 3 * k]) + 0.5
            gt  [:, 3 * k] = np.abs(gt  [:, 3 * k]) + 0.5

        x_axis = np.linspace(-5.0, 5.0, n_elev).astype(np.float32)
        return (pred, gt, x_axis, n_K, out_ch, np.zeros(out_ch, dtype=np.float32), np.ones(out_ch, dtype=np.float32))

    def test_cpu_worker_output_structure(self):
        args = self._args(B=2, H=2, W=2, n_K=1, n_elev=7)
        pc, gc, pgf, ggf, mets, peak = Predictor._cpu_worker(args)

        assert pc.shape == (2, 7, 2, 2)
        assert gc.shape == (2, 7, 2, 2)
        assert pgf.shape == (2, 3, 2, 2)
        assert ggf.shape == (2, 3, 2, 2)
        assert set(mets.keys()) == {"mse", "mae", "r2", "cos"}
        assert mets["mse"].shape == (2, 2, 2)
        assert peak.shape == (2, 2, 2)

    def test_cpu_worker_metrics_finite(self):
        args = self._args(n_K=2, n_elev=9)
        _, _, _, _, mets, peak = Predictor._cpu_worker(args)

        for v in mets.values():
            assert np.all(np.isfinite(v))
        assert np.all(np.isfinite(peak))

    def test_cpu_worker_identical_curves_zero_error(self):
        n_elev = 21
        pred   = np.zeros((1, 3, 1, 1), dtype=np.float32)
        pred[0, 0, 0, 0] = 1.0
        pred[0, 1, 0, 0] = 0.0
        pred[0, 2, 0, 0] = 2.0
        gt     = pred.copy()

        x_axis = np.linspace(-10.0, 10.0, n_elev).astype(np.float32)
        args   = (pred, gt, x_axis, 1, 3, np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))

        pc, gc, _, _, mets, peak = Predictor._cpu_worker(args)

        assert np.allclose(mets["mse"], 0.0)
        assert np.allclose(mets["mae"], 0.0)
        assert np.allclose(peak, 0.0)
        assert np.allclose(mets["cos"], 1.0, atol=1e-4)

    def test_cpu_worker_gt_sorted_by_mu(self):
        n_elev = 7
        pred = np.zeros((1, 6, 1, 1), dtype=np.float32)
        gt   = np.zeros((1, 6, 1, 1), dtype=np.float32)

        gt[0, 0, 0, 0] = 1.0
        gt[0, 1, 0, 0] = 4.0
        gt[0, 2, 0, 0] = 1.0
        gt[0, 3, 0, 0] = 1.0
        gt[0, 4, 0, 0] = -4.0
        gt[0, 5, 0, 0] = 1.0
        pred[...] = gt

        x_axis = np.linspace(-5.0, 5.0, n_elev).astype(np.float32)
        args   = (pred, gt, x_axis, 2, 6, np.zeros(6, dtype=np.float32), np.ones(6, dtype=np.float32))

        _, _, _, ggf, _, _ = Predictor._cpu_worker(args)

        assert np.isclose(ggf[0, 1, 0, 0], -4.0)
        assert np.isclose(ggf[0, 4, 0, 0], 4.0)


class TestReportPayloadBuilder:
    def _run(self):
        input_cfg = SimpleNamespace(as_dict=lambda: {"channels": 4})
        ds_cfg    = SimpleNamespace(
            preprocessing_run_directory = Path("/data/prep"),
            input_config                = input_cfg,
            batch_size                  = 8,
        )
        split_region = SimpleNamespace(as_tuple=lambda: (0, 4, 0, 4))
        global_crop  = SimpleNamespace(as_tuple=lambda: (0, 10, 0, 10))
        grid         = SimpleNamespace(number_of_patches=4, patch_size=(4, 4), stride=4)

        return SimpleNamespace(
            model_name       = "unet",
            in_channels      = 6,
            out_channels     = 6,
            n_gaussians      = 2,
            x_axis_length    = 5,
            split_name       = "test",
            split_region     = split_region,
            global_crop      = global_crop,
            grid             = grid,
            dataset_config   = ds_cfg,
            used_ema         = True,
            secondary_labels = ["PS04", "PS06"],
        )

    def test_run_summary_keys(self):
        run   = self._run()
        x     = np.linspace(-10.0, 10.0, 5)
        out   = ReportPayloadBuilder.run_summary(run, x)

        assert out["model_name"] == "unet"
        assert out["n_gaussians"] == 2
        assert np.isclose(out["x_axis_min"], -10.0)
        assert np.isclose(out["x_axis_max"], 10.0)
        assert out["used_ema"] is True
        assert out["input_config"] == {"channels": 4}

    def test_inference_config_keys(self):
        run = self._run()
        cfg = SimpleNamespace(
            stitch_window      = "hann",
            cube_dtype         = "float32",
            save_cubes         = False,
            n_best_profiles    = 3,
            n_worst_profiles   = 3,
            n_random_profiles  = 5,
            n_range_slices     = 4,
            n_azimuth_slices   = 4,
            n_elevation_slices = 4,
            gif_axes           = ["range"],
            gif_fps            = 12,
            gif_max_frames     = 100,
            device             = "cpu",
            num_workers        = 2,
        )

        out = ReportPayloadBuilder.inference_config(cfg, run)

        assert out["stitch_window"] == "hann"
        assert out["batch_size"] == 8
        assert out["device"] == "cpu"
        assert out["gif_axes"] == ["range"]


class TestReportFormatting:
    def test_fmt_small_float_scientific(self):
        assert "e" in Report._fmt(1e-5)

    def test_fmt_large_float_scientific(self):
        assert "e" in Report._fmt(1e6)

    def test_fmt_mid_float_general(self):
        out = Report._fmt(3.14159)
        assert "e" not in out

    def test_fmt_list_joined(self):
        assert Report._fmt([1, 2, 3]) == "1, 2, 3"

    def test_fmt_tuple_joined(self):
        assert Report._fmt((1, 2)) == "1, 2"

    def test_fmt_string_passthrough(self):
        assert Report._fmt("hello") == "hello"

    def test_fmt_int_passthrough(self):
        assert Report._fmt(42) == "42"

    def test_kv_table_renders_rows(self):
        out = Report._kv_table([("a", 1), ("b", 2.5)])

        assert "`a`" in out
        assert "`b`" in out

    def test_dict_table_sorted(self):
        out   = Report._dict_table({"z": 1, "a": 2})
        a_pos = out.index("`a`")
        z_pos = out.index("`z`")

        assert a_pos < z_pos

    def test_three_col_table_renders(self):
        out = Report._three_col_table([("MSE", 0.5, "mean squared error")])

        assert "MSE" in out
        assert "mean squared error" in out

    def test_is_per_slice_ssim_true_for_indexed(self):
        assert Report._is_per_slice_ssim("ssim_gt_elev_3")
        assert Report._is_per_slice_ssim("elev_mae_gt_10")

    def test_is_per_slice_ssim_false_for_mean(self):
        assert not Report._is_per_slice_ssim("ssim_gt_elev_mean")
        assert not Report._is_per_slice_ssim("curve_mse_gt")


class TestReportAssemble:
    def _report(self, tmp_path, **kw):
        run = SimpleNamespace(
            model_name="unet", in_channels=6, out_channels=6, n_gaussians=2,
            x_axis_length=5, split_name="test",
            split_region=SimpleNamespace(as_tuple=lambda: (0, 4, 0, 4)),
            global_crop=SimpleNamespace(as_tuple=lambda: (0, 10, 0, 10)),
            grid=SimpleNamespace(number_of_patches=4, patch_size=(4, 4), stride=4),
            dataset_config=SimpleNamespace(
                preprocessing_run_directory=Path("/data/prep"),
                input_config=SimpleNamespace(as_dict=lambda: {"channels": 4}),
                batch_size=8,
            ),
            used_ema=False,
            secondary_labels=["PS04", "PS06"],
        )
        run_summary = ReportPayloadBuilder.run_summary(run, np.linspace(-10.0, 10.0, 5))
        inf_cfg     = {"device": "cpu", "batch_size": 8, "num_workers": 1, "stitch_window": "hann"}

        global_metrics = {
            "gt_mean": 0.5, "gt_std": 0.1, "gt_max": 1.0,
            "pred_mean": 0.5, "pred_std": 0.1, "pred_max": 1.0,
            "n_pixels": 9, "n_elevation": 5,
            "x_axis_min": -10.0, "x_axis_max": 10.0, "x_axis_step": 5.0,
            "curve_mse_gt": 0.01, "overall_r2_gt": 0.9,
            "pixel_mse_gt_mean": 0.02,
            "ssim_gt_elev_mean": 0.8, "ssim_gt_elev_0": 0.81,
        }
        global_metrics.update(kw.pop("extra_metrics", {}))

        return Report(
            output_dir       = tmp_path,
            run_summary      = run_summary,
            inference_config = inf_cfg,
            checkpoint_meta  = {"epoch": 5, "best_epoch": 4, "best_val_loss": 0.1},
            global_metrics   = global_metrics,
            figure_paths     = kw.pop("figure_paths", {}),
            gif_paths        = kw.pop("gif_paths", {}),
            report_path      = tmp_path / "report.md",
            extra_sections   = kw.pop("extra_sections", None),
        )

    def test_assemble_writes_report(self, tmp_path):
        rep  = self._report(tmp_path)
        path = rep.assemble()

        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "# TomoSAR Inference Report" in text
        assert "## 1. Run summary" in text
        assert "## 2. Headline metrics" in text
        assert "## 3. Full metric tables" in text

    def test_assemble_excludes_per_slice_ssim_from_full_tables(self, tmp_path):
        rep  = self._report(tmp_path)
        text = rep.assemble().read_text(encoding="utf-8")

        assert "ssim_gt_elev_0" not in text.split("## 3.")[1].split("## 4")[0]

    def test_assemble_with_figures(self, tmp_path):
        fig = tmp_path / "figures" / "f.png"
        fig.parent.mkdir(parents=True, exist_ok=True)
        fig.write_bytes(b"")

        rep  = self._report(tmp_path, figure_paths={"profiles_best": [fig]})
        text = rep.assemble().read_text(encoding="utf-8")

        assert "## 4. Profile reconstructions" in text
        assert "![" in text

    def test_assemble_with_gifs(self, tmp_path):
        gif = tmp_path / "anim" / "walk.gif"
        gif.parent.mkdir(parents=True, exist_ok=True)
        gif.write_bytes(b"")

        rep  = self._report(tmp_path, gif_paths={"walk_range": gif})
        text = rep.assemble().read_text(encoding="utf-8")

        assert "## 9. Animations" in text

    def test_assemble_with_extra_sections(self, tmp_path):
        rep  = self._report(tmp_path, extra_sections=["A custom note."])
        text = rep.assemble().read_text(encoding="utf-8")

        assert "## 10. Notes" in text
        assert "A custom note." in text

    def test_rel_path_relative_to_output(self, tmp_path):
        rep = self._report(tmp_path)
        rel = rep._rel(tmp_path / "figures" / "x.png")

        assert rel == "figures/x.png"

    def test_rel_path_outside_returns_absolute(self, tmp_path):
        rep   = self._report(tmp_path)
        other = Path("/some/where/else.png")

        assert rep._rel(other) == str(other)

    def test_img_returns_markdown_and_blank(self, tmp_path):
        rep = self._report(tmp_path)
        out = rep._img("k", tmp_path / "figures" / "x.png")

        assert out[0].startswith("![k](")
        assert out[1] == ""

    def test_imgs_single_path(self, tmp_path):
        rep = self._report(tmp_path)
        out = rep._imgs("k", tmp_path / "figures" / "x.png")

        assert any("![" in line for line in out)

    def test_imgs_list_of_paths(self, tmp_path):
        rep   = self._report(tmp_path)
        paths = [tmp_path / "figures" / "a.png", tmp_path / "figures" / "b.png"]
        out   = rep._imgs("k", paths)

        assert sum(1 for line in out if line.startswith("![")) == 2


class TestPlotTools:
    def test_gaussian_components_count(self):
        params = np.array([1.0, 0.0, 2.0, 0.5, 3.0, 1.0], dtype=np.float64)
        x      = np.linspace(-10.0, 10.0, 41)

        comps = PlotTools._gaussian_components(params, x, 2)

        assert len(comps) == 2
        assert comps[0].shape == (41,)


class TestPloter:
    def _plotter(self):
        return Ploter(fig_dpi=60, save_dpi=60)

    def test_int_label_default(self):
        assert self._plotter()._int_label == "intensity"

    def test_int_label_normalized(self):
        p = Ploter(fig_dpi=60, save_dpi=60, normalize=True)
        assert p._int_label == "intensity [0-1]"

    def test_maybe_normalize_passthrough(self):
        p   = self._plotter()
        arr = np.array([1.0, 2.0, 3.0])
        out = p._maybe_normalize(arr)

        assert np.allclose(out[0], arr)

    def test_maybe_normalize_scales(self):
        p   = Ploter(fig_dpi=60, save_dpi=60, normalize=True)
        arr = np.array([0.0, 5.0, 10.0])
        out = p._maybe_normalize(arr)

        assert np.isclose(out[0].min(), 0.0)
        assert np.isclose(out[0].max(), 1.0)

    def test_plot_profiles_writes_files(self, tmp_path):
        p           = self._plotter()
        n_elev      = 11
        H = W       = 3
        x_axis      = np.linspace(-10.0, 10.0, n_elev)
        pred_curves = np.random.default_rng(0).random((n_elev, H, W)).astype(np.float32)
        gt_curves   = np.random.default_rng(1).random((n_elev, H, W)).astype(np.float32)
        params_pred = np.random.default_rng(2).random((6, H, W)).astype(np.float32)
        pixels      = np.array([[0, 0], [1, 1]], dtype=np.int32)
        pixel_metrics = {
            "mse": np.zeros((H, W)), "r2": np.ones((H, W)), "cos": np.ones((H, W)),
        }

        paths = p.plot_profiles(
            pred_curves=pred_curves, gt_curves=gt_curves, params_pred=params_pred,
            x_axis=x_axis, pixels=pixels, tag="best", out_dir=tmp_path / "prof",
            n_gaussians=2, pixel_metrics=pixel_metrics, az_offset=0, rg_offset=0,
        )

        assert len(paths) == 2
        for pth in paths:
            assert pth.exists()

    def test_plot_profiles_empty_pixels(self, tmp_path):
        p = self._plotter()
        x_axis = np.linspace(-10.0, 10.0, 11)
        empty  = np.empty((0, 2), dtype=np.int32)

        paths = p.plot_profiles(
            pred_curves=np.zeros((11, 3, 3), dtype=np.float32),
            gt_curves=np.zeros((11, 3, 3), dtype=np.float32),
            params_pred=np.zeros((6, 3, 3), dtype=np.float32),
            x_axis=x_axis, pixels=empty, tag="random", out_dir=tmp_path / "prof",
            n_gaussians=2,
            pixel_metrics={"mse": np.zeros((3, 3)), "r2": np.zeros((3, 3)), "cos": np.zeros((3, 3))},
            az_offset=0, rg_offset=0,
        )

        assert paths == []

    def test_plot_pixel_metric_map_writes(self, tmp_path):
        p          = self._plotter()
        metric_map = np.random.default_rng(0).random((6, 8)).astype(np.float32)

        out = p.plot_pixel_metric_map(
            metric_map=metric_map, title="MSE", label="MSE",
            out_path=tmp_path / "m.png", az_offset=0, rg_offset=0,
        )

        assert out.exists()

    def test_plot_pixel_metric_map_log_scale(self, tmp_path):
        p          = self._plotter()
        metric_map = np.abs(np.random.default_rng(0).random((6, 8)).astype(np.float32)) + 1e-3

        out = p.plot_pixel_metric_map(
            metric_map=metric_map, title="MSE", label="MSE",
            out_path=tmp_path / "mlog.png", az_offset=0, rg_offset=0, log=True,
        )

        assert out.exists()

    def test_plot_tomogram_slice_range(self, tmp_path):
        p       = self._plotter()
        n_elev  = 9
        pred    = np.random.default_rng(0).random((n_elev, 5, 7)).astype(np.float32)
        gt      = np.random.default_rng(1).random((n_elev, 5, 7)).astype(np.float32)
        x_axis  = np.linspace(-10.0, 10.0, n_elev)

        paths = p.plot_tomogram_slice(
            pred_cube=pred, gt_cube=gt, axis="range", index=2, x_axis=x_axis,
            out_dir=tmp_path / "slices", stem="range_2", az_offset=0, rg_offset=0,
            ssim_value=0.8,
        )

        assert len(paths) == 3
        for pth in paths:
            assert pth.exists()

    def test_plot_tomogram_slice_azimuth(self, tmp_path):
        p      = self._plotter()
        n_elev = 9
        pred   = np.random.default_rng(0).random((n_elev, 5, 7)).astype(np.float32)
        gt     = np.random.default_rng(1).random((n_elev, 5, 7)).astype(np.float32)
        x_axis = np.linspace(-10.0, 10.0, n_elev)

        paths = p.plot_tomogram_slice(
            pred_cube=pred, gt_cube=gt, axis="azimuth", index=1, x_axis=x_axis,
            out_dir=tmp_path / "slices", stem="az_1", az_offset=0, rg_offset=0,
        )

        assert len(paths) == 3

    def test_plot_tomogram_slice_invalid_axis(self, tmp_path):
        p      = self._plotter()
        n_elev = 9
        pred   = np.zeros((n_elev, 5, 7), dtype=np.float32)
        x_axis = np.linspace(-10.0, 10.0, n_elev)

        with pytest.raises(ValueError):
            p.plot_tomogram_slice(
                pred_cube=pred, gt_cube=pred, axis="elevation", index=0, x_axis=x_axis,
                out_dir=tmp_path / "slices", stem="bad", az_offset=0, rg_offset=0,
            )

    def test_plot_elevation_intensity_slice(self, tmp_path):
        p      = self._plotter()
        n_elev = 9
        pred   = np.random.default_rng(0).random((n_elev, 5, 7)).astype(np.float32)
        gt     = np.random.default_rng(1).random((n_elev, 5, 7)).astype(np.float32)
        x_axis = np.linspace(-10.0, 10.0, n_elev)

        paths = p.plot_elevation_intensity_slice(
            pred_cube=pred, gt_cube=gt, elev_idx=3, x_axis=x_axis,
            out_dir=tmp_path / "slices", stem="elev_3", az_offset=0, rg_offset=0,
            ssim_value=0.9,
        )

        assert len(paths) == 3
        for pth in paths:
            assert pth.exists()

    def test_plot_metric_histograms(self, tmp_path):
        p   = self._plotter()
        arr = {
            "pixel_mse": np.random.default_rng(0).random((6, 6)).astype(np.float32),
            "pixel_r2":  np.random.default_rng(1).random((6, 6)).astype(np.float32),
        }

        paths = p.plot_metric_histograms(arr, tmp_path / "hist")

        assert len(paths) == 2

    def test_plot_metric_histograms_all_nan_skipped(self, tmp_path):
        p   = self._plotter()
        arr = {"bad": np.full((4, 4), np.nan, dtype=np.float32)}

        paths = p.plot_metric_histograms(arr, tmp_path / "hist")

        assert paths == []

    def test_plot_ssim_curves(self, tmp_path):
        p  = self._plotter()
        gm = {f"ssim_gt_range_{i}": 0.5 + 0.01 * i for i in range(5)}

        out = p.plot_ssim_curves(
            global_metrics=gm, axis="range", out_path=tmp_path / "ssim.png",
            n_slices=5, slice_indices=np.arange(5),
        )

        assert out.exists()

    def test_plot_elev_metric_curves(self, tmp_path):
        p      = self._plotter()
        n_elev = 6
        x_axis = np.linspace(-10.0, 10.0, n_elev)
        gm     = {}
        for key in ("elev_mae", "elev_rmse", "elev_r2", "elev_ce"):
            for i in range(n_elev):
                gm[f"{key}_gt_{i}"] = 0.1 * i

        paths = p.plot_elev_metric_curves(global_metrics=gm, out_dir=tmp_path / "em", n_elev=n_elev, x_axis=x_axis)

        assert len(paths) == 4

    def test_plot_param_maps_with_gt(self, tmp_path):
        p   = self._plotter()
        pp  = np.random.default_rng(0).random((6, 4, 5)).astype(np.float32)
        pg  = np.random.default_rng(1).random((6, 4, 5)).astype(np.float32)

        paths = p.plot_param_maps(params_pred=pp, params_gt=pg, n_gaussians=2,
                                  out_dir=tmp_path / "pm", az_offset=0, rg_offset=0)

        assert len(paths) == 12

    def test_plot_param_maps_without_gt(self, tmp_path):
        p  = self._plotter()
        pp = np.random.default_rng(0).random((6, 4, 5)).astype(np.float32)

        paths = p.plot_param_maps(params_pred=pp, params_gt=None, n_gaussians=2,
                                  out_dir=tmp_path / "pm", az_offset=0, rg_offset=0)

        assert len(paths) == 6

    def test_plot_param_distributions(self, tmp_path):
        p   = self._plotter()
        rng = np.random.default_rng(0)
        pp  = np.abs(rng.random((6, 5, 5))).astype(np.float32) + 0.5
        pg  = np.abs(rng.random((6, 5, 5))).astype(np.float32) + 0.5

        paths = p.plot_param_distributions(params_pred=pp, params_gt=pg, n_gaussians=2,
                                           out_dir=tmp_path / "pd")

        assert len(paths) > 0

    def test_plot_param_scatter(self, tmp_path):
        p   = self._plotter()
        rng = np.random.default_rng(0)
        pp  = np.abs(rng.random((6, 6, 6))).astype(np.float32) + 0.5
        pg  = np.abs(rng.random((6, 6, 6))).astype(np.float32) + 0.5

        paths = p.plot_param_scatter(params_pred=pp, params_gt=pg, n_gaussians=2,
                                     out_dir=tmp_path / "ps")

        assert len(paths) > 0

    def test_plot_param_error_maps(self, tmp_path):
        p   = self._plotter()
        rng = np.random.default_rng(0)
        pp  = rng.random((6, 4, 4)).astype(np.float32)
        pg  = rng.random((6, 4, 4)).astype(np.float32)

        paths = p.plot_param_error_maps(params_pred=pp, params_gt=pg, n_gaussians=2,
                                        out_dir=tmp_path / "pe", az_offset=0, rg_offset=0)

        assert len(paths) == 6

    def test_plot_slot_mu_distributions(self, tmp_path):
        p  = self._plotter()
        gm = {}
        for k in range(2):
            gm[f"slot_{k}_mu_pred_mean"] = 0.5 * k
            gm[f"slot_{k}_mu_pred_std"]  = 0.1
            gm[f"slot_{k}_mu_gt_mean"]   = 0.5 * k
            gm[f"slot_{k}_mu_gt_std"]    = 0.1

        paths = p.plot_slot_mu_distributions(global_metrics=gm, n_gaussians=2, out_dir=tmp_path / "slot")

        assert len(paths) == 2

    def test_plot_placeholder_detection(self, tmp_path):
        p  = self._plotter()
        gm = {"placeholder_precision": 0.9, "placeholder_recall": 0.8, "placeholder_f1": 0.85}
        for k in range(2):
            gm[f"slot_{k}_placeholder_precision"] = 0.9
            gm[f"slot_{k}_placeholder_recall"]    = 0.8
            gm[f"slot_{k}_placeholder_f1"]        = 0.85
            gm[f"slot_{k}_placeholder_gt_rate"]   = 0.3

        paths = p.plot_placeholder_detection(global_metrics=gm, n_gaussians=2, out_dir=tmp_path / "ph")

        assert len(paths) == 2

    def test_plot_slot_ordering_summary(self, tmp_path):
        p  = self._plotter()
        gm = {
            "mu_ordering_rate": 0.95,
            "permutation_consensus_dominant_frac": 0.9,
            "permutation_consensus_identity_frac": 0.85,
        }
        for k in range(2):
            gm[f"slot_{k}_placeholder_gt_rate"] = 0.3
            gm[f"slot_{k}_mu_pred_mean"]        = 0.5 * k
            gm[f"slot_{k}_mu_gt_mean"]          = 0.5 * k
            gm[f"slot_{k}_mu_pred_std"]         = 0.1
            gm[f"slot_{k}_mu_gt_std"]           = 0.1

        paths = p.plot_slot_ordering_summary(global_metrics=gm, n_gaussians=2, out_dir=tmp_path / "so")

        assert len(paths) == 3

    def test_plot_active_count_map(self, tmp_path):
        p   = self._plotter()
        rng = np.random.default_rng(0)
        pp  = np.abs(rng.random((6, 5, 5))).astype(np.float32)
        pg  = np.abs(rng.random((6, 5, 5))).astype(np.float32)

        paths = p.plot_active_count_map(params_pred=pp, params_gt=pg, n_gaussians=2,
                                        out_dir=tmp_path / "ac", az_offset=0, rg_offset=0)

        assert len(paths) == 2
        for pth in paths:
            assert pth.exists()


class TestTrackAndChannelPlots:
    def _plotter(self):
        return Ploter(fig_dpi=60, save_dpi=60)

    def _baselines(self):
        return TrackBaselines(
            labels         = ["PS02", "PS04", "PS26"],
            vertical       = [0.0, -0.76, -2.49],
            horizontal     = [0.0, 7.20, 151.27],
            vertical_std   = [0.48, 0.68, 0.34],
            horizontal_std = [0.39, 0.73, 0.40],
            azimuth_window = (1000, 16000),
        )

    def _profiles(self):
        rng = np.random.default_rng(0)
        return TrackProfiles(
            labels        = ["PS02", "PS04", "PS26"],
            horizontal    = rng.random((3, 50)),
            vertical      = rng.random((3, 50)),
            azimuth_start = 1000,
        )

    def test_plot_track_geometry(self, tmp_path):
        out = self._plotter().plot_track_geometry(self._baselines(), tmp_path / "tracks" / "geometry.png")

        assert out.exists()

    def test_plot_track_profiles(self, tmp_path):
        paths = self._plotter().plot_track_profiles(self._profiles(), tmp_path / "tracks", split_azimuth=(1010, 1030))

        assert len(paths) == 2
        for path in paths:
            assert path.exists()

    def test_plot_track_flight_3d(self, tmp_path):
        out = self._plotter().plot_track_flight_3d(self._profiles(), tmp_path / "tracks" / "flight_tracks_3d.png")

        assert out.exists()

    def test_plot_input_channels(self, tmp_path):
        rng    = np.random.default_rng(1)
        inputs = (rng.normal(size=(5, 8, 9)) + 1j * rng.normal(size=(5, 8, 9))).astype(np.complex64)

        paths = self._plotter().plot_input_channels(
            complex_inputs = inputs,
            n_secondaries  = 2,
            labels         = ["PS04", "PS06"],
            out_dir        = tmp_path / "channels",
            az_offset      = 0,
            rg_offset      = 0,
            primary_label  = "PS02",
        )

        assert len(paths) == 5
        names = {path.name for path in paths}
        assert "pass_primary_amplitude.png" in names
        assert "pass_PS04_amplitude.png" in names
        assert "interferogram_PS06_phase.png" in names

    def test_plot_input_channels_without_labels(self, tmp_path):
        inputs = np.ones((3, 6, 6), dtype=np.complex64)

        paths = self._plotter().plot_input_channels(
            complex_inputs = inputs,
            n_secondaries  = 1,
            labels         = None,
            out_dir        = tmp_path / "channels",
            az_offset      = 0,
            rg_offset      = 0,
        )

        assert len(paths) == 3


class TestReportTracksTable:
    def _report(self, tmp_path, metrics):
        return Report(
            output_dir       = tmp_path,
            run_summary      = {},
            inference_config = {},
            checkpoint_meta  = {},
            global_metrics   = metrics,
            figure_paths     = {},
            gif_paths        = {},
            report_path      = tmp_path / "report.md",
        )

    def test_tracks_table_rendered(self, tmp_path):
        tracks  = TrackBaselines(
            labels         = ["PS02", "PS04"],
            vertical       = [0.0, -0.76],
            horizontal     = [0.0, 7.20],
            vertical_std   = [0.48, 0.68],
            horizontal_std = [0.39, 0.73],
        ).to_payload()
        report  = self._report(tmp_path, {"curve_mse_gt": 0.1, "tracks": tracks})
        content = report.assemble().read_text(encoding="utf-8")

        assert "Tracks used in this run" in content
        assert "PS04" in content

    def test_tracks_excluded_from_full_metric_tables(self, tmp_path):
        tracks  = TrackBaselines(labels=["PS02"], vertical=[0.0], horizontal=[0.0], vertical_std=[0.1], horizontal_std=[0.1]).to_payload()
        report  = self._report(tmp_path, {"curve_mse_gt": 0.1, "tracks": tracks})
        content = report.assemble().read_text(encoding="utf-8")

        assert "`tracks`" not in content

    def test_no_tracks_no_section(self, tmp_path):
        report  = self._report(tmp_path, {"curve_mse_gt": 0.1})
        content = report.assemble().read_text(encoding="utf-8")

        assert "Tracks used in this run" not in content

    def _positions(self):
        return TrackProfiles(
            labels        = ["PS02", "PS04"],
            horizontal    = np.array([[0.0, 0.0], [3.0, 5.0]]),
            vertical      = np.array([[1.0, 1.0], [2.0, 4.0]]),
            azimuth_start = 1000,
        ).position_summary()

    def test_track_positions_table_rendered(self, tmp_path):
        report  = self._report(tmp_path, {"curve_mse_gt": 0.1, "track_positions": self._positions()})
        content = report.assemble().read_text(encoding="utf-8")

        assert "Track positions and temporal deviation" in content
        assert "Planar dev RMS [m]" in content
        assert "PS04" in content

    def test_track_positions_excluded_from_full_metric_tables(self, tmp_path):
        report  = self._report(tmp_path, {"curve_mse_gt": 0.1, "track_positions": self._positions()})
        content = report.assemble().read_text(encoding="utf-8")

        assert "`track_positions`" not in content

    def test_no_positions_no_section(self, tmp_path):
        report  = self._report(tmp_path, {"curve_mse_gt": 0.1})
        content = report.assemble().read_text(encoding="utf-8")

        assert "Track positions and temporal deviation" not in content


def _make_logger(tmp_path) -> "object":
    from tools.logger import Logger
    return Logger(log_dir=str(tmp_path), name="inference_test")


class TestAnimatorSlicing:
    def test_slice_elevation(self):
        pred = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        gt   = pred + 1.0

        p, g, r = Animator._slice_elevation((pred, gt, None), 1)

        assert np.array_equal(p, pred[1])
        assert np.array_equal(g, gt[1])
        assert r is None

    def test_slice_elevation_with_reduced(self):
        pred = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        gt   = pred + 1.0
        red  = pred + 2.0

        p, g, r = Animator._slice_elevation((pred, gt, red), 1)

        assert np.array_equal(r, red[1])

    def test_slice_range_no_sort(self):
        pred = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        gt   = pred + 1.0

        p, g, r = Animator._slice_range((pred, gt, None), None, 2)

        assert np.array_equal(p, pred[:, :, 2])
        assert r is None

    def test_slice_range_with_sort(self):
        pred     = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        gt       = pred + 1.0
        sort_idx = np.array([1, 0])

        p, g, r = Animator._slice_range((pred, gt, None), sort_idx, 0)

        assert np.array_equal(p, pred[:, :, 0][sort_idx])

    def test_slice_azimuth_no_sort(self):
        pred = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        gt   = pred + 1.0

        p, g, r = Animator._slice_azimuth((pred, gt, None), None, 1)

        assert np.array_equal(p, pred[:, 1, :])


class TestAnimatorBuildAxis:
    def _animator(self, tmp_path):
        return Animator(logger=_make_logger(tmp_path), dpi=50, fps=4, max_frames=4, num_workers=1)

    def test_build_axis_elevation(self, tmp_path):
        anim   = self._animator(tmp_path)
        pred   = np.random.default_rng(0).random((4, 3, 5)).astype(np.float32)
        gt     = pred.copy()
        x_axis = np.linspace(-10.0, 10.0, 4)

        spec = anim._build_axis("elevation", (pred, gt, None), x_axis, az_offset=0, rg_offset=0)

        assert spec["n_total"] == 4
        assert spec["origin"] == "upper"
        p, g, r = spec["get_slice"](1)
        assert p.shape == (3, 5)
        assert r is None

    def test_build_axis_range(self, tmp_path):
        anim   = self._animator(tmp_path)
        pred   = np.random.default_rng(0).random((4, 3, 5)).astype(np.float32)
        gt     = pred.copy()
        x_axis = np.linspace(-10.0, 10.0, 4)

        spec = anim._build_axis("range", (pred, gt, None), x_axis, az_offset=0, rg_offset=0)

        assert spec["n_total"] == 5
        assert spec["origin"] == "lower"

    def test_build_axis_azimuth(self, tmp_path):
        anim   = self._animator(tmp_path)
        pred   = np.random.default_rng(0).random((4, 3, 5)).astype(np.float32)
        gt     = pred.copy()
        x_axis = np.linspace(-10.0, 10.0, 4)

        spec = anim._build_axis("azimuth", (pred, gt, None), x_axis, az_offset=0, rg_offset=0)

        assert spec["n_total"] == 3

    def test_build_axis_invalid_raises(self, tmp_path):
        anim   = self._animator(tmp_path)
        pred   = np.zeros((4, 3, 5), dtype=np.float32)
        x_axis = np.linspace(-10.0, 10.0, 4)

        with pytest.raises(ValueError):
            anim._build_axis("diagonal", (pred, pred, None), x_axis, az_offset=0, rg_offset=0)


class TestAnimatorWalkGif:
    def test_walk_gif_elevation(self, tmp_path):
        anim   = Animator(logger=_make_logger(tmp_path), dpi=40, fps=4, max_frames=3, num_workers=1)
        pred   = np.random.default_rng(0).random((3, 4, 5)).astype(np.float32)
        gt     = np.random.default_rng(1).random((3, 4, 5)).astype(np.float32)
        x_axis = np.linspace(-10.0, 10.0, 3)
        out    = tmp_path / "walk.gif"

        result = anim.walk_gif(pred, gt, "elevation", out, x_axis=x_axis, az_offset=0, rg_offset=0)

        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0
