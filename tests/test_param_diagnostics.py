from __future__ import annotations

import numpy as np

from pipelines.param_pipeline.metrics import FittingMetricsCalculator, KSelectionDiagnostics, SnrEstimator
from tools.logger import NullLogger


def _make_diagnostics() -> dict:
    mse = np.full((3, 1, 2), np.nan, dtype=np.float32)
    pen = np.full((3, 1, 2), np.nan, dtype=np.float32)

    mse[:, 0, 0] = [0.50, 0.15, 0.20]
    pen[:, 0, 0] = [0.50, 0.20, 0.30]

    best          = np.zeros((1, 2), dtype=np.int16)
    best[0, 0]    = 2

    return {"mse_per_k": mse, "penalised_per_k": pen, "best_k_map": best}


class TestKSelectionMargins:
    def test_margin_to_adjacent_orders(self):
        maps, _ = KSelectionDiagnostics(k_max=3, logger=NullLogger()).run(_make_diagnostics())

        assert np.isclose(maps["k_margin_prev_map"][0, 0], 0.30)
        assert np.isclose(maps["k_margin_next_map"][0, 0], 0.10)

    def test_margin_to_runner_up_and_relative(self):
        maps, _ = KSelectionDiagnostics(k_max=3, logger=NullLogger()).run(_make_diagnostics())

        assert np.isclose(maps["k_margin_second_map"]  [0, 0], 0.10)
        assert np.isclose(maps["k_relative_margin_map"][0, 0], 0.50)

    def test_inactive_pixel_is_nan(self):
        maps, _ = KSelectionDiagnostics(k_max=3, logger=NullLogger()).run(_make_diagnostics())

        assert np.isnan(maps["k_margin_prev_map"]    [0, 1])
        assert np.isnan(maps["k_margin_next_map"]    [0, 1])
        assert np.isnan(maps["k_margin_second_map"]  [0, 1])
        assert np.isnan(maps["k_relative_margin_map"][0, 1])

    def test_boundary_orders_have_nan_outer_margins(self):
        diag = _make_diagnostics()
        diag["best_k_map"][0, 0] = 1

        maps, _ = KSelectionDiagnostics(k_max=3, logger=NullLogger()).run(diag)

        assert np.isnan (maps["k_margin_prev_map"][0, 0])
        assert np.isfinite(maps["k_margin_next_map"][0, 0])

    def test_single_order_has_no_margins(self):
        mse  = np.full((1, 1, 1), 0.1, dtype=np.float32)
        pen  = np.full((1, 1, 1), 0.1, dtype=np.float32)
        best = np.ones((1, 1), dtype=np.int16)

        maps, summary = KSelectionDiagnostics(k_max=1, logger=NullLogger()).run({"mse_per_k": mse, "penalised_per_k": pen, "best_k_map": best})

        assert np.isnan(maps["k_margin_second_map"]  [0, 0])
        assert np.isnan(maps["k_relative_margin_map"][0, 0])
        assert np.isnan(summary["k_relative_margin_median"])


class TestKSelectionSummary:
    def test_win_fractions_sum_to_one_over_active(self):
        _, summary = KSelectionDiagnostics(k_max=3, logger=NullLogger()).run(_make_diagnostics())

        total = sum(summary[f"k{k}_win_fraction"] for k in range(1, 4))

        assert summary["n_active_pixels"] == 1.0
        assert np.isclose(total, 1.0)
        assert np.isclose(summary["k2_win_fraction"], 1.0)

    def test_per_k_statistics_use_active_pixels_only(self):
        _, summary = KSelectionDiagnostics(k_max=3, logger=NullLogger()).run(_make_diagnostics())

        assert np.isclose(summary["k1_mse_mean"],     0.50)
        assert np.isclose(summary["k2_mse_mean"],     0.15)
        assert np.isclose(summary["k2_penalty_mean"], 0.05)

    def test_ambiguous_fraction_reflects_threshold(self):
        _, summary = KSelectionDiagnostics(k_max=3, logger=NullLogger(), ambiguity_threshold=0.6).run(_make_diagnostics())

        assert np.isclose(summary["k_ambiguous_fraction"], 1.0)


class TestSnrEstimator:
    def test_known_peak_to_floor_ratio(self):
        H        = 100
        tomogram = np.ones((H, 1, 1), dtype=np.float32)
        tomogram[50, 0, 0] = 100.0

        snr_db = SnrEstimator(logger=NullLogger()).run(tomogram)

        assert snr_db.shape == (1, 1)
        assert np.isclose(snr_db[0, 0], 20.0, atol=1e-3)

    def test_zero_profile_is_nan(self):
        tomogram = np.zeros((50, 2, 2), dtype=np.float32)

        snr_db = SnrEstimator(logger=NullLogger()).run(tomogram)

        assert np.all(np.isnan(snr_db))

    def test_chunking_matches_single_pass(self):
        rng      = np.random.default_rng(0)
        tomogram = rng.uniform(0.1, 5.0, size=(40, 4, 10)).astype(np.float32)

        snr_single  = SnrEstimator(logger=NullLogger(), range_chunk=1024).run(tomogram)
        snr_chunked = SnrEstimator(logger=NullLogger(), range_chunk=3   ).run(tomogram)

        assert np.allclose(snr_single, snr_chunked, equal_nan=True)


class TestSnrRelationSummary:
    def test_monotonic_relation_gives_high_spearman(self):
        calc = FittingMetricsCalculator(n_gaussians=2, logger=NullLogger())

        snr = np.linspace(0.0, 30.0, 400, dtype=np.float32).reshape(20, 20)
        r2  = (snr / 40.0).astype(np.float32)

        summary = calc._compute_snr_summary(snr, r2, None)

        assert summary["snr_r2_spearman"] > 0.99
        assert summary["snr_r2_pearson"]  > 0.99

    def test_per_k_medians_present_when_best_k_given(self):
        calc = FittingMetricsCalculator(n_gaussians=2, logger=NullLogger())

        snr    = np.array([[10.0, 20.0], [10.0, 20.0]], dtype=np.float32)
        r2     = np.array([[0.5,  0.9 ], [0.4,  0.8 ]], dtype=np.float32)
        best_k = np.array([[1,    2   ], [1,    2   ]], dtype=np.int16)

        summary = calc._compute_snr_summary(snr, r2, best_k)

        assert np.isclose(summary["snr_db_median_k1"], 10.0)
        assert np.isclose(summary["snr_db_median_k2"], 20.0)
