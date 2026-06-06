from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import pytest
import torch

from tools.permutation_metrics import PermutationMetrics


@dataclass
class MetricsCfg:
    enabled       : bool  = True
    amp_threshold : float = 1e-3


def _seed() -> None:
    torch.manual_seed(0)


def _make_params(B: int, G: int, ppg: int, H: int, W: int, fill: float = 0.0) -> torch.Tensor:
    return torch.full((B, G * ppg, H, W), float(fill), dtype=torch.float32)


class TestConstruction:
    def test_init_reads_enabled_flag(self):
        cfg = MetricsCfg(enabled=True)
        pm  = PermutationMetrics(cfg)

        assert pm.enabled is True
        assert pm.cfg is cfg

    def test_init_disabled_flag(self):
        pm = PermutationMetrics(MetricsCfg(enabled=False))

        assert pm.enabled is False

    def test_silent_constructs_without_logger(self):
        pm = PermutationMetrics.silent(MetricsCfg())

        assert isinstance(pm, PermutationMetrics)
        assert pm.enabled is True

    def test_init_with_logger_calls_kv_table(self):
        calls = []

        class FakeLogger:
            def kv_table(self, data):
                calls.append(data)

        PermutationMetrics(MetricsCfg(enabled=True), logger=FakeLogger())

        assert len(calls) == 1
        assert calls[0] == {"Permutation metrics": "enabled"}

    def test_init_with_logger_disabled_label(self):
        captured = {}

        class FakeLogger:
            def kv_table(self, data):
                captured.update(data)

        PermutationMetrics(MetricsCfg(enabled=False), logger=FakeLogger())

        assert captured == {"Permutation metrics": "disabled"}


class TestMuOrderingRate:
    def test_ordered_active_slots_give_rate_one(self):
        _seed()
        ppg    = 3
        params = _make_params(1, 3, ppg, 2, 2)
        params[:, 0] = 1.0
        params[:, 3] = 1.0
        params[:, 6] = 1.0
        params[:, 1] = -5.0
        params[:, 4] = 0.0
        params[:, 7] = 5.0

        rate = PermutationMetrics._mu_ordering_rate(params, ppg)

        assert rate == 1.0

    def test_disordered_active_slots_give_rate_zero(self):
        ppg    = 3
        params = _make_params(1, 3, ppg, 2, 2)
        params[:, 0] = 1.0
        params[:, 3] = 1.0
        params[:, 6] = 1.0
        params[:, 1] = 5.0
        params[:, 4] = 0.0
        params[:, 7] = -5.0

        rate = PermutationMetrics._mu_ordering_rate(params, ppg)

        assert rate == 0.0

    def test_no_multi_active_pixel_returns_nan(self):
        ppg    = 3
        params = _make_params(1, 3, ppg, 2, 2, fill=0.0)
        params[:, 0] = 1.0

        rate = PermutationMetrics._mu_ordering_rate(params, ppg)

        assert math.isnan(rate)

    def test_amp_threshold_controls_activity(self):
        ppg    = 3
        params = _make_params(1, 2, ppg, 1, 1, fill=0.0)
        params[:, 0] = 0.5
        params[:, 3] = 0.5
        params[:, 1] = 10.0
        params[:, 4] = -10.0

        rate_low  = PermutationMetrics._mu_ordering_rate(params, ppg, amp_threshold=0.1)
        rate_high = PermutationMetrics._mu_ordering_rate(params, ppg, amp_threshold=1.0)

        assert rate_low == 0.0
        assert math.isnan(rate_high)

    def test_equal_mu_counts_as_violation(self):
        ppg    = 3
        params = _make_params(1, 2, ppg, 1, 1, fill=0.0)
        params[:, 0] = 1.0
        params[:, 3] = 1.0
        params[:, 1] = 2.0
        params[:, 4] = 2.0

        rate = PermutationMetrics._mu_ordering_rate(params, ppg)

        assert rate == 0.0

    def test_return_type_is_python_float(self):
        ppg    = 3
        params = _make_params(1, 2, ppg, 1, 1)
        params[:, 0] = 1.0
        params[:, 3] = 1.0
        params[:, 1] = -1.0
        params[:, 4] = 1.0

        rate = PermutationMetrics._mu_ordering_rate(params, ppg)

        assert isinstance(rate, float)


class TestAssignmentCostMargin:
    def test_keys_present(self):
        _seed()
        ppg = 3
        p   = torch.randn(1, 2 * ppg, 2, 2)
        g   = torch.randn(1, 2 * ppg, 2, 2)

        out = PermutationMetrics._assignment_cost_margin(p, g, ppg)

        assert set(out.keys()) == {"mean_margin", "mean_rel_margin", "ambiguous_frac"}
        assert all(isinstance(v, float) for v in out.values())

    def test_identical_pred_gt_margin_equals_swap_cost(self):
        ppg = 3
        p   = _make_params(1, 2, ppg, 1, 1)
        p[:, 1] = 1.0
        p[:, 4] = 2.0
        g       = p.clone()

        out = PermutationMetrics._assignment_cost_margin(p, g, ppg)

        assert abs(out["mean_margin"] - 2.0) < 1e-5
        assert out["mean_margin"] > 0.0

    def test_well_separated_mu_positive_margin(self):
        ppg = 3
        p   = _make_params(1, 2, ppg, 1, 1)
        p[:, 1] = 0.0
        p[:, 4] = 100.0
        g       = p.clone()

        out = PermutationMetrics._assignment_cost_margin(p, g, ppg)

        assert out["mean_margin"] > 0.0

    def test_ambiguous_frac_in_unit_interval(self):
        _seed()
        ppg = 3
        p   = torch.randn(2, 3 * ppg, 3, 3)
        g   = torch.randn(2, 3 * ppg, 3, 3)

        out = PermutationMetrics._assignment_cost_margin(p, g, ppg)

        assert 0.0 <= out["ambiguous_frac"] <= 1.0


class TestSlotActivationStats:
    def test_fully_active_rates_are_one(self):
        ppg    = 3
        params = _make_params(1, 2, ppg, 2, 2, fill=0.0)
        params[:, 0] = 1.0
        params[:, 3] = 1.0

        out = PermutationMetrics._slot_activation_stats(params, ppg)

        assert out["active_rate/slot_0"] == 1.0
        assert out["active_rate/slot_1"] == 1.0
        assert out["activation_rate_std"] == 0.0

    def test_inactive_slot_rate_zero(self):
        ppg    = 3
        params = _make_params(1, 2, ppg, 2, 2, fill=0.0)
        params[:, 0] = 1.0

        out = PermutationMetrics._slot_activation_stats(params, ppg)

        assert out["active_rate/slot_0"] == 1.0
        assert out["active_rate/slot_1"] == 0.0
        assert out["mean_amp/slot_1"] == 0.0

    def test_keys_per_slot(self):
        ppg    = 3
        params = _make_params(1, 3, ppg, 1, 1)

        out = PermutationMetrics._slot_activation_stats(params, ppg)

        for i in range(3):
            assert f"active_rate/slot_{i}" in out
            assert f"mean_amp/slot_{i}" in out
        assert "activation_rate_std" in out

    def test_mean_amp_reflects_value(self):
        ppg    = 3
        params = _make_params(1, 1, ppg, 2, 2, fill=0.0)
        params[:, 0] = 4.0

        out = PermutationMetrics._slot_activation_stats(params, ppg)

        assert abs(out["mean_amp/slot_0"] - 4.0) < 1e-6


class TestPlaceholderDetectionStats:
    def test_perfect_match_precision_recall_one(self):
        ppg = 3
        p   = _make_params(1, 2, ppg, 2, 2, fill=0.0)
        g   = _make_params(1, 2, ppg, 2, 2, fill=0.0)
        p[:, 0] = 1.0
        g[:, 0] = 1.0

        out = PermutationMetrics._placeholder_detection_stats(p, g, ppg)

        assert out["placeholder/precision"] > 1.0 - 1e-4
        assert out["placeholder/recall"] > 1.0 - 1e-4
        assert out["placeholder/f1"] > 1.0 - 1e-4

    def test_all_placeholder_rates(self):
        ppg = 3
        p   = _make_params(1, 2, ppg, 2, 2, fill=0.0)
        g   = _make_params(1, 2, ppg, 2, 2, fill=0.0)

        out = PermutationMetrics._placeholder_detection_stats(p, g, ppg)

        assert abs(out["placeholder/gt_rate"] - 1.0) < 1e-6
        assert abs(out["placeholder/pred_rate"] - 1.0) < 1e-6

    def test_metric_values_in_unit_interval(self):
        _seed()
        ppg = 3
        p   = torch.rand(2, 3 * ppg, 3, 3) - 0.5
        g   = torch.rand(2, 3 * ppg, 3, 3) - 0.5

        out = PermutationMetrics._placeholder_detection_stats(p, g, ppg)

        for k, v in out.items():
            assert 0.0 <= v <= 1.0 + 1e-6, k

    def test_per_slot_keys_present(self):
        ppg = 3
        p   = _make_params(1, 2, ppg, 1, 1)
        g   = _make_params(1, 2, ppg, 1, 1)

        out = PermutationMetrics._placeholder_detection_stats(p, g, ppg)

        for i in range(2):
            for kind in ("precision", "recall", "f1", "gt_rate", "pred_rate"):
                assert f"placeholder/{kind}/slot_{i}" in out


class TestActiveCountStats:
    def test_identical_counts_zero_mae_bias(self):
        ppg = 3
        p   = _make_params(1, 2, ppg, 2, 2, fill=0.0)
        g   = _make_params(1, 2, ppg, 2, 2, fill=0.0)
        p[:, 0] = 1.0
        g[:, 0] = 1.0

        out = PermutationMetrics._active_count_stats(p, g, ppg)

        assert out["count/mae"] == 0.0
        assert out["count/bias"] == 0.0

    def test_over_counting_positive_bias(self):
        ppg = 3
        p   = _make_params(1, 2, ppg, 1, 1, fill=0.0)
        g   = _make_params(1, 2, ppg, 1, 1, fill=0.0)
        p[:, 0] = 1.0
        p[:, 3] = 1.0

        out = PermutationMetrics._active_count_stats(p, g, ppg)

        assert out["count/bias"] == 2.0
        assert out["count/mae"] == 2.0

    def test_under_counting_negative_bias(self):
        ppg = 3
        p   = _make_params(1, 2, ppg, 1, 1, fill=0.0)
        g   = _make_params(1, 2, ppg, 1, 1, fill=0.0)
        g[:, 0] = 1.0
        g[:, 3] = 1.0

        out = PermutationMetrics._active_count_stats(p, g, ppg)

        assert out["count/bias"] == -2.0
        assert out["count/mae"] == 2.0

    def test_confusion_fractions_sum_to_one_per_gt(self):
        _seed()
        ppg = 3
        p   = torch.rand(2, 2 * ppg, 3, 3) - 0.4
        g   = torch.rand(2, 2 * ppg, 3, 3) - 0.4

        out = PermutationMetrics._active_count_stats(p, g, ppg)

        by_gt: dict[int, float] = {}
        for k, v in out.items():
            if k.startswith("count/confusion/"):
                gt_part = k.split("/")[2].split("_")[0]
                gt_k    = int(gt_part.replace("gt", ""))
                by_gt[gt_k] = by_gt.get(gt_k, 0.0) + v

        for total in by_gt.values():
            assert abs(total - 1.0) < 1e-5

    def test_confusion_present_for_matched_counts(self):
        ppg = 3
        p   = _make_params(1, 2, ppg, 1, 1, fill=0.0)
        g   = _make_params(1, 2, ppg, 1, 1, fill=0.0)
        p[:, 0] = 1.0
        g[:, 0] = 1.0

        out = PermutationMetrics._active_count_stats(p, g, ppg)

        assert out["count/confusion/gt1_pred1"] == 1.0


class TestPermutationConsensus:
    def test_keys_present(self):
        _seed()
        ppg = 3
        p   = torch.randn(2, 2 * ppg, 2, 2)
        g   = torch.randn(2, 2 * ppg, 2, 2)

        out = PermutationMetrics._permutation_consensus(p, g, ppg)

        assert set(out.keys()) == {"consensus/mean", "consensus/min", "consensus/global_dominant_frac"}

    def test_full_consensus_when_all_pixels_match_identity(self):
        ppg = 3
        p   = _make_params(1, 2, ppg, 2, 2)
        p[:, 1] = 0.0
        p[:, 4] = 10.0
        g       = p.clone()

        out = PermutationMetrics._permutation_consensus(p, g, ppg)

        assert out["consensus/mean"] == 1.0
        assert out["consensus/min"] == 1.0
        assert out["consensus/global_dominant_frac"] == 1.0

    def test_values_in_unit_interval(self):
        _seed()
        ppg = 3
        p   = torch.randn(2, 3 * ppg, 3, 3)
        g   = torch.randn(2, 3 * ppg, 3, 3)

        out = PermutationMetrics._permutation_consensus(p, g, ppg)

        for v in out.values():
            assert 0.0 <= v <= 1.0 + 1e-6

    def test_consensus_min_le_mean(self):
        _seed()
        ppg = 3
        p   = torch.randn(3, 2 * ppg, 3, 3)
        g   = torch.randn(3, 2 * ppg, 3, 3)

        out = PermutationMetrics._permutation_consensus(p, g, ppg)

        assert out["consensus/min"] <= out["consensus/mean"] + 1e-6


class TestAmplitudeCalibration:
    def test_gap_positive_when_active_higher(self):
        ppg = 3
        p   = _make_params(1, 2, ppg, 2, 2, fill=0.0)
        g   = _make_params(1, 2, ppg, 2, 2, fill=0.0)
        g[:, 0] = 1.0
        p[:, 0] = 5.0
        p[:, 3] = 0.0

        out = PermutationMetrics._amplitude_calibration(p, g, ppg)

        assert out["amp_cal/gap"] > 0.0
        assert abs(out["amp_cal/active_gt"] - 5.0) < 1e-6
        assert abs(out["amp_cal/inactive_gt"] - 0.0) < 1e-6

    def test_only_active_no_gap_key(self):
        ppg = 3
        p   = _make_params(1, 1, ppg, 2, 2, fill=0.0)
        g   = _make_params(1, 1, ppg, 2, 2, fill=0.0)
        g[:, 0] = 1.0
        p[:, 0] = 2.0

        out = PermutationMetrics._amplitude_calibration(p, g, ppg)

        assert "amp_cal/active_gt/slot_0" in out
        assert "amp_cal/gap" not in out

    def test_slot_keys_conditional_on_presence(self):
        ppg = 3
        p   = _make_params(1, 1, ppg, 2, 2, fill=0.0)
        g   = _make_params(1, 1, ppg, 2, 2, fill=0.0)

        out = PermutationMetrics._amplitude_calibration(p, g, ppg)

        assert "amp_cal/inactive_gt/slot_0" in out
        assert "amp_cal/active_gt/slot_0" not in out


class TestSigmaDegeneration:
    def test_returns_empty_when_ppg_below_three(self):
        p = _make_params(1, 2, 2, 2, 2)
        g = _make_params(1, 2, 2, 2, 2)

        out = PermutationMetrics._sigma_degeneration(p, g, 2)

        assert out == {}

    def test_reports_active_and_inactive_means(self):
        ppg = 3
        p   = _make_params(1, 2, ppg, 2, 2, fill=0.0)
        g   = _make_params(1, 2, ppg, 2, 2, fill=0.0)
        g[:, 0] = 1.0
        p[:, 2] = 3.0
        p[:, 5] = 7.0

        out = PermutationMetrics._sigma_degeneration(p, g, ppg)

        assert abs(out["sigma/active_gt/mean"] - 3.0) < 1e-6
        assert abs(out["sigma/inactive_gt/mean"] - 7.0) < 1e-6

    def test_per_slot_keys_present(self):
        ppg = 3
        p   = _make_params(1, 2, ppg, 2, 2, fill=0.0)
        g   = _make_params(1, 2, ppg, 2, 2, fill=0.0)
        g[:, 0] = 1.0

        out = PermutationMetrics._sigma_degeneration(p, g, ppg)

        assert "sigma/active_gt/mean/slot_0" in out
        assert "sigma/inactive_gt/mean/slot_1" in out


class TestSlotMuSpread:
    def test_keys_and_spread(self):
        ppg    = 3
        params = _make_params(1, 2, ppg, 2, 2, fill=0.0)
        params[:, 1] = 1.0
        params[:, 4] = 3.0

        out = PermutationMetrics._slot_mu_spread(params, ppg)

        assert abs(out["mu_mean/slot_0"] - 1.0) < 1e-6
        assert abs(out["mu_mean/slot_1"] - 3.0) < 1e-6
        assert out["mu_std/slot_0"] == 0.0
        assert out["mu_mean_spread"] > 0.0

    def test_identical_slots_zero_spread(self):
        ppg    = 3
        params = _make_params(1, 2, ppg, 2, 2, fill=0.0)
        params[:, 1] = 2.0
        params[:, 4] = 2.0

        out = PermutationMetrics._slot_mu_spread(params, ppg)

        assert out["mu_mean_spread"] == 0.0


class TestCompute:
    def test_disabled_returns_empty(self):
        pm = PermutationMetrics.silent(MetricsCfg(enabled=False))

        result = pm.compute(_make_params(1, 2, 3, 2, 2), _make_params(1, 2, 3, 2, 2), ppg=3)

        assert result == {}

    def test_enabled_returns_prefixed_metrics(self):
        _seed()
        pm = PermutationMetrics.silent(MetricsCfg(enabled=True))
        p  = torch.randn(2, 2 * 3, 3, 3)
        g  = torch.randn(2, 2 * 3, 3, 3)

        result = pm.compute(p, g, ppg=3)

        assert len(result) > 0
        assert all(k.startswith("perm/") for k in result)
        assert "perm/mu_ordering_rate" in result

    def test_compute_values_are_floats(self):
        _seed()
        pm = PermutationMetrics.silent(MetricsCfg(enabled=True))
        p  = torch.randn(1, 2 * 3, 2, 2)
        g  = torch.randn(1, 2 * 3, 2, 2)

        result = pm.compute(p, g, ppg=3)

        assert all(isinstance(v, float) for v in result.values())

    def test_compute_ppg_two_skips_sigma(self):
        _seed()
        pm = PermutationMetrics.silent(MetricsCfg(enabled=True))
        p  = torch.randn(1, 2 * 2, 2, 2)
        g  = torch.randn(1, 2 * 2, 2, 2)

        result = pm.compute(p, g, ppg=2)

        assert not any(k.startswith("perm/sigma/") for k in result)

    def test_compute_runs_on_cpu(self):
        _seed()
        pm = PermutationMetrics.silent(MetricsCfg(enabled=True))
        p  = torch.randn(1, 2 * 3, 2, 2, device="cpu")
        g  = torch.randn(1, 2 * 3, 2, 2, device="cpu")

        result = pm.compute(p, g, ppg=3)

        assert isinstance(result, dict)


class TestPermutationEinsumConsistency:
    def test_perm_cost_matches_brute_force(self):
        _seed()
        ppg = 3
        G   = 2
        p   = torch.randn(1, G * ppg, 1, 1)
        g   = torch.randn(1, G * ppg, 1, 1)

        out = PermutationMetrics._assignment_cost_margin(p, g, ppg)

        p_mu = p.reshape(1, G, ppg, 1, 1)[:, :, 1].reshape(G)
        g_mu = g.reshape(1, G, ppg, 1, 1)[:, :, 1].reshape(G)

        costs = []
        for perm in itertools.permutations(range(G)):
            c = sum(abs(float(p_mu[i]) - float(g_mu[perm[i]])) for i in range(G))
            costs.append(c)
        costs.sort()

        expected_margin = costs[1] - costs[0]

        assert abs(out["mean_margin"] - expected_margin) < 1e-4
