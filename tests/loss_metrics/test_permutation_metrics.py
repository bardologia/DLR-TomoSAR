from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.metrics.permutation_metrics import PermutationMetrics


PPG = 3


def _cfg(enabled=True, amp_threshold=1e-3) -> SimpleNamespace:
    return SimpleNamespace(enabled=enabled, amp_threshold=amp_threshold)


def _params(seed: int, b=2, g=3, h=4, w=4) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(b, g * PPG, h, w, generator=gen, dtype=torch.float32)


def _reorder_groups(params: torch.Tensor, order: list[int]) -> torch.Tensor:
    b, c, h, w = params.shape
    g          = c // PPG
    grouped    = params.reshape(b, g, PPG, h, w)
    permuted   = grouped[:, order]
    return permuted.reshape(b, c, h, w)


def test_disabled_returns_empty():
    pm  = PermutationMetrics(_cfg(enabled=False))
    out = pm.compute(_params(0), _params(1), PPG)
    assert out == {}


def test_silent_constructor():
    pm = PermutationMetrics.silent(_cfg())
    assert pm.enabled is True


def test_compute_returns_finite_or_nan_floats():
    pm  = PermutationMetrics.silent(_cfg())
    out = pm.compute(_params(2), _params(3), PPG)
    assert len(out) > 0
    for k, v in out.items():
        assert isinstance(v, float)


def test_mu_ordering_rate_perfectly_ordered():
    b, g, h, w = 1, 3, 1, 1
    p = torch.zeros(b, g * PPG, h, w)
    p[0, 0, 0, 0] = 1.0
    p[0, 3, 0, 0] = 1.0
    p[0, 6, 0, 0] = 1.0
    p[0, 1, 0, 0] = -1.0
    p[0, 4, 0, 0] = 0.0
    p[0, 7, 0, 0] = 1.0
    rate = PermutationMetrics._mu_ordering_rate(p, PPG)
    assert rate == 1.0


def test_mu_ordering_rate_violation():
    b, g, h, w = 1, 3, 1, 1
    p = torch.zeros(b, g * PPG, h, w)
    p[0, 0, 0, 0] = 1.0
    p[0, 3, 0, 0] = 1.0
    p[0, 6, 0, 0] = 1.0
    p[0, 1, 0, 0] = 5.0
    p[0, 4, 0, 0] = -3.0
    p[0, 7, 0, 0] = 1.0
    rate = PermutationMetrics._mu_ordering_rate(p, PPG)
    assert rate == 0.0


def test_mu_ordering_rate_nan_when_no_multi_active():
    b, g, h, w = 1, 3, 1, 1
    p = torch.zeros(b, g * PPG, h, w)
    p[0, 0, 0, 0] = 1.0
    rate = PermutationMetrics._mu_ordering_rate(p, PPG)
    assert math.isnan(rate)


def test_perm_costs_invariant_to_pred_group_reorder():
    pred = _params(4)
    gt   = _params(5)

    base, _    = PermutationMetrics._perm_costs(pred, gt, PPG)
    reordered, _ = PermutationMetrics._perm_costs(_reorder_groups(pred, [2, 0, 1]), gt, PPG)

    base_sorted, _ = base.sort(dim=-1)
    reor_sorted, _ = reordered.sort(dim=-1)
    assert torch.allclose(base_sorted, reor_sorted, atol=1e-5)


def test_assignment_margin_invariant_to_group_reorder():
    pred = _params(6)
    gt   = _params(7)

    base = PermutationMetrics._assignment_cost_margin(pred, gt, PPG)
    reor = PermutationMetrics._assignment_cost_margin(_reorder_groups(pred, [1, 2, 0]), gt, PPG)

    assert math.isclose(base["mean_margin"], reor["mean_margin"], rel_tol=1e-4, abs_tol=1e-5)


def test_assignment_margin_zero_when_pred_matches_gt():
    pred = _params(8)
    out  = PermutationMetrics._assignment_cost_margin(pred, pred.clone(), PPG)
    assert out["mean_margin"] >= 0.0
    assert out["ambiguous_frac"] >= 0.0


def test_consensus_keys_and_range():
    out = PermutationMetrics._permutation_consensus(_params(9), _params(10), PPG)
    assert 0.0 <= out["consensus/mean"] <= 1.0
    assert 0.0 <= out["consensus/min"] <= out["consensus/mean"] + 1e-6
    assert 0.0 <= out["consensus/global_dominant_frac"] <= 1.0


def test_consensus_self_match_is_perfect():
    pred = _params(30)
    out  = PermutationMetrics._permutation_consensus(pred, pred.clone(), PPG)
    assert math.isclose(out["consensus/mean"], 1.0, abs_tol=1e-6)


def test_slot_activation_stats_rates_in_range():
    out = PermutationMetrics._slot_activation_stats(_params(11), PPG)
    for k, v in out.items():
        if k.startswith("active_rate"):
            assert 0.0 <= v <= 1.0


def test_placeholder_perfect_match_gives_f1_one():
    b, g, h, w = 1, 3, 2, 2
    p = torch.zeros(b, g * PPG, h, w)
    p[0, 0] = 1.0
    g_t = p.clone()
    out = PermutationMetrics._placeholder_detection_stats(p, g_t, PPG)
    assert out["placeholder/f1"] > 0.99
    assert out["placeholder/precision"] > 0.99
    assert out["placeholder/recall"] > 0.99


def test_active_count_stats_exact_match():
    b, g, h, w = 1, 3, 2, 2
    p = torch.zeros(b, g * PPG, h, w)
    p[0, 0] = 1.0
    p[0, 3] = 1.0
    g_t = p.clone()
    out = PermutationMetrics._active_count_stats(p, g_t, PPG)
    assert out["count/mae"] == 0.0
    assert out["count/bias"] == 0.0


def test_amplitude_calibration_gap_keys():
    b, g, h, w = 1, 3, 1, 4
    pred = _params(12, b=b, g=g, h=h, w=w)
    gt   = torch.zeros(b, g * PPG, h, w)
    gt[0, 0, 0, :2] = 1.0
    gt[0, 3, 0, :2] = 1.0
    gt[0, 6, 0, :2] = 1.0
    out = PermutationMetrics._amplitude_calibration(pred, gt, PPG)
    assert "amp_cal/gap" in out


def test_sigma_degeneration_present_for_ppg3():
    pred = _params(14)
    gt   = _params(15)
    out  = PermutationMetrics._sigma_degeneration(pred, gt, PPG)
    assert len(out) > 0


def test_sigma_degeneration_empty_for_ppg2():
    gen  = torch.Generator().manual_seed(16)
    pred = torch.rand(1, 6, 4, 4, generator=gen)
    gt   = torch.rand(1, 6, 4, 4, generator=gen)
    out  = PermutationMetrics._sigma_degeneration(pred, gt, 2)
    assert out == {}


def test_slot_mu_spread_keys():
    out = PermutationMetrics._slot_mu_spread(_params(17), PPG)
    assert "mu_mean_spread" in out


@pytest.mark.real_data
def test_compute_on_real_parameters(parameters):
    win = np.asarray(parameters[:, :8, :8]).astype(np.float32)
    p   = torch.from_numpy(win)[None]

    gen = torch.Generator().manual_seed(0)
    gt  = p + torch.randn(p.shape, generator=gen) * 0.05

    pm  = PermutationMetrics.silent(_cfg())
    out = pm.compute(p, gt, PPG)
    assert len(out) > 0
    assert "perm/mu_ordering_rate" in out
    assert "perm/count/mae" in out


@pytest.mark.real_data
def test_real_params_self_placeholder_f1(parameters):
    win = np.asarray(parameters[:, :16, :16]).astype(np.float32)
    p   = torch.from_numpy(win)[None]
    out = PermutationMetrics._placeholder_detection_stats(p, p.clone(), PPG)
    assert out["placeholder/f1"] >= 0.0
