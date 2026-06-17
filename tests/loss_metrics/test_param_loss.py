from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.loss.param_loss import ParamLoss


PARAM_NAMES = ["amp", "mu", "sigma"]


def _params(seed: int, b=2, g=5, h=4, w=4) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(b, g, 3, h, w, generator=gen, dtype=torch.float64)


def _weights(b=2, g=5, h=4, w=4) -> torch.Tensor:
    return torch.ones(b, g, 3, h, w, dtype=torch.float64)


def test_l1_zero_on_identical():
    p     = _params(0)
    total, per = ParamLoss.l1(p, p.clone(), _weights(), PARAM_NAMES)
    assert total.item() == 0.0
    for v in per.values():
        assert v.item() == 0.0


def test_l1_nonnegative():
    total, _ = ParamLoss.l1(_params(1), _params(2), _weights(), PARAM_NAMES)
    assert total.item() >= 0.0


def test_l1_per_param_keys_and_sum():
    pred = _params(3)
    gt   = _params(4)
    total, per = ParamLoss.l1(pred, gt, _weights(), PARAM_NAMES)
    assert set(per.keys()) == set(PARAM_NAMES)

    recombined = sum(p.item() for p in per.values()) / len(PARAM_NAMES)
    assert abs(recombined - total.item()) < 1e-9


def test_l1_weight_zero_kills_term():
    pred = _params(5)
    gt   = _params(6)
    w    = _weights()
    w[:, :, 1] = 0.0
    _, per = ParamLoss.l1(pred, gt, w, PARAM_NAMES)
    assert per["mu"].item() == 0.0
    assert per["amp"].item() > 0.0


def test_l1_gradient_flow():
    pred = _params(7).requires_grad_(True)
    gt   = _params(8)
    total, _ = ParamLoss.l1(pred, gt, _weights(), PARAM_NAMES)
    total.backward()
    assert torch.isfinite(pred.grad).all()


def test_huber_zero_on_identical():
    p = _params(9)
    out = ParamLoss.huber(p, p.clone(), _weights(), 1.0)
    assert out.item() == 0.0


def test_huber_nonnegative():
    out = ParamLoss.huber(_params(10), _params(11), _weights(), 1.0)
    assert out.item() >= 0.0


def test_huber_quadratic_branch():
    pred = torch.full((1, 1, 3, 1, 1), 0.1, dtype=torch.float64)
    gt   = torch.zeros_like(pred)
    out  = ParamLoss.huber(pred, gt, torch.ones_like(pred), 1.0)
    diff = pred - gt
    assert torch.allclose(out, (0.5 * diff * diff).mean())


def test_huber_gradient_flow():
    pred = _params(12).requires_grad_(True)
    gt   = _params(13)
    ParamLoss.huber(pred, gt, _weights(), 1.0).backward()
    assert torch.isfinite(pred.grad).all()


def test_tv_zero_on_constant():
    flat = torch.ones(1, 5, 3, 6, 6, dtype=torch.float64) * 0.3
    assert ParamLoss.tv(flat).item() == 0.0


def test_tv_nonnegative_and_positive_on_variation():
    out = ParamLoss.tv(_params(14, h=6, w=6))
    assert out.item() > 0.0


def test_tv_gradient_flow():
    p = _params(15, h=6, w=6).requires_grad_(True)
    ParamLoss.tv(p).backward()
    assert torch.isfinite(p.grad).all()


def test_match_passthrough_other_strategy():
    pred, pp, gt, gp = (_params(i) for i in (16, 17, 18, 19))
    out = ParamLoss.match("none", pred, pp, gt, gp)
    assert out[0] is pred
    assert torch.equal(out[2], gt)


def test_match_sort_gt_by_mu_is_permutation_invariant():
    b, g, h, w = 1, 3, 1, 1
    gt = torch.zeros(b, g, 3, h, w, dtype=torch.float64)
    gt[0, :, 0, 0, 0] = torch.tensor([1.0, 1.0, 1.0])
    gt[0, :, 1, 0, 0] = torch.tensor([5.0, -2.0, 1.0])
    gt_phys = gt.clone()

    pred = torch.zeros_like(gt)

    _, _, sorted_gt, _ = ParamLoss.match("sort_gt_by_mu", pred, pred.clone(), gt, gt_phys)
    mus = sorted_gt[0, :, 1, 0, 0]
    assert torch.all(mus[:-1] <= mus[1:])


def test_match_sort_gt_by_mu_pushes_inactive_last():
    b, g, h, w = 1, 3, 1, 1
    gt = torch.zeros(b, g, 3, h, w, dtype=torch.float64)
    gt[0, :, 0, 0, 0] = torch.tensor([1.0, 0.0, 1.0])
    gt[0, :, 1, 0, 0] = torch.tensor([5.0, -8.0, 1.0])
    gt_phys = gt.clone()
    pred    = torch.zeros_like(gt)

    _, _, sorted_gt, sorted_phys = ParamLoss.match("sort_gt_by_mu", pred, pred.clone(), gt, gt_phys)
    last_amp = sorted_phys[0, -1, 0, 0, 0]
    assert last_amp.item() == 0.0


@pytest.mark.real_data
def test_param_loss_on_real_parameters(parameters):
    win = np.asarray(parameters[:, :8, :8]).astype(np.float64)
    t   = torch.from_numpy(win).reshape(5, 3, 8, 8)[None]

    pred = t.clone()
    pred[:, :, 0] += 0.02

    total, per = ParamLoss.l1(pred, t, torch.ones_like(t), PARAM_NAMES)
    assert total.item() > 0.0
    assert per["mu"].item() < 1e-9
    assert per["amp"].item() > 0.0

    self_loss, _ = ParamLoss.l1(t, t.clone(), torch.ones_like(t), PARAM_NAMES)
    assert self_loss.item() == 0.0
