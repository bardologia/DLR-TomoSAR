from __future__ import annotations

import pytest
import torch

from configuration.training.profile_autoencoder  import ProfileAeLossConfig
from pipelines.profile_autoencoder.training.loss import Loss


CURVE_KINDS = ["mse", "l1", "huber", "charbonnier"]


def _loss(kind):
    return Loss(ProfileAeLossConfig(curve_kind=kind))


def _curve_pair(seed=0):
    g     = torch.Generator().manual_seed(seed)
    recon = torch.randn(4, 16, 1, 1, generator=g)
    curve = torch.randn(4, 16, 1, 1, generator=g)
    return recon, curve


def test_loss_generation_is_zero():
    assert _loss("mse").loss_generation == 0


@pytest.mark.parametrize("kind", CURVE_KINDS)
def test_forward_returns_finite_scalar(kind):
    recon, curve = _curve_pair()
    out          = _loss(kind)(recon, curve)

    total = out["total_loss"]

    assert total.ndim == 0
    assert torch.isfinite(total)
    assert total.item() >= 0.0


@pytest.mark.parametrize("kind", CURVE_KINDS)
def test_dict_structure_and_consistency(kind):
    recon, curve = _curve_pair()
    out          = _loss(kind)(recon, curve)

    assert set(out.keys()) == {"total_loss", "components", "monitor", "occupancy", "physical"}
    assert set(out["components"].keys()) == {"curve_recon"}
    assert out["monitor"] == {}

    assert torch.equal(out["total_loss"], out["components"]["curve_recon"])


@pytest.mark.parametrize("kind", ["mse", "l1", "huber"])
def test_zero_on_identical_input(kind):
    curve = torch.randn(4, 16, 1, 1)
    out   = _loss(kind)(curve.clone(), curve.clone())

    assert out["total_loss"].item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("kind", CURVE_KINDS)
def test_gradient_flow(kind):
    recon, curve = _curve_pair()
    recon        = recon.clone().requires_grad_(True)

    out = _loss(kind)(recon, curve)
    out["total_loss"].backward()

    assert recon.grad is not None
    assert torch.isfinite(recon.grad).all()
    assert recon.grad.abs().sum() > 0.0


def test_mse_matches_manual():
    recon, curve = _curve_pair()
    out          = _loss("mse")(recon, curve)

    manual = ((recon - curve) ** 2).mean()

    assert out["total_loss"].item() == pytest.approx(manual.item(), rel=1e-6)


def test_l1_matches_manual():
    recon, curve = _curve_pair()
    out          = _loss("l1")(recon, curve)

    manual = (recon - curve).abs().mean()

    assert out["total_loss"].item() == pytest.approx(manual.item(), rel=1e-6)


def test_huber_delta_changes_value():
    recon, curve = _curve_pair()

    small = Loss(ProfileAeLossConfig(curve_kind="huber", huber_delta=0.1))(recon, curve)
    large = Loss(ProfileAeLossConfig(curve_kind="huber", huber_delta=5.0))(recon, curve)

    assert small["total_loss"].item() != large["total_loss"].item()


def test_charbonnier_eps_floor_on_identical():
    curve = torch.zeros(4, 16, 1, 1)
    eps   = 1e-2
    out   = Loss(ProfileAeLossConfig(curve_kind="charbonnier", charbonnier_eps=eps))(curve, curve)

    assert out["total_loss"].item() == pytest.approx(eps, abs=1e-5)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        _loss("nope")(*_curve_pair())


def test_sobolev_off_by_default():
    out = _loss("mse")(*_curve_pair())

    assert set(out["components"]) == {"curve_recon"}


def test_sobolev_adds_component_and_weighted_total():
    recon, curve = _curve_pair()
    out          = Loss(ProfileAeLossConfig(curve_kind="mse", use_sobolev=True, weight_sobolev=2.0))(recon, curve)

    assert set(out["components"]) == {"curve_recon", "curve_sobolev"}

    expected = out["components"]["curve_recon"] + 2.0 * out["components"]["curve_sobolev"]
    assert out["total_loss"].item() == pytest.approx(expected.item(), rel=1e-6)


def test_sobolev_zero_for_linear_residual():
    curve = _curve_pair()[1]
    ramp  = torch.linspace(0.0, 1.0, curve.shape[1]).view(1, -1, 1, 1)
    out   = Loss(ProfileAeLossConfig(curve_kind="mse", use_sobolev=True, weight_sobolev=1.0))(curve + ramp, curve)

    assert out["components"]["curve_recon"].item() > 0.0
    assert out["components"]["curve_sobolev"].item() == pytest.approx(0.0, abs=1e-10)


def test_sobolev_penalizes_jitter_over_smooth_error_at_equal_mse():
    target = torch.zeros(1, 16, 1, 1)
    offset = torch.full((1, 16, 1, 1), 0.5)
    jitter = 0.5 * torch.tensor([1.0, -1.0]).repeat(8).view(1, 16, 1, 1)
    loss   = Loss(ProfileAeLossConfig(curve_kind="mse", use_sobolev=True, weight_sobolev=1.0))

    smooth_out = loss(target + offset, target)
    jitter_out = loss(target + jitter, target)

    assert smooth_out["components"]["curve_recon"].item() == pytest.approx(jitter_out["components"]["curve_recon"].item(), rel=1e-6)
    assert jitter_out["total_loss"].item() > smooth_out["total_loss"].item()
