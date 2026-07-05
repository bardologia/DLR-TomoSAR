from __future__ import annotations

import pytest
import torch

from models.unrolled import GammaNet, TomoOperator, UNROLLED_CONFIG_REGISTRY, UNROLLED_MODEL_REGISTRY, get_unrolled


BATCH    = 2
TRACKS   = 4
POINTS   = 64
WINDOW   = 8

X_AXIS = torch.linspace(-20.0, 40.0, POINTS)
DX     = float(X_AXIS[1] - X_AXIS[0])


def _kz_map():
    torch.manual_seed(0)
    base = torch.linspace(-0.15, 0.15, TRACKS).reshape(1, TRACKS, 1, 1)

    return (base + 0.01 * torch.randn(BATCH, TRACKS, WINDOW, WINDOW)).float()


def _gaussian_profiles(mu: float = 12.0, sigma: float = 3.0):
    curve = torch.exp(-0.5 * ((X_AXIS - mu) / sigma) ** 2)
    curve = curve / (curve.sum() * DX)

    return curve.reshape(1, POINTS, 1, 1).expand(BATCH, POINTS, WINDOW, WINDOW).contiguous()


def test_registries_share_keys():
    assert set(UNROLLED_MODEL_REGISTRY) == set(UNROLLED_CONFIG_REGISTRY)


def test_operator_adjoint_identity():
    torch.manual_seed(1)
    kz       = _kz_map()
    profiles = torch.rand(BATCH, POINTS, WINDOW, WINDOW)
    coherent = torch.randn(BATCH, TRACKS, WINDOW, WINDOW) + 1j * torch.randn(BATCH, TRACKS, WINDOW, WINDOW)

    forward = TomoOperator.forward(profiles, kz, X_AXIS, DX)
    adjoint = TomoOperator.adjoint(coherent, kz, X_AXIS)

    lhs = (forward * coherent.conj()).real.sum()
    rhs = (profiles * adjoint).sum() * DX

    assert torch.allclose(lhs, rhs, rtol=1e-4, atol=1e-4)


def test_matched_filter_init_peaks_at_true_height():
    kz       = _kz_map()
    profiles = _gaussian_profiles(mu=12.0)

    measurements = TomoOperator.forward(profiles, kz, X_AXIS, DX)
    beamformed   = TomoOperator.adjoint(measurements, kz, X_AXIS)

    peak_positions = X_AXIS[beamformed.argmax(dim=1)]

    assert (peak_positions - 12.0).abs().max() < 3.0


def test_forward_shape_and_nonnegativity():
    model = get_unrolled("gamma_net", n_iterations=3, prox_hidden=4)[0].eval()

    kz           = _kz_map()
    measurements = TomoOperator.forward(_gaussian_profiles(), kz, X_AXIS, DX)

    with torch.no_grad():
        out = model(measurements, kz, X_AXIS)

    assert out.shape == (BATCH, POINTS, WINDOW, WINDOW)
    assert (out >= 0).all()
    assert torch.isfinite(out).all()


def test_backward_reaches_all_parameter_groups():
    model, config = get_unrolled("gamma_net", n_iterations=2, prox_hidden=4)
    model.train()

    kz           = _kz_map()
    measurements = TomoOperator.forward(_gaussian_profiles(), kz, X_AXIS, DX)

    out  = model(measurements, kz, X_AXIS)
    loss = out.pow(2).mean() + out.mean()
    loss.backward()

    assert model.raw_steps.grad is not None
    assert torch.isfinite(model.raw_steps.grad).all()
    assert all(p.grad is not None for p in model.prox_blocks.parameters())

    groups = config.get_param_groups(model)
    assert {g["name"] for g in groups} == {"steps", "prox"}


def test_rejects_real_measurements():
    model = get_unrolled("gamma_net", n_iterations=1)[0]

    kz = _kz_map()

    with pytest.raises(ValueError):
        model(torch.randn(BATCH, TRACKS, WINDOW, WINDOW), kz, X_AXIS)


def test_rejects_mismatched_shapes():
    model = get_unrolled("gamma_net", n_iterations=1)[0]

    kz           = _kz_map()
    measurements = TomoOperator.forward(_gaussian_profiles(), kz, X_AXIS, DX)

    with pytest.raises(ValueError):
        model(measurements[:, :2], kz, X_AXIS)
