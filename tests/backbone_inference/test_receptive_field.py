from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pipelines.backbone.inference.probes          import ProbeWindows
from pipelines.backbone.inference.analysis.receptive_field import ErfComputation


WINDOW = 16


class _ConvModel(torch.nn.Module):
    def __init__(self, kernel: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, kernel_size=kernel, padding=kernel // 2, bias=False)
        torch.nn.init.constant_(self.conv.weight, 1.0)

    def forward(self, x):
        return self.conv(x)


class _DeadModel(torch.nn.Module):
    def forward(self, x):
        return x * 0.0


class _AmpOnlyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        torch.nn.init.constant_(self.conv.weight, 1.0)

    def forward(self, x):
        amp = self.conv(x)
        return torch.cat([amp, torch.ones_like(amp) * 5.0, torch.ones_like(amp) * 2.0], dim=1)


def _windows(n_probes: int = 2) -> torch.Tensor:
    rng = np.random.default_rng(0)
    return torch.from_numpy(rng.uniform(0.5, 1.0, size=(n_probes, 2, WINDOW, WINDOW))).float()


def test_conv3_mass_is_complete_inside_a_small_window():
    erf     = ErfComputation(_ConvModel(3), WINDOW, [4, 8, 16])
    maps, _ = erf.gradient_maps(_windows())
    grad    = maps["combined"]

    masses = erf.mass_profile(grad)

    assert grad.shape  == (WINDOW, WINDOW)
    assert masses[4]   == pytest.approx(1.0)
    assert masses[16]  == pytest.approx(1.0)


def test_wider_kernel_has_wider_sigma():
    erf3 = ErfComputation(_ConvModel(3), WINDOW, [8])
    erf9 = ErfComputation(_ConvModel(9), WINDOW, [8])

    sigma3 = erf3.sigma(erf3.gradient_maps(_windows())[0]["combined"])
    sigma9 = erf9.sigma(erf9.gradient_maps(_windows())[0]["combined"])

    assert sigma9[0] > sigma3[0]
    assert sigma9[1] > sigma3[1]


def test_dead_model_raises():
    erf = ErfComputation(_DeadModel(), WINDOW, [8])

    with pytest.raises(ValueError):
        erf.gradient_maps(_windows())


def test_mass_windows_exceeding_the_probe_window_raise():
    with pytest.raises(ValueError):
        ErfComputation(_ConvModel(3), WINDOW, [8, 32])


def test_family_maps_follow_channel_dependence():
    erf     = ErfComputation(_AmpOnlyModel(), WINDOW, [8])
    maps, _ = erf.gradient_maps(_windows())

    assert maps["amp"]   is not None
    assert maps["mu"]    is None
    assert maps["sigma"] is None
    assert np.allclose(maps["combined"], maps["amp"])


def test_probe_sigmas_cover_every_probe():
    erf              = ErfComputation(_ConvModel(3), WINDOW, [8])
    _, probe_sigmas  = erf.gradient_maps(_windows(n_probes=3))

    assert probe_sigmas.shape == (3, 2)
    assert (probe_sigmas > 0.0).all()


def test_fine_mass_curve_is_monotone_and_reaches_one():
    erf         = ErfComputation(_ConvModel(3), WINDOW, [8])
    maps, _     = erf.gradient_maps(_windows())
    sides, fine = erf.fine_mass_curve(maps["combined"])

    assert sides[-1] == WINDOW
    assert fine[-1]  == pytest.approx(1.0)
    assert (np.diff(fine) >= -1e-12).all()


def test_side_at_mass_interpolates_between_samples():
    sides  = np.array([2.0, 4.0, 6.0])
    masses = np.array([0.4, 0.8, 1.0])

    assert ErfComputation.side_at_mass(sides, masses, 0.6)  == pytest.approx(3.0)
    assert ErfComputation.side_at_mass(sides, masses, 0.1)  == pytest.approx(2.0)
    assert ErfComputation.side_at_mass(sides, masses, 1.0)  == pytest.approx(6.0)


def test_gaussian_grad_matches_gaussian_mass_reference():
    erf    = ErfComputation(_ConvModel(3), WINDOW, [8])
    center = WINDOW // 2
    coords = np.arange(WINDOW) - center
    rr, cc = np.meshgrid(coords, coords, indexing="ij")
    grad   = np.exp(-(rr ** 2 + cc ** 2) / (2.0 * 2.0 ** 2))

    sigma_az, sigma_rg = erf.sigma(grad)
    within             = erf.mass_within_2sigma(grad, sigma_az, sigma_rg)

    assert within == pytest.approx(0.911, abs=0.05)


def test_gaussian_mass_requires_positive_sigmas():
    with pytest.raises(ValueError):
        ErfComputation.gaussian_mass(np.array([2.0]), 0.0, 1.0)


def _region_run(n_az: int, n_rg: int) -> SimpleNamespace:
    return SimpleNamespace(split_region=SimpleNamespace(azimuth_size=n_az, range_size=n_rg))


def test_probe_centers_keep_the_window_margin():
    centers = ProbeWindows(_region_run(32, 20), window=8).centers(3, 2)

    azs = sorted({az for az, _rg in centers})
    rgs = sorted({rg for _az, rg in centers})

    assert len(centers) == 6
    assert azs[0] >= 4 and azs[-1] <= 27
    assert rgs[0] >= 4 and rgs[-1] <= 15


def test_probe_centers_raise_when_the_region_is_too_small():
    with pytest.raises(ValueError):
        ProbeWindows(_region_run(32, 200), window=64).centers(2, 2)
