from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from tools.tomo_geometry import TomoGeometry


@dataclass
class GeoCfg:
    wavelength  : float = 0.23
    slant_range : float = 5000.0
    baselines   : tuple = (0.0, 11.25, 22.5)
    kz_values   : tuple = ()


def _axis(n: int = 11, lo: float = -10.0, hi: float = 10.0, dtype=torch.float32) -> torch.Tensor:
    return torch.linspace(lo, hi, n, dtype=dtype)


class TestKzConstruction:
    def test_explicit_kz_values_used_verbatim(self):
        cfg  = GeoCfg(kz_values=(0.1, 0.2, 0.3))
        geo  = TomoGeometry(cfg, _axis())

        assert geo.n_tracks == 3
        assert torch.allclose(geo.kz, torch.tensor([0.1, 0.2, 0.3]))

    def test_baselines_scaled_when_no_kz(self):
        cfg   = GeoCfg(baselines=(0.0, 10.0), kz_values=())
        geo   = TomoGeometry(cfg, _axis())

        scale = 4.0 * math.pi / (cfg.wavelength * cfg.slant_range)
        expected = torch.tensor([0.0, scale * 10.0])

        assert geo.n_tracks == 2
        assert torch.allclose(geo.kz, expected, atol=1e-6)

    def test_kz_takes_priority_over_baselines(self):
        cfg = GeoCfg(baselines=(0.0, 99.0), kz_values=(1.0, 2.0))
        geo = TomoGeometry(cfg, _axis())

        assert torch.allclose(geo.kz, torch.tensor([1.0, 2.0]))

    def test_n_tracks_matches_kz_length(self):
        cfg = GeoCfg(kz_values=(0.1, 0.2, 0.3, 0.4, 0.5))
        geo = TomoGeometry(cfg, _axis())

        assert geo.n_tracks == 5
        assert geo.kz.shape[0] == 5


class TestDtypeAndDevice:
    def test_kz_matches_axis_dtype(self):
        axis = _axis(dtype=torch.float64)
        geo  = TomoGeometry(GeoCfg(kz_values=(0.1, 0.2)), axis)

        assert geo.kz.dtype == torch.float64

    def test_kz_on_axis_device(self):
        axis = _axis()
        geo  = TomoGeometry(GeoCfg(kz_values=(0.1,)), axis)

        assert geo.kz.device == axis.device
        assert geo.steering.device == axis.device

    def test_steering_is_complex(self):
        geo = TomoGeometry(GeoCfg(kz_values=(0.1, 0.2)), _axis())

        assert geo.steering.is_complex()
        assert geo.outer.is_complex()


class TestSteeringShapes:
    def test_steering_shape(self):
        n_axis = 11
        geo    = TomoGeometry(GeoCfg(kz_values=(0.1, 0.2, 0.3)), _axis(n_axis))

        assert geo.steering.shape == (3, n_axis)

    def test_outer_shape(self):
        n_axis = 7
        geo    = TomoGeometry(GeoCfg(kz_values=(0.1, 0.2, 0.3)), _axis(n_axis))

        assert geo.outer.shape == (3, 3, n_axis)

    def test_steering_unit_modulus(self):
        geo = TomoGeometry(GeoCfg(kz_values=(0.1, 0.5, 1.0)), _axis())

        assert torch.allclose(geo.steering.abs(), torch.ones_like(geo.steering.abs()), atol=1e-5)

    def test_steering_phase_matches_kz_times_axis(self):
        axis = _axis(5, dtype=torch.float64)
        cfg  = GeoCfg(kz_values=(0.3, 0.7))
        geo  = TomoGeometry(cfg, axis)

        phase    = geo.kz.reshape(-1, 1) * axis.reshape(1, -1)
        expected = torch.polar(torch.ones_like(phase), phase)

        assert torch.allclose(geo.steering, expected, atol=1e-6)


class TestOuterProduct:
    def test_outer_is_steering_self_outer(self):
        axis = _axis(6, dtype=torch.float64)
        geo  = TomoGeometry(GeoCfg(kz_values=(0.2, 0.4, 0.6)), axis)

        expected = torch.einsum("ik,jk->ijk", geo.steering, geo.steering.conj())

        assert torch.allclose(geo.outer, expected)

    def test_outer_diagonal_is_one(self):
        geo = TomoGeometry(GeoCfg(kz_values=(0.1, 0.2, 0.3)), _axis())

        n = geo.n_tracks
        for i in range(n):
            diag = geo.outer[i, i]
            assert torch.allclose(diag.real, torch.ones_like(diag.real), atol=1e-5)
            assert torch.allclose(diag.imag, torch.zeros_like(diag.imag), atol=1e-5)

    def test_outer_hermitian_across_track_pairs(self):
        geo = TomoGeometry(GeoCfg(kz_values=(0.1, 0.5, 0.9)), _axis())

        for i in range(geo.n_tracks):
            for j in range(geo.n_tracks):
                assert torch.allclose(geo.outer[i, j], geo.outer[j, i].conj(), atol=1e-5)


class TestSingleTrack:
    def test_single_track_shapes(self):
        n_axis = 9
        geo    = TomoGeometry(GeoCfg(kz_values=(0.4,)), _axis(n_axis))

        assert geo.n_tracks == 1
        assert geo.steering.shape == (1, n_axis)
        assert geo.outer.shape == (1, 1, n_axis)


class TestZeroKz:
    def test_zero_kz_gives_constant_steering(self):
        geo = TomoGeometry(GeoCfg(kz_values=(0.0,)), _axis())

        assert torch.allclose(geo.steering.real, torch.ones_like(geo.steering.real))
        assert torch.allclose(geo.steering.imag, torch.zeros_like(geo.steering.imag))


class TestDescribe:
    def test_describe_keys(self):
        geo = TomoGeometry(GeoCfg(kz_values=(0.1, 0.2)), _axis())
        d   = geo.describe()

        assert set(d.keys()) == {"Tracks", "kz min", "kz max", "Wavelength", "Slant range", "kz source", "Baselines origin"}

    def test_describe_reports_track_count(self):
        geo = TomoGeometry(GeoCfg(kz_values=(0.1, 0.2, 0.3)), _axis())

        assert geo.describe()["Tracks"] == 3

    def test_describe_kz_source_explicit(self):
        geo = TomoGeometry(GeoCfg(kz_values=(0.1, 0.2)), _axis())

        assert geo.describe()["kz source"] == "explicit kz_values"

    def test_describe_kz_source_baselines(self):
        geo = TomoGeometry(GeoCfg(baselines=(0.0, 10.0), kz_values=()), _axis())

        assert geo.describe()["kz source"] == "baselines via 4*pi*b/(lambda*r0)"

    def test_describe_min_max_formatted(self):
        geo = TomoGeometry(GeoCfg(kz_values=(0.1, 0.5)), _axis())
        d   = geo.describe()

        assert d["kz min"] == "0.1000 rad/m"
        assert d["kz max"] == "0.5000 rad/m"

    def test_describe_wavelength_and_range_strings(self):
        cfg = GeoCfg(wavelength=0.23, slant_range=5000.0, kz_values=(0.1,))
        d   = TomoGeometry(cfg, _axis()).describe()

        assert d["Wavelength"] == "0.23 m"
        assert d["Slant range"] == "5000.0 m"


class TestEmptyConfiguration:
    def test_empty_baselines_and_kz_marks_xfail_on_describe(self):
        cfg = GeoCfg(baselines=(), kz_values=())
        geo = TomoGeometry(cfg, _axis())

        assert geo.n_tracks == 0
        assert geo.steering.shape == (0, _axis().shape[0])
        assert geo.describe()["kz min"] == "n/a"
        assert geo.describe()["kz max"] == "n/a"
