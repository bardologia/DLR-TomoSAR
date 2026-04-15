"""
TomoSAR Simulator — refactored from pyrat.dlr.tomo.sim.TomoSAR

Simulates a tomographic SAR acquisition: generates multi-baseline signals
from scatterer distributions, forms covariance/steering matrices, and
focuses via classical spectral estimators (Capon/MUSIC/MARIA/WISE).

References
    [1] Reigber, A. (2002). Airborne polarimetric SAR tomography.
    [2] Nannini, M. (2010). Advanced SAR Tomography: Processing Algorithms
        and Constellation Design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Sequence

import math
import numpy as np
import scipy.stats as st

from pyrat.dlr.tomo.specan import fCLS, fMUSIC, fMARIA_1, fMARIA_2, fWISE
from pyrat.dlr.tomo.specan.CS import fCS


class GeometryModel(Enum):
    FLAT       = 0
    BROADSIDE  = 1
    FULL_SLOPE = 2


# Methods that accept (steering, covariance, avg_signal, ...)
SPECTRAL_METHODS: dict[str, Callable] = {
    "CLS":     fCLS,
    "MUSIC":   fMUSIC,
    "MARIA_1": fMARIA_1,
    "MARIA_2": fMARIA_2,
    "WISE":    fWISE,
}

# Methods that accept only (steering, covariance, ...)
COV_ONLY_METHODS: dict[str, Callable] = {
    "CS": fCS,
}

def normalize(x: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    xmin, xmax = x.min(), x.max()
    if xmax == xmin:
        return np.full_like(x, (lo + hi) / 2.0, dtype=float)
    return (hi - lo) * (x - xmin) / (xmax - xmin) + lo


@dataclass
class GaussianMixture:
    """Specification for a K-component Gaussian scatterer distribution."""
    centres: np.ndarray          # (K,) mean height per component
    spreads: np.ndarray          # (K,) std-dev per component
    counts:  np.ndarray          # (K,) number of scatterers per component


class ScattererDistribution:
    """Generate theoretical or Monte-Carlo scatterer height distributions."""

    @staticmethod
    def theoretical(spec: GaussianMixture, heights: np.ndarray) -> np.ndarray:
        profile = np.zeros_like(heights, dtype=float)
        total   = spec.counts.sum()
        
        for mu, sigma, n in zip(spec.centres, spec.spreads, spec.counts):
            profile += st.norm.pdf(heights, loc=mu, scale=sigma) * (n / total)
        
        return normalize(profile, 0.0, 1.0)

    @staticmethod
    def sample(spec: GaussianMixture) -> np.ndarray:
        parts = [
            np.random.normal(loc=mu, scale=sigma, size=int(n))
            for mu, sigma, n in zip(spec.centres, spec.spreads, spec.counts)
        ]
        return np.concatenate(parts)

    @staticmethod
    def theoretical_perturbed(
        spec: GaussianMixture,
        n_height_bins: int,
        centre_jitter: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        centres = spec.centres.copy()
        if centre_jitter > 0:
            centres += np.random.uniform(-centre_jitter, centre_jitter, size=centres.shape)

        samples = np.concatenate([
            np.random.normal(loc=c, scale=s, size=int(n))
            for c, s, n in zip(centres, spec.spreads, spec.counts)
        ])
        heights = np.linspace(samples.min() - 1, samples.max() + 1, n_height_bins)

        total   = spec.counts.sum()
        profile = np.zeros(n_height_bins, dtype=float)
        for c, s, n in zip(centres, spec.spreads, spec.counts):
            profile += st.norm.pdf(heights, loc=c, scale=s) * (n / total)

        return heights, profile


class SlantRangeGeometry:
    """Compute slant-range distances for different acquisition models."""

    def __init__(
        self,
        baselines:    np.ndarray,
        ground_range: float,
        platform_h:   float,
        wavelength:   float,
        slope_rad:    float = 0.0,
        model:        GeometryModel = GeometryModel.FULL_SLOPE,
    ):
        self.baselines    = np.asarray(baselines, dtype=float)
        self.ground_range = ground_range
        self.platform_h   = platform_h
        self.wavelength   = wavelength
        self.slope_rad    = slope_rad
        self.model        = model
        self.n_tracks     = self.baselines.size

    def ranges_to(self, height: float) -> np.ndarray:
        bl = self.baselines
        gr = self.ground_range
        h  = self.platform_h

        if self.model == GeometryModel.FLAT:
            r = np.empty(self.n_tracks, dtype=float)
            neg = bl < 0
            r[neg]  = np.sqrt((bl[neg]  + height) ** 2 + gr ** 2)
            r[~neg] = np.sqrt((bl[~neg] - height) ** 2 + gr ** 2)
            return r

        if self.model == GeometryModel.BROADSIDE:
            return np.sqrt((h - height) ** 2 + (gr + bl) ** 2)

        if self.model == GeometryModel.FULL_SLOPE:
            r0      = math.sqrt(gr ** 2 + (h - height) ** 2)
            theta_0 = math.acos(((h - height) ** 2 + r0 ** 2 - gr ** 2) / (2 * (h - height) * r0))
            slope   = self.slope_rad

            r = np.empty(self.n_tracks, dtype=float)
            neg = bl < 0
            r[neg]   = np.sqrt(r0 ** 2 + bl[neg]  ** 2 + 2 * r0 * bl[neg]  * math.sin(theta_0 - slope))
            r[~neg]  = np.sqrt(r0 ** 2 + bl[~neg] ** 2 + 2 * r0 * bl[~neg] * math.sin(theta_0 + slope))
            return r

    def phase_shift(self, height: float) -> np.ndarray:
        return np.exp(-1j * 4 * math.pi * self.ranges_to(height) / self.wavelength)

    def phase_shift_conj(self, height: float) -> np.ndarray:
        return np.exp(1j * 4 * math.pi * self.ranges_to(height) / self.wavelength)


class Beamforming:
    """Beamforming/MSF focusing directly on the multi-look signal matrix."""

    def __init__(self, geometry: SlantRangeGeometry):
        self.geom = geometry

    def focus(self, signal: np.ndarray, height_axis: np.ndarray) -> np.ndarray:
        n_bins   = height_axis.size
        n_looks  = signal.shape[1]
        spectrum = np.zeros((n_bins, n_looks), dtype=complex)

        for k, z_k in enumerate(height_axis):
            steering = self.geom.phase_shift(z_k)  # (L,)
            spectrum[k] = steering @ signal        # (L,)·(L, J) → (J,)

        return np.abs(spectrum.mean(axis=1))       # (K,)


# ─── Main simulator ─────────────────────────────────────────────────────────

@dataclass
class TomoSARConfig:
    wavelength:   float      = 0.23
    ground_range: float      = 4000.0
    platform_h:   float      = 3000.0
    slope_deg:    float      = 0.0
    n_looks:      int        = 350
    n_height_bins: int       = 400
    noise_power:  float      = 0.01
    baselines:    np.ndarray = field(default_factory=lambda: np.linspace(0, 90, num=9))
    height_range: tuple[float, float] = (-15.0, 15.0)
    model:        GeometryModel       = GeometryModel.FULL_SLOPE


class TomoSARSimulator:
    """End-to-end TomoSAR simulation: signal → covariance → focusing."""

    def __init__(self, cfg: TomoSARConfig | None = None):
        self.cfg = cfg or TomoSARConfig()
        c = self.cfg

        self.geometry = SlantRangeGeometry(
            baselines    = c.baselines,
            ground_range = c.ground_range,
            platform_h   = c.platform_h,
            wavelength   = c.wavelength,
            slope_rad    = math.radians(c.slope_deg),
            model        = c.model,
        )

        self.n_tracks      = self.geometry.n_tracks
        self.height_axis   = np.linspace(c.height_range[0], c.height_range[1], c.n_height_bins)
        self.signal        = np.zeros((self.n_tracks, c.n_looks), dtype=complex)
        self.distribution  = np.zeros((c.n_looks, c.n_height_bins), dtype=float)
        self.covariance    = np.zeros((self.n_tracks, self.n_tracks), dtype=complex)
        self.avg_signal    = np.zeros(self.n_tracks, dtype=complex)
        self.steering      = np.zeros((self.n_tracks, c.n_height_bins), dtype=complex)
        self.spectrum      = np.zeros(c.n_height_bins, dtype=float)

    # ── Signal generation ────────────────────────────────────────────────

    def simulate_signal(self, spec: GaussianMixture) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c                    = self.cfg
        self.signal[:]       = 0
        self.distribution[:] = 0

        for j in range(c.n_looks):
            sample_heights = ScattererDistribution.sample(spec)
            for h in sample_heights:
                self.signal[:, j] += self.geometry.phase_shift_conj(h)
                nearest            = np.argmin(np.abs(self.height_axis - h))
                self.distribution[j, nearest] = 1

        ref_profile = ScattererDistribution.theoretical(spec, self.height_axis)
        return self.signal, self.distribution, ref_profile

    # ── Covariance matrix ────────────────────────────────────────────────

    def compute_covariance(self) -> np.ndarray:
        c  = self.cfg
        n  = self._make_noise()
        observed = self.signal + n

        self.covariance = (observed.conj() @ observed.T) / c.n_looks
        self.avg_signal = observed.mean(axis=1)
        return self.covariance

    def _make_noise(self) -> np.ndarray:
        c = self.cfg
        
        if c.noise_power == 0:
            return np.zeros_like(self.signal)
        
        amplitude = np.random.normal(0, c.noise_power ** 2, (self.n_tracks, c.n_looks))
        phase     = np.random.uniform(0, math.pi / 2,       (self.n_tracks, c.n_looks))
        return amplitude * np.exp(1j * phase)

    # ── Steering matrix ──────────────────────────────────────────────────

    def compute_steering(self) -> np.ndarray:
        for k, z_k in enumerate(self.height_axis):
            self.steering[:, k] = self.geometry.phase_shift(z_k)
        return self.steering

    # ── Tomographic focusing ─────────────────────────────────────────────

    def focus(self, method: str | Callable = "CLS", *args, **kwargs) -> np.ndarray:
        if callable(method):
            # Caller is responsible for passing the right signature.
            # Try (steering, cov, avg_signal); fall back to (steering, cov).
            try:
                self.spectrum = np.abs(
                    method(self.steering, self.covariance, self.avg_signal, *args, **kwargs)
                )
            except TypeError:
                self.spectrum = np.abs(
                    method(self.steering, self.covariance, *args, **kwargs)
                )
            return self.spectrum

        if method == "Fourier":
            self.spectrum = Beamforming(self.geometry).focus(self.signal, self.height_axis)
            return self.spectrum

        if method in SPECTRAL_METHODS:
            fn = SPECTRAL_METHODS[method]
            self.spectrum = np.abs(fn(self.steering, self.covariance, self.avg_signal, *args, **kwargs))
        elif method in COV_ONLY_METHODS:
            fn = COV_ONLY_METHODS[method]
            self.spectrum = np.abs(fn(self.steering, self.covariance, *args, **kwargs))
        else:
            raise ValueError(f"Unknown spectral method: {method!r}")

        return self.spectrum

    # ── Full pipeline convenience ────────────────────────────────────────

    def run(
        self,
        spec:   GaussianMixture,
        method: str | Callable = "CLS",
        *args,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        
        signal, dist, ref = self.simulate_signal(spec)
        cov                = self.compute_covariance()
        steer              = self.compute_steering()
        spectrum           = self.focus(method, *args, **kwargs)

        return {
            "height_axis":  self.height_axis,
            "signal":       signal,
            "distribution": dist,
            "ref_profile":  ref,
            "covariance":   cov,
            "steering":     steer,
            "spectrum":     spectrum,
        }
