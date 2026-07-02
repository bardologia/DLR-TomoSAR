from __future__ import annotations

from typing import List

import numpy as np
import torch


class GaussianAxis:
    @staticmethod
    def build(x_min: float, x_max: float, length: int) -> np.ndarray:
        return np.linspace(x_min, x_max, length, dtype=np.float32)


class GaussianMixture:
    SIGMA_FLOOR = 1e-6
    EXPON_FLOOR = -100.0
    EXPON_CEIL  = 0.0

    @classmethod
    def _safe_sigma_sq(cls, sigmas: np.ndarray) -> np.ndarray:
        clamped = np.maximum(sigmas, cls.SIGMA_FLOOR)
        return 2.0 * clamped ** 2

    @classmethod
    def evaluate_batch(cls, height_axis: np.ndarray, amps: np.ndarray, mus: np.ndarray, sigs: np.ndarray) -> np.ndarray:
        pred = np.zeros((amps.shape[0], height_axis.shape[0]), dtype=np.float32)
        h    = height_axis[None, :]

        for g in range(amps.shape[1]):
            amp_g  = amps[:, g:g + 1]
            mu_g   = mus [:, g:g + 1]
            sig_sq = cls._safe_sigma_sq(sigs[:, g:g + 1])

            expon  = np.clip(-((h - mu_g) ** 2) / sig_sq, cls.EXPON_FLOOR, cls.EXPON_CEIL)
            pred  += amp_g * np.exp(expon)

        return pred

    @classmethod
    def evaluate_slice(cls, parameters_array: np.ndarray, h_val: float, n_gaussians: int) -> np.ndarray:
        reconstructed = np.zeros(parameters_array.shape[1:], dtype=np.float32)

        for k in range(n_gaussians):
            amp    = parameters_array[3 * k    ]
            mu     = parameters_array[3 * k + 1]
            sig    = parameters_array[3 * k + 2]
            sig_sq = cls._safe_sigma_sq(sig)

            expon          = np.clip(-((h_val - mu) ** 2) / sig_sq, cls.EXPON_FLOOR, cls.EXPON_CEIL)
            reconstructed += amp * np.exp(expon)

        return reconstructed

    @classmethod
    def evaluate_pixel(cls, params: np.ndarray, height_axis: np.ndarray, n_gaussians: int) -> tuple:
        components = []
        total      = np.zeros_like(height_axis, dtype=np.float64)

        for k in range(n_gaussians):
            amp    = float(params[3 * k    ])
            mu     = float(params[3 * k + 1])
            sig    = float(params[3 * k + 2])
            sig_sq = 2.0 * max(sig, cls.SIGMA_FLOOR) ** 2

            expon = np.clip(-((height_axis - mu) ** 2) / sig_sq, cls.EXPON_FLOOR, cls.EXPON_CEIL)
            comp  = amp * np.exp(expon)
            components.append(comp)
            total += comp

        return total, components


class GaussianReconstructor:
    @staticmethod
    def _single(a: np.ndarray, mu: np.ndarray, sig: np.ndarray, x: np.ndarray) -> np.ndarray:
        return a * np.exp(-((x - mu) ** 2) / (2.0 * sig * sig + 1e-8))

    @staticmethod
    def reconstruct_batch(gauss: np.ndarray, x: np.ndarray) -> np.ndarray:
        a   = np.maximum(gauss[:, :, 0:1], 0.0)
        mu  = gauss[:, :, 1:2]
        sig = gauss[:, :, 2:3]

        out = GaussianReconstructor._single(a, mu, sig, x).sum(axis=1)

        return out.astype(np.float32)

    @staticmethod
    def components(params: np.ndarray, x_axis: np.ndarray, n_gaussians: int) -> List[np.ndarray]:
        out = []

        for k in range(n_gaussians):
            a   = float(params[3 * k])
            mu  = float(params[3 * k + 1])
            sig = float(params[3 * k + 2])
            out.append(GaussianReconstructor._single(a, mu, sig, x_axis))

        return out


class GaussianCurve:
    @staticmethod
    def reconstruct(params: torch.Tensor, x_axis: torch.Tensor, ppg: int = 3) -> torch.Tensor:
        B, C, H, W = params.shape

        if C % ppg != 0:
            raise ValueError(f"Gaussian param channels ({C}) must be divisible by {ppg}")

        n_gaussians = C // ppg
        p           = params.reshape(B, n_gaussians, ppg, H, W)

        a   = p[:, :, 0, :, :]
        mu  = p[:, :, 1, :, :]
        sig = p[:, :, 2, :, :]
        x   = x_axis.reshape(1, -1, 1, 1)

        curves = torch.zeros((B, x.shape[1], H, W), dtype=params.dtype, device=params.device)

        for g in range(n_gaussians):
            a_g   = a[:, g:g+1, :, :]
            mu_g  = mu[:, g:g+1, :, :]
            sig_g = sig[:, g:g+1, :, :].clamp(min=GaussianMixture.SIGMA_FLOOR)

            sig2_g     = sig_g ** 2
            exponent_g = (-((x - mu_g) ** 2) / (2.0 * sig2_g)).clamp(GaussianMixture.EXPON_FLOOR, GaussianMixture.EXPON_CEIL)
            curves     = curves + a_g * torch.exp(exponent_g)

        return curves


class GaussianHead:
    @staticmethod
    def total_channels(ppg: int, n_gaussians: int) -> int:
        return ppg * n_gaussians


class GaussianClamp:
    @staticmethod
    def apply(
        params      : torch.Tensor,
        x_axis      : torch.Tensor,
        amp_max     : float,
        ppg         : int   = 3,
        leaky_slope : float = 0.0,
    ) -> torch.Tensor:
        x_min   = x_axis.min()
        x_max   = x_axis.max()
        x_step  = (x_max - x_min) / (x_axis.shape[0] - 1)
        x_range = x_max - x_min
        n_gauss = params.shape[1] // ppg
        slices  = []

        def _clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
            if leaky_slope > 0.0:
                clamped = x.clamp(lo, hi)
                return clamped + leaky_slope * (x - clamped)
            return x.clamp(lo, hi)

        for _g in range(n_gauss):
            _b = _g * ppg
            slices.append(_clamp(params[:, _b + 0:_b + 1], 0.0,          amp_max      ))
            slices.append(_clamp(params[:, _b + 1:_b + 2], x_min,        x_max        ))
            slices.append(_clamp(params[:, _b + 2:_b + 3], x_step * 0.5, x_range * 0.5))

        return torch.cat(slices, dim=1)
