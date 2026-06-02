from __future__ import annotations

import numpy as np


class GaussianMixture:
    SIGMA_FLOOR  = 1e-6
    EXPON_FLOOR  = -100.0
    EXPON_CEIL   = 0.0

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

            expon  = np.clip(-((height_axis - mu) ** 2) / sig_sq, cls.EXPON_FLOOR, cls.EXPON_CEIL)
            comp   = amp * np.exp(expon)
            components.append(comp)
            total += comp

        return total, components
