from __future__ import annotations

from typing import List

import numpy as np


class GaussianReconstructor:
    @staticmethod
    def _single(a: np.ndarray, mu: np.ndarray, sig: np.ndarray, x: np.ndarray) -> np.ndarray:
        return a * np.exp(-((x - mu) ** 2) / (2.0 * sig * sig + 1e-8))

    @staticmethod
    def reconstruct_batch(gauss: np.ndarray, x: np.ndarray) -> np.ndarray:
        a   = np.maximum(gauss[:, :, 0:1], 0.0)
        mu  =            gauss[:, :, 1:2]
        sig =            gauss[:, :, 2:3]

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
