from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class GaussianConfig:
    """Configuration for the Gaussian curve reconstruction and noise head."""

    params_per_gaussian : int = 3
    noise_param_name    : str = "log_noise_var"

    def make_param_names(self, n_gaussians: int | None = None) -> list[str]:
        """Generate parameter names for K superimposed Gaussians."""
        k = n_gaussians
        return [
            f"{prefix}{i + 1}"
            for i in range(k)
            for prefix in ("a", "mu", "sig")
        ]

    @property
    def default_param_names(self) -> list[str]:
        """Parameter names for the default number of Gaussians."""
        return self.make_param_names(self.n_default_gaussians)


@dataclass
class ChannelNorm:
    """Min-max bounds for normalising one channel to [0, 1]."""
    channel: int
    vmin:    float
    vmax:    float

    def normalize(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.vmin) / (self.vmax - self.vmin)

    def denormalize(self, arr: np.ndarray) -> np.ndarray:
        return arr * (self.vmax - self.vmin) + self.vmin

    @staticmethod
    def apply(data: np.ndarray, norms: Sequence[ChannelNorm]) -> np.ndarray:
        """In-place min-max normalisation on selected channels. data: (N, C, …)."""
        for n in norms:
            data[:, n.channel] = n.normalize(data[:, n.channel])
        return data


@dataclass
class SplitConfig:
    """Train/val/test split ratios."""
    train: float = 0.70
    val:   float = 0.15
    test:  float = 0.15

    def __post_init__(self):
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")

    def to_sizes(self, n: int) -> tuple[int, int, int]:
        n_test  = int(n * self.test)
        n_val   = int(n * self.val)
        n_train = n - n_val - n_test
        return n_train, n_val, n_test
