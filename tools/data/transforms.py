from __future__ import annotations

import math

import numpy as np
import torch


class Log1pTransform:
    CEIL = 1000.0

    @staticmethod
    def compress(x):
        if isinstance(x, torch.Tensor):
            return torch.log1p(torch.clamp(x, min=0.0))

        return np.log1p(np.maximum(x, 0.0))

    @staticmethod
    def decompress(x, floor: float = 0.0, ceil: float = CEIL, enabled: bool = True, leaky_slope: float = 0.0):
        lo = math.log1p(floor)
        hi = math.log1p(ceil)

        if isinstance(x, torch.Tensor):
            bounded = torch.clamp(x, min=lo, max=hi) if enabled else x
            if enabled and leaky_slope > 0.0:
                bounded = bounded + leaky_slope * torch.clamp(x - lo, max=0.0)
            return torch.expm1(bounded)

        bounded = np.clip(x, lo, hi) if enabled else x
        if enabled and leaky_slope > 0.0:
            bounded = bounded + leaky_slope * np.minimum(x - lo, 0.0)
        return np.expm1(bounded)
