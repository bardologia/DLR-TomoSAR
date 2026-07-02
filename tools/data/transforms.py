from __future__ import annotations

import numpy as np
import torch


class Log1pTransform:
    CEIL = 80.0

    @staticmethod
    def compress(x):
        if isinstance(x, torch.Tensor):
            return torch.log1p(torch.clamp(x, min=0.0))

        return np.log1p(np.maximum(x, 0.0))

    @staticmethod
    def decompress(x, floor: float = 0.0, ceil: float = CEIL, enabled: bool = True, leaky_slope: float = 0.0):
        if isinstance(x, torch.Tensor):
            bounded = torch.clamp(x, min=floor, max=ceil) if enabled else x
            if enabled and leaky_slope > 0.0:
                bounded = bounded + leaky_slope * torch.clamp(x - floor, max=0.0)
            return torch.expm1(bounded)

        bounded = np.clip(x, floor, ceil) if enabled else x
        if enabled and leaky_slope > 0.0:
            bounded = bounded + leaky_slope * np.minimum(x - floor, 0.0)
        return np.expm1(bounded)
