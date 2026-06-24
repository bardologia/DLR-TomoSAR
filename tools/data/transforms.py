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
    def decompress(x, floor: float = 0.0, ceil: float = CEIL, enabled: bool = True):
        if isinstance(x, torch.Tensor):
            bounded = torch.clamp(x, min=floor, max=ceil) if enabled else x
            return torch.expm1(bounded)

        bounded = np.clip(x, floor, ceil) if enabled else x
        return np.expm1(bounded)
