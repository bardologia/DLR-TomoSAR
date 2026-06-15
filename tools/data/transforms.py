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
    def decompress(x, ceil: float = CEIL):
        if isinstance(x, torch.Tensor):
            return torch.expm1(torch.clamp(x, min=0.0, max=ceil))

        return np.expm1(np.clip(x, 0.0, ceil))
