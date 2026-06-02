from __future__ import annotations

import numpy as np
import torch

from configuration.norm_config            import ChannelStats
from pipelines.dataset_pipeline.stats      import Stats


class Normalizer:
    def __init__(self, stats: Stats) -> None:
        self.stats = stats

        self._vectors: dict[int, dict] = {}

    def _channel_vectors(self, stats: ChannelStats) -> dict:
        key = id(stats)
        if key in self._vectors:
            return self._vectors[key]

        loc       = np.asarray(stats.loc,   dtype=np.float32)
        scale     = np.asarray(stats.scale, dtype=np.float32)
        log1p     = np.asarray([strat.apply_log1p for strat in stats.strategies], dtype=bool)
        inv_scale = (1.0 / scale).astype(np.float32)

        vectors = {
            "loc"       : loc,
            "scale"     : scale,
            "inv_scale" : inv_scale,
            "log1p"     : log1p,
        }
        self._vectors[key] = vectors

        return vectors

    def _apply_normalization(self, tensor, stats: ChannelStats, inverse: bool):
        is_torch = isinstance(tensor, torch.Tensor)
        vectors  = self._channel_vectors(stats)

        shape    = (1, -1, 1, 1) if tensor.ndim == 4 else (-1, 1, 1)

        if is_torch:
            device    = tensor.device
            loc       = torch.as_tensor(vectors["loc"],       device=device).reshape(shape)
            scale     = torch.as_tensor(vectors["scale"],     device=device).reshape(shape)
            inv_scale = torch.as_tensor(vectors["inv_scale"], device=device).reshape(shape)
            log1p     = torch.as_tensor(vectors["log1p"],     device=device).reshape(shape)

            if not inverse:
                x   = torch.where(log1p, torch.log1p(torch.clamp(tensor, min=0.0)), tensor)
                out = (x - loc) * inv_scale
            else:
                x   = tensor * scale + loc
                out = torch.where(log1p, torch.expm1(x.clamp(max=15.0)), x)

            return out

        loc       = vectors["loc"].reshape(shape)
        scale     = vectors["scale"].reshape(shape)
        inv_scale = vectors["inv_scale"].reshape(shape)
        log1p     = vectors["log1p"].reshape(shape)

        if not inverse:
            x   = np.where(log1p, np.log1p(np.maximum(tensor, 0.0)), tensor)
            out = (x - loc) * inv_scale
        else:
            x   = tensor * scale + loc
            out = np.where(log1p, np.expm1(np.clip(x, a_min=None, a_max=15.0)), x)

        return np.ascontiguousarray(out, dtype=np.float32)

    def normalize_input(self, tensor: np.ndarray) -> np.ndarray:
        return self._apply_normalization(tensor, self.stats.input_stats, inverse=False)

    def normalize_output(self, tensor) -> np.ndarray:
        return self._apply_normalization(tensor, self.stats.output_stats, inverse=False)

    def denormalize_input(self, tensor):
        return self._apply_normalization(tensor, self.stats.input_stats, inverse=True)

    def denormalize_output(self, tensor):
        return self._apply_normalization(tensor, self.stats.output_stats, inverse=True)
