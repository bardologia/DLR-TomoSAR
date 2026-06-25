from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch

from configuration.normalization import ChannelStats
from tools.data.transforms             import Log1pTransform

from pipelines.backbone.dataset.stats import Stats


Array = Union[np.ndarray, torch.Tensor]


class Normalizer:
    def __init__(self, stats: Stats) -> None:
        self.stats = stats

        self._vectors: dict[int, dict] = {}

    def _channel_vectors(self, stats: ChannelStats) -> dict:
        key = id(stats)
        if key in self._vectors:
            return self._vectors[key]

        if stats.strategies is None:
            raise ValueError("ChannelStats is missing per-channel strategies; cannot build normalization vectors.")

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

    def _apply_normalization(self, tensor: Array, stats: ChannelStats, inverse: bool) -> Array:
        is_torch = isinstance(tensor, torch.Tensor)
        vectors  = self._channel_vectors(stats)
        clamp    = self.stats.clamp

        shape    = (1, -1, 1, 1) if tensor.ndim == 4 else (-1, 1, 1)

        if is_torch:
            device    = tensor.device
            loc       = torch.as_tensor(vectors["loc"],       device=device).reshape(shape)
            scale     = torch.as_tensor(vectors["scale"],     device=device).reshape(shape)
            inv_scale = torch.as_tensor(vectors["inv_scale"], device=device).reshape(shape)
            log1p     = torch.as_tensor(vectors["log1p"],     device=device).reshape(shape)

            if not inverse:
                x   = torch.where(log1p, Log1pTransform.compress(tensor), tensor)
                out = (x - loc) * inv_scale
            else:
                x   = tensor * scale + loc
                out = torch.where(log1p, Log1pTransform.decompress(x, clamp.floor, clamp.ceil, clamp.enabled), x)

            return out

        loc       = vectors["loc"].reshape(shape)
        scale     = vectors["scale"].reshape(shape)
        inv_scale = vectors["inv_scale"].reshape(shape)
        log1p     = vectors["log1p"].reshape(shape)

        if not inverse:
            x   = np.where(log1p, Log1pTransform.compress(tensor), tensor)
            out = (x - loc) * inv_scale
        else:
            x   = tensor * scale + loc
            out = np.where(log1p, Log1pTransform.decompress(x, clamp.floor, clamp.ceil, clamp.enabled), x)

        return np.ascontiguousarray(out, dtype=np.float32)

    @staticmethod
    def _require(stats: Optional[ChannelStats], role: str) -> ChannelStats:
        if stats is None:
            raise ValueError(f"Normalizer has no {role} stats; cannot (de)normalize {role}.")

        return stats

    def normalize_input(self, tensor: Array) -> Array:
        return self._apply_normalization(tensor, self._require(self.stats.input_stats, "input"), inverse=False)

    def normalize_output(self, tensor: Array) -> Array:
        return self._apply_normalization(tensor, self._require(self.stats.output_stats, "output"), inverse=False)

    def denormalize_input(self, tensor: Array) -> Array:
        return self._apply_normalization(tensor, self._require(self.stats.input_stats, "input"), inverse=True)

    def denormalize_output(self, tensor: Array) -> Array:
        return self._apply_normalization(tensor, self._require(self.stats.output_stats, "output"), inverse=True)
