from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from pipelines.dataset_creation_pipeline.patch import GridInfo

class CubeStitcher:
    def __init__(
        self,
        grid           : GridInfo,
        n_channels     : int,
        window_kind    : str           = "hann",
        dtype          : str           = "float32",
        memmap_path    : Optional[str] = None,
    ) -> None:
        self.grid       = grid
        self.n_channels = int(n_channels)
        self.dtype      = np.dtype(dtype)
        self.window     = CubeStitcher.make_patch_window(grid.patch_size, kind=window_kind)

        H_pad, W_pad = grid.padded_size
        shape_pad    = (self.n_channels, H_pad, W_pad)

        if memmap_path is not None:
            self._accum = np.lib.format.open_memmap(memmap_path, mode="w+", dtype=self.dtype, shape=shape_pad)
            self._accum[...] = 0
        else:
            self._accum = np.zeros(shape_pad, dtype=self.dtype)

        self._weight = np.zeros((H_pad, W_pad), dtype=np.float32)

    @staticmethod
    def make_patch_window(patch_size: Tuple[int, int], kind: str = "hann") -> np.ndarray:
        ph, pw = patch_size
        if kind == "uniform":
            return np.ones((ph, pw), dtype=np.float32)

        if kind == "hann":
            wv = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(ph) + 0.5) / ph)
            wh = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(pw) + 0.5) / pw)

        elif kind == "triangular":
            wv = 1.0 - np.abs((np.arange(ph) + 0.5) / ph * 2.0 - 1.0)
            wh = 1.0 - np.abs((np.arange(pw) + 0.5) / pw * 2.0 - 1.0)

        else:
            raise ValueError(f"Unknown window kind: {kind!r}")

        wv = np.clip(wv, 1e-3, None).astype(np.float32)
        wh = np.clip(wh, 1e-3, None).astype(np.float32)

        return np.outer(wv, wh)

    @property
    def number_of_patches(self) -> int:
        return self.grid.number_of_patches

    def add(self, idx: int, patch: np.ndarray) -> None:
        ph, pw = self.grid.patch_size
        iv, ih = divmod(idx, self.grid.n_h)
        v0     = iv * self.grid.stride
        h0     = ih * self.grid.stride
        w      = self.window
        
        self._accum[:, v0:v0 + ph, h0:h0 + pw] += (patch * w[None, :, :]).astype(self.dtype, copy=False)
        self._weight[v0:v0 + ph, h0:h0 + pw]   += w

    def add_batch(self, indices: np.ndarray, batch_patches: np.ndarray) -> None:
        for b, idx in enumerate(indices):
            self.add(int(idx), batch_patches[b])

    def finalize(self) -> np.ndarray:
        H, W         = self.grid.spatial_size
        pad_t, pad_l = self.grid.pad_top, self.grid.pad_left

        weight_safe = np.where(self._weight > 0, self._weight, 1.0)
        cube        = self._accum / weight_safe[None, :, :]
        cube        = cube[:, pad_t:pad_t + H, pad_l:pad_l + W]
        
        return np.ascontiguousarray(cube.astype(self.dtype, copy=False))
