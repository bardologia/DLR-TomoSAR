from __future__ import annotations

import math
from dataclasses import dataclass
from typing      import Tuple
import numpy as np


@dataclass
class GridInfo:
    n_v                    : int
    n_h                    : int
    pad_top                : int
    pad_bot                : int
    pad_left               : int
    pad_right              : int
    patch_size             : Tuple[int, int]
    stride                 : int
    spatial_size           : Tuple[int, int]
    use_reflective_padding : bool = True

    @property
    def padding_vertical(self) -> int:
        return self.pad_top + self.pad_bot

    @property
    def padding_horizontal(self) -> int:
        return self.pad_left + self.pad_right

    @property
    def number_of_patches(self) -> int:
        return self.n_v * self.n_h

    @property
    def padded_size(self) -> Tuple[int, int]:
        H, W = self.spatial_size
        return (H + self.padding_vertical, W + self.padding_horizontal)

    def as_dict(self) -> dict:
        return {
            "n_v"                    : self.n_v,
            "n_h"                    : self.n_h,
            "pad_top"                : self.pad_top,
            "pad_bot"                : self.pad_bot,
            "pad_left"               : self.pad_left,
            "pad_right"              : self.pad_right,
            "patch_size"             : list(self.patch_size),
            "stride"                 : self.stride,
            "spatial_size"           : list(self.spatial_size),
            "use_reflective_padding" : self.use_reflective_padding,
            "number_of_patches"      : self.number_of_patches,
        }


class Patcher:
    def __init__(self, grid: GridInfo, patch_coords: list) -> None:
        self.grid          = grid
        self._patch_coords = patch_coords  

    @classmethod
    def build(cls, spatial_size : Tuple[int, int], patch_size : Tuple[int, int], stride : int, use_reflective_padding : bool = True) -> "Patcher":
        ph, pw = patch_size
        H, W   = spatial_size

        n_v = 1 if H <= ph else math.ceil((H - ph) / stride) + 1
        n_h = 1 if W <= pw else math.ceil((W - pw) / stride) + 1

        pad_v = (ph + (n_v - 1) * stride) - H
        pad_h = (pw + (n_h - 1) * stride) - W

        pad_top, pad_bot    = pad_v // 2, pad_v - pad_v // 2
        pad_left, pad_right = pad_h // 2, pad_h - pad_h // 2

        grid = GridInfo(
            n_v                    = n_v,
            n_h                    = n_h,
            pad_top                = pad_top,
            pad_bot                = pad_bot,
            pad_left               = pad_left,
            pad_right              = pad_right,
            patch_size             = (ph, pw),
            stride                 = stride,
            spatial_size           = (H, W),
            use_reflective_padding = use_reflective_padding,
        )

        mode         = "symmetric" if use_reflective_padding else "constant"
        patch_coords = []
        
        for iv in range(n_v):
            for ih in range(n_h):
                v0 = iv * stride - pad_top
                h0 = ih * stride - pad_left
                v1 = v0 + ph
                h1 = h0 + pw

                pt = max(0, -v0);  pb = max(0, v1 - H)
                pl = max(0, -h0);  pr = max(0, h1 - W)

                v0c, v1c = max(0, v0), min(H, v1)
                h0c, h1c = max(0, h0), min(W, h1)

                pw_spec = None
                if pt or pb or pl or pr:
                    pw_spec = (pt, pb, pl, pr, mode)

                patch_coords.append((v0c, v1c, h0c, h1c, pw_spec))

        return cls(grid, patch_coords)

    def extract(self, array: np.ndarray, idx: int) -> np.ndarray:
        v0c, v1c, h0c, h1c, pw_spec = self._patch_coords[idx]
        sub                         = np.ascontiguousarray(array[..., v0c:v1c, h0c:h1c])

        if pw_spec is not None:
            pt, pb, pl, pr, mode = pw_spec
            pad_width            = [(0, 0)] * (sub.ndim - 2) + [(pt, pb), (pl, pr)]
            sub                  = np.pad(sub, pad_width, mode=mode)

        return sub
