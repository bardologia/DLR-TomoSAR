from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib     import Path
from typing      import Tuple

import numpy as np

from configuration.processing_config import CropRegion
from tools.regions                   import SplitRegions
from tools.logger                    import Logger


class Layout:
    def __init__(self, run_directory: Path, logger: Logger, parameters_path: Path) -> None:
        self.run_directory    = Path(run_directory)
        self.logger           = logger
        self.data_directory   = self.run_directory / "data"
        self.parameters_path  = Path(parameters_path)

        layout_path = self.data_directory / "dataset.json"
        with open(layout_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.global_crop    : CropRegion = CropRegion(*payload["global_crop"])
        self.dataset_type   : str        = payload["dataset_type"]
        self.tomogram_tag   : str        = payload["tomogram_tag"]
        self.parameter_tag  : str        = payload["parameter_tag"]
        self.artifacts      : dict       = payload["artifacts"]

        self.logger.section("[Layout Loaded]")
        self.logger.kv_table({
            "Run Directory":   self.run_directory,
            "Global Crop":     self.global_crop.as_tuple(),
            "Azimuth (lines)": self.global_crop.azimuth_size,
            "Range (samples)": self.global_crop.range_size,
            "Tomogram Tag":    self.tomogram_tag,
            "Parameters":      self.parameters_path,
        })

    def artifact_path(self, artifact_key: str) -> Path:
        if artifact_key == "parameters":
            return self.parameters_path

        return self.data_directory / self.artifacts[artifact_key]


class Cropper:
    def __init__(self, layout: Layout, split_regions: SplitRegions, logger: Logger) -> None:
        self.layout        = layout
        self.split_regions = split_regions
        self.logger        = logger

        self.logger.section("[Cropper Initialized]")
        rows = []
        for name, regions in split_regions.region_lists():
            for index, region in enumerate(regions):
                label = name if len(regions) == 1 else f"{name}[{index}]"
                rows.append({"Split": label, "Crop": str(region.as_tuple()), "Azimuth (lines)": region.azimuth_size, "Range (samples)": region.range_size})
        self.logger.metrics_table(rows, ["Split", "Crop", "Azimuth (lines)", "Range (samples)"])

    def to_local_slices(self, region: CropRegion) -> Tuple[slice, slice]:
        return region.local_slices(self.layout.global_crop)

    def load_split(self, region: CropRegion) -> dict[str, np.ndarray]:
        az_slice, rg_slice = self.to_local_slices(region)

        primary_reduced        = np.load(str(self.layout.artifact_path("primary_reduced")),        mmap_mode="r", allow_pickle=False)
        secondaries_reduced    = np.load(str(self.layout.artifact_path("secondaries_reduced")),    mmap_mode="r", allow_pickle=False)
        interferograms_reduced = np.load(str(self.layout.artifact_path("interferograms_reduced")), mmap_mode="r", allow_pickle=False)
        parameters_reduced     = np.load(str(self.layout.artifact_path("parameters")),             mmap_mode="r", allow_pickle=False)
        dem_reduced            = np.load(str(self.layout.artifact_path("dem_reduced")),            mmap_mode="r", allow_pickle=False)

        primary_view        = primary_reduced        [..., az_slice, rg_slice]
        secondaries_view    = secondaries_reduced    [..., az_slice, rg_slice]
        interferograms_view = interferograms_reduced [..., az_slice, rg_slice]

        n_secondaries    = secondaries_view.shape[0]
        n_interferograms = interferograms_view.shape[0]
        n_passes         = 1 + n_secondaries + n_interferograms
        az_size          = primary_view.shape[-2]
        rg_size          = primary_view.shape[-1]

        inputs_split     = np.empty((n_passes, az_size, rg_size), dtype=primary_reduced.dtype)
        inputs_split[0]                                  = primary_view
        inputs_split[1:1 + n_secondaries]                = secondaries_view
        inputs_split[1 + n_secondaries:]                 = interferograms_view

        parameters_split = np.ascontiguousarray(parameters_reduced [..., az_slice, rg_slice])
        dem_split        = np.ascontiguousarray(dem_reduced        [az_slice, rg_slice].astype(np.float32))

        self.logger.section("[Crop Loaded]")
        self.logger.kv_table({
            "Primary"          : primary_view.shape,
            "Secondaries"      : secondaries_view.shape,
            "Interferograms"   : interferograms_view.shape,
            "Inputs (stacked)" : f"{inputs_split.shape}  ({inputs_split.nbytes/1e9:.2f} GB)  [1 primary + {n_secondaries} secondaries + {n_interferograms} interferograms]",
            "DEM reduced"      : f"{dem_split.shape}  ({dem_split.nbytes/1e6:.2f} MB)",
            "Parameters"       : f"{parameters_split.shape}  ({parameters_split.nbytes/1e9:.2f} GB)",
        })

        return {
            "inputs"           : inputs_split,
            "dem"              : dem_split,
            "parameters"       : parameters_split,
            "n_secondaries"    : n_secondaries,
            "n_interferograms" : n_interferograms,
        }


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
