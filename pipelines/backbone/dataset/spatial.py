from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib     import Path
from typing      import Optional, Tuple

import numpy as np

from tools.data.io           import FileIO
from tools.data.regions      import CropRegion, SplitRegions
from tools.monitoring.logger import Logger
from tools.baselines         import SecondarySelection


class Layout:
    def __init__(self, run_directory: Path, logger: Logger, parameters_path: Path) -> None:
        self.run_directory   = Path(run_directory)
        self.logger          = logger
        self.data_directory  = self.run_directory / "data"
        self.parameters_path = Path(parameters_path)

        layout_path = self.data_directory / "dataset.json"
        payload     = FileIO.load_json(layout_path)

        self.global_crop  : CropRegion  = CropRegion(*payload["global_crop"])
        self.tomogram_tag : str         = payload["tomogram_tag"]
        self.artifacts    : dict        = payload["artifacts"]
        self.pass_labels  : list | None = payload["pass_labels"]

        self.logger.section("[Layout Loaded]")
        self.logger.kv_table({
            "Run Directory":   self.run_directory,
            "Global Crop":     self.global_crop.as_tuple(),
            "Azimuth (lines)": self.global_crop.azimuth_size,
            "Range (samples)": self.global_crop.range_size,
            "Tomogram Tag":    self.tomogram_tag,
            "Pass Labels":     ", ".join(self.pass_labels) if self.pass_labels else "unavailable",
            "Parameters":      self.parameters_path,
        })

    def artifact_path(self, artifact_key: str) -> Path:
        if artifact_key == "parameters":
            return self.parameters_path

        return self.data_directory / self.artifacts[artifact_key]

    def secondary_indices(self, secondary_labels) -> list | None:
        if secondary_labels is None:
            return None

        if not self.pass_labels:
            raise ValueError("Dataset records no pass labels in dataset.json; baseline extraction must succeed during pre-processing before secondaries can be selected by label.")

        return SecondarySelection.indices(self.pass_labels, secondary_labels)


class Cropper:
    def __init__(self, layout: Layout, split_regions: SplitRegions, logger: Logger, secondary_labels=None) -> None:
        self.layout            = layout
        self.split_regions     = split_regions
        self.logger            = logger
        self.secondary_indices = layout.secondary_indices(secondary_labels)
        self.secondary_labels  = self._resolve_labels(secondary_labels)

        self.logger.section("[Cropper Initialized]")
        self.logger.subsection(f"Secondary selection : {', '.join(self.secondary_labels) if self.secondary_labels else 'all passes'}")
        rows = []
        for name, regions in split_regions.region_lists():
            for index, region in enumerate(regions):
                label = name if len(regions) == 1 else f"{name}[{index}]"
                rows.append({"Split": label, "Crop": str(region.as_tuple()), "Azimuth (lines)": region.azimuth_size, "Range (samples)": region.range_size})
        self.logger.metrics_table(rows, ["Split", "Crop", "Azimuth (lines)", "Range (samples)"])

    def _resolve_labels(self, secondary_labels) -> list | None:
        if self.secondary_indices is None:
            return None
        secondaries = self.layout.pass_labels[1:]
        return [secondaries[index] for index in self.secondary_indices]

    def to_local_slices(self, region: CropRegion) -> Tuple[slice, slice]:
        return region.local_slices(self.layout.global_crop)

    def _select_channels(self, array: np.ndarray, az_slice: slice, rg_slice: slice) -> np.ndarray:
        if self.secondary_indices is None:
            return array[..., az_slice, rg_slice]

        return np.stack([array[index, az_slice, rg_slice] for index in self.secondary_indices])

    def _stack_inputs(self, az_slice: slice, rg_slice: slice) -> Tuple[np.ndarray, int, int]:
        primary        = np.load(str(self.layout.artifact_path("primary")),        mmap_mode="r", allow_pickle=False)
        secondaries    = np.load(str(self.layout.artifact_path("secondaries")),    mmap_mode="r", allow_pickle=False)
        interferograms = np.load(str(self.layout.artifact_path("interferograms")), mmap_mode="r", allow_pickle=False)

        primary_view        = primary[..., az_slice, rg_slice]
        secondaries_view    = self._select_channels(secondaries,    az_slice, rg_slice)
        interferograms_view = self._select_channels(interferograms, az_slice, rg_slice)

        n_secondaries    = secondaries_view.shape[0]
        n_interferograms = interferograms_view.shape[0]
        n_passes         = 1 + n_secondaries + n_interferograms
        az_size          = primary_view.shape[-2]
        rg_size          = primary_view.shape[-1]

        inputs_split                      = np.empty((n_passes, az_size, rg_size), dtype=primary.dtype)
        inputs_split[0]                   = primary_view
        inputs_split[1:1 + n_secondaries] = secondaries_view
        inputs_split[1 + n_secondaries:]  = interferograms_view

        return inputs_split, n_secondaries, n_interferograms

    def _load_targets(self, az_slice: slice, rg_slice: slice) -> Tuple[np.ndarray, np.ndarray]:
        parameters = np.load(str(self.layout.artifact_path("parameters")), mmap_mode="r", allow_pickle=False)
        dem        = np.load(str(self.layout.artifact_path("dem_full")),   mmap_mode="r", allow_pickle=False)

        parameters_split = np.ascontiguousarray(parameters[..., az_slice, rg_slice])
        dem_split        = np.ascontiguousarray(dem[az_slice, rg_slice].astype(np.float32))

        return parameters_split, dem_split

    def _load_tomogram(self, az_slice: slice, rg_slice: slice) -> np.ndarray:
        tomogram = np.load(str(self.layout.artifact_path("tomogram_full")), mmap_mode="r", allow_pickle=False)
        return np.ascontiguousarray(np.abs(tomogram[:, az_slice, rg_slice]).astype(np.float32))

    def _log_crop(self, inputs_split: np.ndarray, dem_split: np.ndarray, parameters_split: np.ndarray, tomogram_split: Optional[np.ndarray], n_secondaries: int, n_interferograms: int) -> None:
        self.logger.section("[Crop Loaded]")
        self.logger.kv_table({
            "Primary"          : inputs_split[0].shape,
            "Secondaries"      : inputs_split[1:1 + n_secondaries].shape,
            "Interferograms"   : inputs_split[1 + n_secondaries:].shape,
            "Selection"        : ", ".join(self.secondary_labels) if self.secondary_labels else "all passes",
            "Inputs (stacked)" : f"{inputs_split.shape}  ({inputs_split.nbytes/1e9:.2f} GB)  [1 primary + {n_secondaries} secondaries + {n_interferograms} interferograms]",
            "DEM"              : f"{dem_split.shape}  ({dem_split.nbytes/1e6:.2f} MB)",
            "Parameters"       : f"{parameters_split.shape}  ({parameters_split.nbytes/1e9:.2f} GB)",
            "Full tomogram"    : f"{tomogram_split.shape}  ({tomogram_split.nbytes/1e9:.2f} GB)" if tomogram_split is not None else "not loaded",
        })

    def load_split(self, region: CropRegion, load_tomogram: bool = False) -> dict[str, np.ndarray]:
        az_slice, rg_slice = self.to_local_slices(region)

        inputs_split, n_secondaries, n_interferograms = self._stack_inputs(az_slice, rg_slice)
        parameters_split, dem_split                   = self._load_targets(az_slice, rg_slice)
        tomogram_split                                = self._load_tomogram(az_slice, rg_slice) if load_tomogram else None

        self._log_crop(inputs_split, dem_split, parameters_split, tomogram_split, n_secondaries, n_interferograms)

        return {
            "inputs"           : inputs_split,
            "dem"              : dem_split,
            "parameters"       : parameters_split,
            "tomogram"         : tomogram_split,
            "n_secondaries"    : n_secondaries,
            "n_interferograms" : n_interferograms,
            "secondary_labels" : self.secondary_labels,
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

                pt = max(0, -v0)
                pb = max(0, v1 - H)
                pl = max(0, -h0)
                pr = max(0, h1 - W)

                v0c = max(0, v0)
                v1c = min(H, v1)
                h0c = max(0, h0)
                h1c = min(W, h1)

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
            pad_width = [(0, 0)] * (sub.ndim - 2) + [(pt, pb), (pl, pr)]
            sub       = np.pad(sub, pad_width, mode=mode)

        return sub
