from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from tools.split_regions                      import SplitRegions
from configuration.processing_config          import CropRegion
from pipelines.dataset_pipeline.layout        import Layout
from tools.logger                             import Logger


class Cropper:
    def __init__(self, layout: Layout, split_regions: SplitRegions, logger: Logger) -> None:
        self.layout        = layout
        self.split_regions = split_regions
        self.logger        = logger
  
        self.logger.section("[Cropper Initialized]")
        rows = [
            {"Split": name, "Crop": str(region.as_tuple()), "Azimuth (lines)": region.azimuth_size, "Range (samples)": region.range_size}
            for name, region in split_regions.items()
        ]
        self.logger.metrics_table(rows, ["Split", "Crop", "Azimuth (lines)", "Range (samples)"])

    def to_local_slices(self, region: CropRegion) -> Tuple[slice, slice]:
        global_crop = self.layout.global_crop
        az_slice    = slice(region.azimuth_start - global_crop.azimuth_start, region.azimuth_end   - global_crop.azimuth_start)
        rg_slice    = slice(region.range_start   - global_crop.range_start, region.range_end       - global_crop.range_start)
        
        return az_slice, rg_slice

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
            "inputs"     : inputs_split,
            "dem"        : dem_split,
            "parameters" : parameters_split,
        }
