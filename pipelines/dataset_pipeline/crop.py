from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from tools.split_regions                      import SplitRegions
from configuration.processing_config          import CropRegion
from pipelines.dataset_pipeline.metadata      import Layout
from tools.logger                             import Logger


class Cropper:
    def __init__(self, layout: Layout, split_regions: SplitRegions, logger: Logger) -> None:
        self.layout        = layout
        self.split_regions = split_regions
        self.logger        = logger
  
        self.logger.section("[Cropper Initialized]")
        for name, region in split_regions.items():
            self.logger.subsection(f"{name:<5} : {region.as_tuple()}  ({region.azimuth_size} x {region.range_size})")

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
        tomogram_full          = np.load(str(self.layout.artifact_path("tomogram_full")),          mmap_mode="r", allow_pickle=False)

        primary_split        = np.ascontiguousarray(primary_reduced        [..., az_slice, rg_slice])
        secondaries_split    = np.ascontiguousarray(secondaries_reduced    [..., az_slice, rg_slice])
        interferograms_split = np.ascontiguousarray(interferograms_reduced [..., az_slice, rg_slice])
        parameters_split     = np.ascontiguousarray(parameters_reduced     [..., az_slice, rg_slice])
        tomogram_split       = tomogram_full[..., az_slice, rg_slice]

        inputs_split = np.concatenate([primary_split[np.newaxis], secondaries_split, interferograms_split], axis=0)

        self.logger.section(f"[Crop Loaded]")
        self.logger.subsection(f" Primary         = {primary_split.shape}")
        self.logger.subsection(f" Secondaries     = {secondaries_split.shape}")
        self.logger.subsection(f" Interferograms  = {interferograms_split.shape}")
        self.logger.subsection(f" Inputs (stacked)= {inputs_split.shape}  (resident, {inputs_split.nbytes/1e9:.2f} GB)  [1 primary + {secondaries_split.shape[0]} secondaries + {interferograms_split.shape[0]} interferograms]")
        self.logger.subsection(f" Parameters      = {parameters_split.shape}  (resident, {parameters_split.nbytes/1e9:.2f} GB)")
        self.logger.subsection(f" Tomogram        = {tomogram_split.shape}  (mmap) \n")

        return {
            "primary"        : primary_split,
            "secondaries"    : secondaries_split,
            "interferograms" : interferograms_split,
            "inputs"         : inputs_split,
            "parameters"     : parameters_split,
            "tomogram"       : tomogram_split,
        }
