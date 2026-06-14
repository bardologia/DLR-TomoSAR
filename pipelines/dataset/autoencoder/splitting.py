from __future__ import annotations

import numpy as np

from pipelines.dataset.backbone.spatial import Layout
from tools.data.regions                 import SplitRegions
from tools.monitoring.logger            import Logger


class ParameterCropper:
    def __init__(self, layout: Layout, split_regions: SplitRegions, logger: Logger) -> None:
        self.layout        = layout
        self.split_regions = split_regions
        self.logger        = logger

        self.logger.section("[ParameterCropper Initialized]")
        rows = []
        for name, regions in split_regions.region_lists():
            for index, region in enumerate(regions):
                label = name if len(regions) == 1 else f"{name}[{index}]"
                rows.append({"Split": label, "Crop": str(region.as_tuple()), "Azimuth (lines)": region.azimuth_size, "Range (samples)": region.range_size})
        self.logger.metrics_table(rows, ["Split", "Crop", "Azimuth (lines)", "Range (samples)"])

    def profile_length(self) -> int:
        tomo_path = self.layout.artifact_path("tomogram_full")
        tomo_mmap = np.load(str(tomo_path), mmap_mode="r", allow_pickle=False)

        return int(tomo_mmap.shape[0])

    def load_split(self, split_name: str) -> list[np.ndarray]:
        regions    = self.split_regions.regions(split_name)
        parameters = np.load(str(self.layout.artifact_path("parameters")), mmap_mode="r", allow_pickle=False)

        arrays = []
        for region in regions:
            az_slice, rg_slice = region.local_slices(self.layout.global_crop)
            arrays.append(np.ascontiguousarray(parameters[..., az_slice, rg_slice]))

        total_pixels = sum(int(a.shape[-2] * a.shape[-1]) for a in arrays)

        self.logger.section(f"[Parameters Loaded: {split_name}]")
        self.logger.kv_table({
            "Regions":      len(arrays),
            "Channels":     int(arrays[0].shape[0]),
            "Total pixels": f"{total_pixels:,}",
        })

        return arrays
