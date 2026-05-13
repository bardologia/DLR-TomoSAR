from __future__ import annotations

from dataclasses import dataclass

from tools.crop_region import CropRegion


@dataclass
class SplitRegions:
    train : CropRegion
    val   : CropRegion
    test  : CropRegion

    def items(self):
        return [("train", self.train), ("val", self.val), ("test", self.test)]

    def bounding_global_crop(self) -> CropRegion:
        regions = [self.train, self.val, self.test]
        return CropRegion(
            azimuth_start = min(r.azimuth_start for r in regions),
            azimuth_end   = max(r.azimuth_end   for r in regions),
            range_start   = min(r.range_start   for r in regions),
            range_end     = max(r.range_end     for r in regions),
        )

    @classmethod
    def from_ratios(cls, global_crop: CropRegion, train_ratio: float = 0.70, val_ratio: float = 0.15) -> "SplitRegions":
        total_az  = global_crop.azimuth_end - global_crop.azimuth_start
        train_end = global_crop.azimuth_start + int(total_az * train_ratio)
        val_end   = train_end + int(total_az * val_ratio)

        return cls(
            train = CropRegion(global_crop.azimuth_start, train_end,               global_crop.range_start, global_crop.range_end),
            val   = CropRegion(train_end,                 val_end,                 global_crop.range_start, global_crop.range_end),
            test  = CropRegion(val_end,                   global_crop.azimuth_end, global_crop.range_start, global_crop.range_end),
        )
