from __future__ import annotations

from dataclasses import dataclass

from tools.crop_region import CropRegion


@dataclass
class SplitRegions:
    train : CropRegion | list[CropRegion]
    val   : CropRegion | list[CropRegion]
    test  : CropRegion | list[CropRegion]

    @staticmethod
    def as_list(value) -> list[CropRegion]:
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def items(self):
        return [("train", self.train), ("val", self.val), ("test", self.test)]

    def region_lists(self):
        return [(name, self.as_list(value)) for name, value in self.items()]

    def regions(self, split_name: str) -> list[CropRegion]:
        return self.as_list(dict(self.items())[split_name])

    def bounding_global_crop(self) -> CropRegion:
        regions = [region for _, region_list in self.region_lists() for region in region_list]

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
