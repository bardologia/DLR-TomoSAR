from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class CropRegion:
    azimuth_start : int
    azimuth_end   : int
    range_start   : int
    range_end     : int

    def __post_init__(self) -> None:
        if self.azimuth_start >= self.azimuth_end:
            raise ValueError(f"azimuth_start ({self.azimuth_start}) must be < azimuth_end ({self.azimuth_end})")
        if self.range_start   >= self.range_end:
            raise ValueError(f"range_start ({self.range_start}) must be < range_end ({self.range_end})")

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.azimuth_start, self.azimuth_end, self.range_start, self.range_end)

    def as_identifier_string(self) -> str:
        return "a".join(str(value) for value in self.as_tuple())

    @property
    def azimuth_size(self) -> int:
        return self.azimuth_end - self.azimuth_start

    @property
    def range_size(self) -> int:
        return self.range_end - self.range_start
