from __future__ import annotations

from enum import Enum
import numpy as np


_CHANNELS_PER_PASS: dict[str, int] = {
    "real_imag"     : 2,
    "mag_real_imag" : 3,
    "mag_angle"     : 2,
    "mag_ri_angle"  : 4,
    "angle_only"    : 1,
    "mag_only"      : 1,
}


class Representation(Enum):
    REAL_IMAG     = "real_imag"
    MAG_REAL_IMAG = "mag_real_imag"
    MAG_ANGLE     = "mag_angle"
    MAG_RI_ANGLE  = "mag_ri_angle"
    ANGLE_ONLY    = "angle_only"
    MAG_ONLY      = "mag_only"

    @property
    def channels_per_pass(self) -> int:
        return _CHANNELS_PER_PASS[self.value]

    @property
    def slot_kinds(self) -> list[str]:
        return {
            Representation.REAL_IMAG     : ["raw_re_im",  "raw_re_im"],
            Representation.MAG_REAL_IMAG : ["mag",    "norm_re_im", "norm_re_im"],
            Representation.MAG_ANGLE     : ["mag",    "phase"],
            Representation.MAG_RI_ANGLE  : ["mag",    "norm_re_im", "norm_re_im", "phase"],
            Representation.ANGLE_ONLY    : ["phase"],
            Representation.MAG_ONLY      : ["mag"],
        }[self]

    def convert(self, data: np.ndarray) -> np.ndarray:
        n_samples, n_passes, h, w = data.shape
        cpp                       = _CHANNELS_PER_PASS[self.value]

        if self is Representation.MAG_ONLY:
            channels = [np.abs(data)]

        elif self is Representation.ANGLE_ONLY:
            channels = [np.angle(data)]

        elif self is Representation.REAL_IMAG:
            channels = [data.real, data.imag]

        elif self is Representation.MAG_ANGLE:
            channels = [np.abs(data), np.angle(data)]

        elif self is Representation.MAG_REAL_IMAG:
            magnitude = np.abs(data)
            safe_mag  = np.where(magnitude > 0, magnitude, 1.0)
            channels  = [magnitude, data.real / safe_mag, data.imag / safe_mag]

        elif self is Representation.MAG_RI_ANGLE:
            magnitude = np.abs(data)
            safe_mag  = np.where(magnitude > 0, magnitude, 1.0)
            channels  = [
                magnitude,
                data.real / safe_mag,
                data.imag / safe_mag,
                np.angle(data),
            ]
        else:
            raise ValueError(f"Unsupported representation: {self}")

        out = np.empty((n_samples, n_passes * cpp, h, w), dtype=np.float32)
        for c, arr in enumerate(channels):
            out[:, c::cpp] = arr

        return out
