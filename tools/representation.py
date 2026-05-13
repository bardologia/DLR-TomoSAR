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

    def convert(self, data: np.ndarray) -> np.ndarray:
        n_samples, n_passes, h, w = data.shape
        cpp                       = _CHANNELS_PER_PASS[self.value]

        magnitude     = np.abs(data)
        log_magnitude = np.log1p(magnitude)
        safe_mag      = np.where(magnitude > 0, magnitude, 1.0)
        normalised_re = data.real / safe_mag
        normalised_im = data.imag / safe_mag
        phase         = np.angle(data)

        Re, Im = data.real, data.imag
        channels = {
            Representation.REAL_IMAG     : [Re, Im],
            Representation.MAG_REAL_IMAG : [log_magnitude, normalised_re, normalised_im],
            Representation.MAG_ANGLE     : [log_magnitude, phase],
            Representation.MAG_RI_ANGLE  : [log_magnitude, normalised_re, normalised_im, phase],
            Representation.ANGLE_ONLY    : [phase],
            Representation.MAG_ONLY      : [log_magnitude],
        }[self]

        out = np.empty((n_samples, n_passes * cpp, h, w), dtype=np.float32)
        for c, arr in enumerate(channels):
            out[:, c::cpp] = arr

        return np.nan_to_num(out, nan=0.0)
