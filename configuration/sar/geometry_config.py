from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GeometryConfig:
    wavelength         : float = 0.23
    slant_range        : float = 5000.0
    look_angle_deg     : float = 45.0
    baselines          : tuple = (0.0, 11.25, 22.5, 33.75, 45.0, 56.25, 67.5, 78.75, 90.0)
    kz_values          : tuple = ()
    baselines_source   : str   = "auto"
    baseline_component : str   = "perpendicular"
    baselines_origin   : str   = "config"
    height_axis_convention : str = "height"
