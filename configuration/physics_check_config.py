from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from configuration.training_config import GeometryConfig


@dataclass
class PhysicsCheckEntryConfig:
    dataset_path      : Path         = Path("/ste/rnd/User/vice_vi/Dataset")
    height_range      : tuple | None = None
    secondary_labels  : tuple | None = ("FL01_PS04", "FL01_PS06", "FL01_PS08", "FL01_PS26")

    fit_k_max         : int          = 5
    output_prefix     : str          = "params"
    output_suffix     : str | None   = None

    n_pixels          : int          = 20000
    seed              : int          = 0
    device            : str          = "cuda"

    physics_floor     : float        = 1e-3
    capon_loading     : float        = 1e-2
    moments_weights   : tuple        = (1.0, 1.0, 1.0)

    geometry          : GeometryConfig = field(default_factory=GeometryConfig)
