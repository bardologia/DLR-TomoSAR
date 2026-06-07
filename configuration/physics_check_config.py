from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from configuration.training_config import GeometryConfig


@dataclass
class PhysicsCheckEntryConfig:
    dataset_path      : Path         = Path("/ste/rnd/User/vice_vi/Dataset")
    tomogram_filename : str          = "tomogram_full_1000a16000a500a4000_1_Xtomo_id2X.npy"
    height_range      : tuple | None = None
    secondary_labels  : tuple | None = ("PS04", "PS06", "PS08", "PS26")

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
