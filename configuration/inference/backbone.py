from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List

from configuration.inference.general import InferenceConfig


@dataclass
class InferenceEntryConfig:
    logs_dir   : Path      = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/backbone")
    run_filter : List[str] = field(default_factory=list)
    gpu        : int       = 0

    inference : InferenceConfig = field(default_factory=lambda: InferenceConfig(
        run_directory = Path("."),
        save_cubes    = True,
        cpu_workers   = 16,
        gif_axes      = ["elevation", "range", "azimuth"],
    ))
