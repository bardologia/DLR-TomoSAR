from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List

from configuration.inference.general             import InferenceConfig
from configuration.inference.image_autoencoder   import ImageAeInferenceConfig
from configuration.inference.profile_autoencoder import ProfileAeInferenceConfig

_RUNS_ROOT = "/ste/rnd/User/vice_vi/DLR-TomoSAR/runs"


@dataclass
class InferenceEntryConfig:
    logs_dirs : List[str] = field(default_factory=lambda: [
        f"{_RUNS_ROOT}/backbone",
        f"{_RUNS_ROOT}/jepa",
        f"{_RUNS_ROOT}/profile_autoencoder",
        f"{_RUNS_ROOT}/image_autoencoder",
    ])
    run_filter      : List[str] = field(default_factory=list)
    gpus            : List[int] = field(default_factory=lambda: [0])
    poll_interval_s : float     = 5.0

    inference : InferenceConfig = field(default_factory=lambda: InferenceConfig(
        run_directory = Path("."),
        save_cubes    = True,
        cpu_workers   = 16,
        gif_axes      = ["elevation", "range", "azimuth"],
    ))

    profile_inference : ProfileAeInferenceConfig = field(default_factory=lambda: ProfileAeInferenceConfig(run_directory=Path(".")))
    image_inference   : ImageAeInferenceConfig   = field(default_factory=lambda: ImageAeInferenceConfig(run_directory=Path(".")))
