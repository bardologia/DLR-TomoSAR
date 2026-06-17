from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path


@dataclass
class ModelConfigMigrationConfig:
    runs_dir  : Path = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/backbone")
    dry_run   : bool = False
    log_level : str  = "INFO"
