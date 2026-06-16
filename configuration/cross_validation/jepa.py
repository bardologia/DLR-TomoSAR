from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.training.jepa import EmbeddingLossConfig


@dataclass
class JepaCvConfig:
    profile_autoencoder_logdir : Path                = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/profile_autoencoder")
    profile_autoencoder_run    : str | None          = None
    profile_autoencoder_mode   : str                 = "frozen"
    target_provider            : str                 = "stopgrad"
    embedding_loss             : EmbeddingLossConfig = field(default_factory=EmbeddingLossConfig)
