from __future__ import annotations

from dataclasses import dataclass, field

from configuration.training.autoencoder_config import ProfileAeEntryConfig
from configuration.training.backbone_config     import BackboneEntryConfig
from configuration.training.jepa_config         import JepaEntryConfig


@dataclass
class TrainEntryConfig:
    mode : str = "backbone"

    backbone    : BackboneEntryConfig  = field(default_factory=BackboneEntryConfig)
    jepa        : JepaEntryConfig       = field(default_factory=JepaEntryConfig)
    autoencoder : ProfileAeEntryConfig = field(default_factory=ProfileAeEntryConfig)
