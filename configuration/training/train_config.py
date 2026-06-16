from __future__ import annotations

from dataclasses import dataclass, field

from configuration.training.autoencoder_config       import ProfileAeEntryConfig
from configuration.training.backbone_config           import BackboneEntryConfig
from configuration.training.image_autoencoder_config  import ImageAeEntryConfig
from configuration.training.jepa_config               import JepaEntryConfig


@dataclass
class TrainEntryConfig:
    mode : str = "backbone"

    backbone            : BackboneEntryConfig  = field(default_factory=BackboneEntryConfig)
    jepa                : JepaEntryConfig       = field(default_factory=JepaEntryConfig)
    profile_autoencoder : ProfileAeEntryConfig  = field(default_factory=ProfileAeEntryConfig)
    image_autoencoder   : ImageAeEntryConfig    = field(default_factory=ImageAeEntryConfig)
