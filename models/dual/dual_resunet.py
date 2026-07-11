from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures import DualResUNetConfig, ResUNetConfig
from ..backbone.resunet          import ResUNetBackbone
from ..backbone.unet             import UNetBackbone
from ..blocks                    import OutputHeadsMixin, PixelMLP, initialize_weights


class DualResUNet(nn.Module, OutputHeadsMixin):

    TRUNKS = ("resunet", "unet_skip", "unet")

    def __init__(self, config: DualResUNetConfig | None = None):
        super().__init__()
        self.config = config if config is not None else DualResUNetConfig()

        self._validate()

        self.trunk_params    = self._build_trunk(self.config.params_backbone,    self.config.params_channels,    self.config.params_features)
        self.trunk_existence = self._build_trunk(self.config.existence_backbone, self.config.existence_channels, self.config.existence_features)

        self.embedding_channels = self.trunk_params.embedding_channels
        self._build_output_head()

        self.register_buffer("params_index",    torch.tensor(list(self.config.params_channels),    dtype=torch.long), persistent=False)
        self.register_buffer("existence_index", torch.tensor(list(self.config.existence_channels), dtype=torch.long), persistent=False)

        initialize_weights(module=self, mode=self.config.init_mode)

    def _validate(self) -> None:
        config = self.config

        if config.head != "set_pred":
            raise ValueError(f"DualResUNet only supports the set_pred head; got '{config.head}'")

        for name, backbone in (("params_backbone", config.params_backbone), ("existence_backbone", config.existence_backbone)):
            if backbone not in self.TRUNKS:
                raise ValueError(f"Unknown {name} '{backbone}'. Available: {list(self.TRUNKS)}")

        for name, channels in (("params_channels", config.params_channels), ("existence_channels", config.existence_channels)):
            if len(channels) == 0:
                raise ValueError(f"{name} must list the input channel indices that feed the trunk")
            if any(index < 0 or index >= config.in_channels for index in channels):
                raise ValueError(f"{name} {tuple(channels)} out of range for in_channels={config.in_channels}")

    def _build_trunk(self, backbone: str, channels: tuple, features: list[int]):
        trunk_config = self._trunk_config(len(channels), features)

        if backbone == "unet":
            return UNetBackbone(trunk_config)

        return ResUNetBackbone(trunk_config, downsample="stride" if backbone == "resunet" else "maxpool")

    def _trunk_config(self, in_channels: int, features: list[int]) -> ResUNetConfig:
        return ResUNetConfig(
            in_channels         = in_channels,
            out_channels        = self.config.out_channels,
            params_per_gaussian = self.config.params_per_gaussian,
            head                = self.config.head,
            features            = list(features),
            bottleneck_factor   = self.config.bottleneck_factor,
            dropout             = self.config.dropout,
            activation          = self.config.activation,
            normalization       = self.config.normalization,
            upsample_mode       = self.config.upsample_mode,
            conv_bias           = self.config.conv_bias,
            init_mode           = self.config.init_mode,
        )

    def _build_set_prediction_heads(self) -> None:
        self._build_per_gaussian_heads()

        existence_channels  = self.trunk_existence.embedding_channels
        existence_hidden    = max(existence_channels // 2, 16)
        self.existence_head = PixelMLP(existence_channels, existence_hidden, self.n_gaussians, self._head_activation())
        self.amp_off        = nn.Parameter(torch.zeros(self.n_gaussians))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding_params    = self.trunk_params.encode_decode(x.index_select(1, self.params_index))
        embedding_existence = self.trunk_existence.encode_decode(x.index_select(1, self.existence_index))

        return self._set_prediction_forward(embedding_params, embedding_existence)
