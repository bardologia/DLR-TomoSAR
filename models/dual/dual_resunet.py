from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures import DualResUNetConfig, ResUNetConfig
from ..backbone.resunet          import ResUNetBackbone
from ..blocks                    import OutputHeadsMixin, PixelMLP, initialize_weights


class DualResUNet(nn.Module, OutputHeadsMixin):
    def __init__(self, config: DualResUNetConfig | None = None):
        super().__init__()
        self.config = config if config is not None else DualResUNetConfig()

        if self.config.head != "set_pred":
            raise ValueError(f"DualResUNet only supports the set_pred head; got '{self.config.head}'")
        if len(self.config.ifg_channels) == 0:
            raise ValueError("ifg_channels must list the interferogram channel indices that feed the existence trunk")
        if any(index < 0 or index >= self.config.in_channels for index in self.config.ifg_channels):
            raise ValueError(f"ifg_channels {tuple(self.config.ifg_channels)} out of range for in_channels={self.config.in_channels}")

        self.trunk_params    = ResUNetBackbone(self._trunk_config(self.config.in_channels, self.config.params_features))
        self.trunk_existence = ResUNetBackbone(self._trunk_config(len(self.config.ifg_channels), self.config.existence_features))

        self.embedding_channels = self.trunk_params.embedding_channels
        self._build_output_head()

        self.register_buffer("ifg_index", torch.tensor(list(self.config.ifg_channels), dtype=torch.long), persistent=False)

        initialize_weights(module=self, mode=self.config.init_mode)

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
        embedding_params    = self.trunk_params.encode_decode(x)
        embedding_existence = self.trunk_existence.encode_decode(x.index_select(1, self.ifg_index))

        return self._set_prediction_forward(embedding_params, embedding_existence)
