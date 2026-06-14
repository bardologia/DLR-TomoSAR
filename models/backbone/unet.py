from __future__ import annotations

import torch
import torch.nn as nn

from configuration.model.models_config import UNetConfig, UNetMultiHeadConfig, UNetPerGaussianConfig
from ..blocks                          import initialize_weights
from ..blocks                          import ConvBlock, Decoder, Encoder, GaussianHeadsMixin


class UNetBackbone(nn.Module, GaussianHeadsMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if len(config.features) == 0:
            raise ValueError("features must contain at least one channel size")

        feature_sizes       = config.features
        bottleneck_channels = feature_sizes[-1] * config.bottleneck_factor

        self.encoder = Encoder(
            input_channels = config.in_channels,
            feature_sizes  = feature_sizes,
            dropout        = config.dropout,
            activation     = config.activation,
            normalization  = config.normalization,
            bias           = config.conv_bias,
        )

        self.bottleneck = ConvBlock(
            input_channels  = feature_sizes[-1],
            output_channels = bottleneck_channels,
            dropout         = config.dropout,
            activation      = config.activation,
            normalization   = config.normalization,
            bias            = config.conv_bias,
        )

        decoder_feature_sizes = [bottleneck_channels] + feature_sizes[::-1]
        self.decoder = Decoder(
            feature_sizes = decoder_feature_sizes,
            dropout       = config.dropout,
            activation    = config.activation,
            normalization = config.normalization,
            bias          = config.conv_bias,
            upsample_mode = config.upsample_mode,
        )

        self.embedding_channels = feature_sizes[0]
        self.hidden_channels    = max(self.embedding_channels // 2, 16)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        return self.decoder(x, skip_connections[::-1])


class UNet(UNetBackbone):
    def __init__(self, config: UNetConfig | None = None):
        super().__init__(config if config is not None else UNetConfig())

        self.output_head = nn.Conv2d(
            in_channels  = self.embedding_channels,
            out_channels = self.config.out_channels,
            kernel_size  = 1,
        )

        initialize_weights(module=self, mode=self.config.init_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encode_decode(x)
        return self.output_head(embedding)


class UNetMultiHead(UNetBackbone):
    def __init__(self, config: UNetMultiHeadConfig | None = None):
        super().__init__(config if config is not None else UNetMultiHeadConfig())
        self._resolve_gaussian_layout()
        self._build_triple_heads()

        initialize_weights(module=self, mode=self.config.init_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._triple_head_forward(self.encode_decode(x))


class UNetPerGaussian(UNetBackbone):
    def __init__(self, config: UNetPerGaussianConfig | None = None):
        super().__init__(config if config is not None else UNetPerGaussianConfig())
        self._resolve_gaussian_layout()
        self._build_per_gaussian_heads()

        initialize_weights(module=self, mode=self.config.init_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._per_gaussian_forward(self.encode_decode(x))
