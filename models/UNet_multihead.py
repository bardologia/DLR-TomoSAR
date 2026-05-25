from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import UNetMultiHeadConfig, build_activation, initialize_weights
from .UNet import ConvBlock, Encoder, Decoder, match_spatial_size


class PixelMLP(nn.Module):
    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int,
        out_channels:    int,
        activation:      str = "relu",
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True),
            build_activation(activation),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class UNetMultiHead(nn.Module):

    def __init__(self, config: UNetMultiHeadConfig | None = None):
        super().__init__()
        if config is None:
            config = UNetMultiHeadConfig()
        self.config = config

        n_params = config.params_per_gaussian
        if config.out_channels % n_params != 0:
            raise ValueError(
                f"out_channels ({config.out_channels}) must be divisible by "
                f"params_per_gaussian ({n_params})"
            )
        self.n_gaussians = config.out_channels // n_params

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

        embedding_channels = feature_sizes[0]
        hidden_channels    = max(embedding_channels // 2, 16)

        self.head_amp   = PixelMLP(embedding_channels, hidden_channels, self.n_gaussians, config.activation)
        self.head_mu    = PixelMLP(embedding_channels, hidden_channels, self.n_gaussians, config.activation)
        self.head_sigma = PixelMLP(embedding_channels, hidden_channels, self.n_gaussians, config.activation)

        initialize_weights(module=self, mode=config.init_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        embedding = self.decoder(x, skip_connections[::-1])

        amp   = self.head_amp(embedding)
        mu    = self.head_mu(embedding)
        sigma = self.head_sigma(embedding)

        B, K, H, W = amp.shape
        out = torch.stack([amp, mu, sigma], dim=2)
        out = out.view(B, K * 3, H, W)
        return out
