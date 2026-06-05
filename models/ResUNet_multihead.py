from __future__ import annotations

import torch
import torch.nn as nn

from configuration.models_config import ResUNetMultiHeadConfig, build_upsample, initialize_weights
from .ResUNet import ResidualConvBlock, match_spatial_size
from .UNet_multihead import PixelMLP


class ResUNetMultiHead(nn.Module):
    def __init__(self, config: ResUNetMultiHeadConfig | None = None):
        super().__init__()
        if config is None:
            config = ResUNetMultiHeadConfig()
        self.config = config

        n_params = config.params_per_gaussian
        if config.out_channels % n_params != 0:
            raise ValueError(f"out_channels ({config.out_channels}) must be divisible by params_per_gaussian ({n_params})")
        self.n_gaussians = config.out_channels // n_params

        if len(config.features) == 0:
            raise ValueError("features must contain at least one channel size")

        feature_sizes       = config.features
        bottleneck_channels = feature_sizes[-1] * config.bottleneck_factor

        self.encoder_blocks = nn.ModuleList()
        channels = config.in_channels
        for index, feature_size in enumerate(feature_sizes):
            self.encoder_blocks.append(
                ResidualConvBlock(
                    input_channels  = channels,
                    output_channels = feature_size,
                    dropout         = config.dropout,
                    activation      = config.activation,
                    normalization   = config.normalization,
                    bias            = config.conv_bias,
                    stride          = 1 if index == 0 else 2,
                    first_unit      = index == 0,
                )
            )
            channels = feature_size

        self.bottleneck = ResidualConvBlock(
            input_channels  = feature_sizes[-1],
            output_channels = bottleneck_channels,
            dropout         = config.dropout,
            activation      = config.activation,
            normalization   = config.normalization,
            bias            = config.conv_bias,
            stride          = 2,
        )

        reversed_features    = [bottleneck_channels] + feature_sizes[::-1]
        self.upsample_layers = nn.ModuleList()
        self.decoder_blocks  = nn.ModuleList()
        for index in range(len(reversed_features) - 1):
            self.upsample_layers.append(
                build_upsample(
                    mode         = config.upsample_mode,
                    in_channels  = reversed_features[index],
                    out_channels = reversed_features[index + 1],
                )
            )
            self.decoder_blocks.append(
                ResidualConvBlock(
                    input_channels  = reversed_features[index + 1] * 2,
                    output_channels = reversed_features[index + 1],
                    dropout         = config.dropout,
                    activation      = config.activation,
                    normalization   = config.normalization,
                    bias            = config.conv_bias,
                )
            )

        embedding_channels = feature_sizes[0]
        hidden_channels    = max(embedding_channels // 2, 16)

        self.head_amp   = PixelMLP(embedding_channels, hidden_channels, self.n_gaussians, config.activation)
        self.head_mu    = PixelMLP(embedding_channels, hidden_channels, self.n_gaussians, config.activation)
        self.head_sigma = PixelMLP(embedding_channels, hidden_channels, self.n_gaussians, config.activation)

        initialize_weights(module=self, mode=config.init_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections: list[torch.Tensor] = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)

        x = self.bottleneck(x)

        for upsample, decoder_block, skip in zip(self.upsample_layers, self.decoder_blocks, reversed(skip_connections)):
            x = upsample(x)
            x = match_spatial_size(source=x, reference=skip)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        amp   = self.head_amp(x)
        mu    = self.head_mu(x)
        sigma = self.head_sigma(x)

        B, K, H, W = amp.shape
        out = torch.stack([amp, mu, sigma], dim=2)
        out = out.view(B, K * 3, H, W)
        return out
