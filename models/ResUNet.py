from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import ResUNetConfig, build_activation, build_norm2d, build_upsample, initialize_weights


class ResidualConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dropout: float = 0.0,
                 activation: str = "relu", normalization: str = "batch", bias: bool = False):
        super().__init__()
        layers = [
            build_norm2d(normalization, input_channels),
            build_activation(activation),
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=bias),
            build_norm2d(normalization, output_channels),
            build_activation(activation),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=bias),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)

        if input_channels != output_channels:
            self.shortcut = nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=bias)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.layers(x) + self.shortcut(x)


def match_spatial_size(source, reference):
    if source.shape[2:] != reference.shape[2:]:
        return functional.interpolate(
            source,
            size=reference.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
    return source


class ResUNet(nn.Module):
    def __init__(self, config: ResUNetConfig | None = None):
        super().__init__()
        if config is None:
            config = ResUNetConfig()
        self.config = config

        feature_sizes = config.features
        bottleneck_channels = feature_sizes[-1] * config.bottleneck_factor

        self.encoder_blocks    = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        channels = config.in_channels
        for feature_size in feature_sizes:
            self.encoder_blocks.append(ResidualConvBlock(channels, feature_size, config.dropout, config.activation, config.normalization, config.conv_bias))
            self.downsample_layers.append(nn.MaxPool2d(2))
            channels = feature_size

        self.bottleneck = ResidualConvBlock(feature_sizes[-1], bottleneck_channels, config.dropout,
                                            config.activation, config.normalization, config.conv_bias)

        reversed_features = [bottleneck_channels] + feature_sizes[::-1]
        self.upsample_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for index in range(len(reversed_features) - 1):
            self.upsample_layers.append(
                build_upsample(config.upsample_mode, reversed_features[index], reversed_features[index + 1])
            )
            self.decoder_blocks.append(
                ResidualConvBlock(reversed_features[index], reversed_features[index + 1], config.dropout,
                                   config.activation, config.normalization, config.conv_bias)
            )

        self.output_head = nn.Conv2d(feature_sizes[0], config.out_channels, kernel_size=1)

        initialize_weights(self, config.init_mode)

    def forward(self, x):
        skip_connections = []
        for encoder_block, downsample in zip(self.encoder_blocks, self.downsample_layers):
            x = encoder_block(x)
            skip_connections.append(x)
            x = downsample(x)

        x = self.bottleneck(x)

        for upsample, decoder_block, skip in zip(
            self.upsample_layers, self.decoder_blocks, reversed(skip_connections)
        ):
            x = upsample(x)
            x = match_spatial_size(x, skip)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        return self.output_head(x)
