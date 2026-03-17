from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import UNetConfig, build_activation, build_norm2d, build_upsample, initialize_weights


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dropout: float = 0.0,
                 activation: str = "relu", normalization: str = "batch", bias: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=bias),
            build_norm2d(normalization, output_channels),
            build_activation(activation),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=bias),
            build_norm2d(normalization, output_channels),
            build_activation(activation),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def match_spatial_size(source, reference):
    if source.shape[2:] != reference.shape[2:]:
        return functional.interpolate(
            source,
            size=reference.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
    return source


class Encoder(nn.Module):
    def __init__(self, input_channels: int, feature_sizes: list[int], dropout: float = 0.0,
                 activation: str = "relu", normalization: str = "batch", bias: bool = False):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        channels = input_channels
        for feature_size in feature_sizes:
            self.conv_blocks.append(ConvBlock(channels, feature_size, dropout, activation, normalization, bias))
            self.downsample_layers.append(nn.MaxPool2d(2))
            channels = feature_size

    def forward(self, x):
        skip_connections = []
        for conv_block, downsample in zip(self.conv_blocks, self.downsample_layers):
            x = conv_block(x)
            skip_connections.append(x)
            x = downsample(x)
        return x, skip_connections


class Decoder(nn.Module):
    def __init__(self, feature_sizes: list[int], dropout: float = 0.0,
                 activation: str = "relu", normalization: str = "batch", bias: bool = False,
                 upsample_mode: str = "convtranspose"):
        super().__init__()
        self.upsample_layers = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        for index in range(len(feature_sizes) - 1):
            self.upsample_layers.append(
                build_upsample(upsample_mode, feature_sizes[index], feature_sizes[index + 1])
            )
            self.conv_blocks.append(
                ConvBlock(feature_sizes[index], feature_sizes[index + 1], dropout, activation, normalization, bias)
            )

    def forward(self, x, skip_connections):
        for upsample, conv_block, skip in zip(self.upsample_layers, self.conv_blocks, skip_connections):
            x = upsample(x)
            x = match_spatial_size(x, skip)
            x = torch.cat([skip, x], dim=1)
            x = conv_block(x)
        return x


class UNet(nn.Module):
    def __init__(self, config: UNetConfig | None = None):
        super().__init__()
        self.config = config

        feature_sizes       = config.features
        bottleneck_channels = feature_sizes[-1] * config.bottleneck_factor

        self.encoder = Encoder(config.in_channels, feature_sizes, config.dropout,
                               config.activation, config.normalization, config.conv_bias)
        self.bottleneck = ConvBlock(feature_sizes[-1], bottleneck_channels, config.dropout,
                                    config.activation, config.normalization, config.conv_bias)

        decoder_feature_sizes = [bottleneck_channels] + feature_sizes[::-1]
        self.decoder = Decoder(decoder_feature_sizes, config.dropout,
                               config.activation, config.normalization, config.conv_bias,
                               config.upsample_mode)

        self.output_head = nn.Conv2d(feature_sizes[0], config.out_channels, kernel_size=1)

        initialize_weights(self, config.init_mode)

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x                   = self.bottleneck(x)
        x                   = self.decoder(x, skip_connections[::-1])
        
        return self.output_head(x)
