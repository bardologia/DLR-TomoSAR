from __future__ import annotations

import torch.nn as nn

from .config import FCNConfig


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FCN(nn.Module):
    def __init__(self, config: FCNConfig | None = None):
        super().__init__()
        if config is None:
            config = FCNConfig()
        self.config = config

        feature_sizes = config.features
        bottleneck_channels = feature_sizes[-1] * config.bottleneck_factor

        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        channels = config.in_channels
        for feature_size in feature_sizes:
            self.encoder_blocks.append(ConvBlock(channels, feature_size, config.dropout))
            self.downsample_layers.append(nn.MaxPool2d(2))
            channels = feature_size

        self.bottleneck = ConvBlock(feature_sizes[-1], bottleneck_channels, config.dropout)

        reversed_features = [bottleneck_channels] + feature_sizes[::-1]
        self.upsample_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for index in range(len(reversed_features) - 1):
            self.upsample_layers.append(
                nn.ConvTranspose2d(reversed_features[index], reversed_features[index + 1], kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(
                ConvBlock(reversed_features[index + 1], reversed_features[index + 1], config.dropout)
            )

        self.output_head = nn.Conv2d(feature_sizes[0], config.out_channels, kernel_size=1)

    def forward(self, x):
        for encoder_block, downsample in zip(self.encoder_blocks, self.downsample_layers):
            x = encoder_block(x)
            x = downsample(x)

        x = self.bottleneck(x)

        for upsample, decoder_block in zip(self.upsample_layers, self.decoder_blocks):
            x = upsample(x)
            x = decoder_block(x)

        return self.output_head(x)
