from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as functional

from .config import LinkNetConfig


def match_spatial_size(source, reference):
    if source.shape[2:] != reference.shape[2:]:
        return functional.interpolate(
            source,
            size=reference.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
    return source


class ResidualEncoderBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.main_path = nn.Sequential(*layers)

        self.shortcut_path = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(output_channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.main_path(x) + self.shortcut_path(x))


class BottleneckDecoderBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, compression_ratio: int = 4):
        super().__init__()
        compressed_channels = max(1, input_channels // compression_ratio)
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, compressed_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(compressed_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                compressed_channels, compressed_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1, bias=False,
            ),
            nn.BatchNorm2d(compressed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(compressed_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class LinkNet(nn.Module):
    def __init__(self, config: LinkNetConfig | None = None):
        super().__init__()
        if config is None:
            config = LinkNetConfig()
        self.config = config

        feature_sizes = config.features
        kernel_size = config.initial_kernel_size
        padding = kernel_size // 2

        self.initial_conv = nn.Sequential(
            nn.Conv2d(config.in_channels, feature_sizes[0], kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(feature_sizes[0]),
            nn.ReLU(inplace=True),
        )

        self.encoder_stages = nn.ModuleList()
        channels = feature_sizes[0]
        for feature_size in feature_sizes:
            self.encoder_stages.append(ResidualEncoderBlock(channels, feature_size, config.dropout))
            channels = feature_size

        self.decoder_stages = nn.ModuleList()
        for index in range(len(feature_sizes) - 1, 0, -1):
            self.decoder_stages.append(
                BottleneckDecoderBlock(feature_sizes[index], feature_sizes[index - 1], config.decoder_bottleneck_ratio)
            )
        self.decoder_stages.append(
            BottleneckDecoderBlock(feature_sizes[0], feature_sizes[0], config.decoder_bottleneck_ratio)
        )

        self.output_head = nn.Conv2d(feature_sizes[0], config.out_channels, kernel_size=1)

    def forward(self, x):
        x = self.initial_conv(x)

        encoder_outputs = []
        for encoder_stage in self.encoder_stages:
            x = encoder_stage(x)
            encoder_outputs.append(x)

        for stage_index, decoder_stage in enumerate(self.decoder_stages):
            current_skip = encoder_outputs[-(stage_index + 1)]
            decoded = decoder_stage(current_skip if stage_index == 0 else x)

            if stage_index + 1 < len(encoder_outputs):
                target_skip = encoder_outputs[-(stage_index + 2)]
                decoded = match_spatial_size(decoded, target_skip)
                x = decoded + target_skip
            else:
                x = decoded

        return self.output_head(x)
