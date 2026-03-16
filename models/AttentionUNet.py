from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import AttentionUNetConfig


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


class AttentionGate(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, intermediate_channels: int):
        super().__init__()
        self.gate_projection = nn.Sequential(
            nn.Conv2d(gate_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
        )
        self.skip_projection = nn.Sequential(
            nn.Conv2d(skip_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
        )
        self.attention_score = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate_signal, skip_connection):
        projected_gate = self.gate_projection(gate_signal)
        projected_skip = self.skip_projection(skip_connection)

        if projected_gate.shape[2:] != projected_skip.shape[2:]:
            projected_gate = functional.interpolate(
                projected_gate,
                size=projected_skip.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        attention_weights = self.attention_score(self.relu(projected_gate + projected_skip))
        return skip_connection * attention_weights


def match_spatial_size(source, reference):
    if source.shape[2:] != reference.shape[2:]:
        return functional.interpolate(
            source,
            size=reference.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
    return source


class AttentionUNet(nn.Module):
    def __init__(self, config: AttentionUNetConfig | None = None):
        super().__init__()
        if config is None:
            config = AttentionUNetConfig()
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
        self.attention_gates = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for index in range(len(reversed_features) - 1):
            decoder_channels = reversed_features[index + 1]
            intermediate_channels = max(1, int(decoder_channels * config.attention_intermediate_ratio))

            self.upsample_layers.append(
                nn.ConvTranspose2d(reversed_features[index], decoder_channels, kernel_size=2, stride=2)
            )
            self.attention_gates.append(
                AttentionGate(
                    gate_channels=decoder_channels,
                    skip_channels=decoder_channels,
                    intermediate_channels=intermediate_channels,
                )
            )
            self.decoder_blocks.append(
                ConvBlock(reversed_features[index], decoder_channels, config.dropout)
            )

        self.output_head = nn.Conv2d(feature_sizes[0], config.out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for encoder_block, downsample in zip(self.encoder_blocks, self.downsample_layers):
            x = encoder_block(x)
            skip_connections.append(x)
            x = downsample(x)

        x = self.bottleneck(x)

        for upsample, attention_gate, decoder_block, skip in zip(
            self.upsample_layers,
            self.attention_gates,
            self.decoder_blocks,
            reversed(skip_connections),
        ):
            x = upsample(x)
            x = match_spatial_size(x, skip)
            skip = attention_gate(gate_signal=x, skip_connection=skip)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        return self.output_head(x)
