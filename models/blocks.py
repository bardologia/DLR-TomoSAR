from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from configuration.models_config import build_activation, build_norm2d, build_upsample


def match_spatial_size(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if source.shape[2:] != reference.shape[2:]:
        return functional.interpolate(
            input         = source,
            size          = reference.shape[2:],
            mode          = "bilinear",
            align_corners = False,
        )
    return source


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_channels:  int,
        output_channels: int,
        dropout:         float = 0.0,
        activation:      str   = "relu",
        normalization:   str   = "batch",
        bias:            bool  = False,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels  = input_channels,
                out_channels = output_channels,
                kernel_size  = 3,
                padding      = 1,
                bias         = bias,
            ),
            build_norm2d(normalization, output_channels),
            build_activation(activation),
            nn.Conv2d(
                in_channels  = output_channels,
                out_channels = output_channels,
                kernel_size  = 3,
                padding      = 1,
                bias         = bias,
            ),
            build_norm2d(normalization, output_channels),
            build_activation(activation),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        input_channels:  int,
        output_channels: int,
        dropout:         float = 0.0,
        activation:      str   = "relu",
        normalization:   str   = "batch",
        bias:            bool  = False,
        stride:          int   = 1,
        first_unit:      bool  = False,
    ):
        super().__init__()
        layers = []
        if not first_unit:
            layers.append(build_norm2d(normalization, input_channels))
            layers.append(build_activation(activation))

        layers.append(
            nn.Conv2d(
                in_channels  = input_channels,
                out_channels = output_channels,
                kernel_size  = 3,
                stride       = stride,
                padding      = 1,
                bias         = bias,
            )
        )
        layers.append(build_norm2d(normalization, output_channels))
        layers.append(build_activation(activation))
        layers.append(
            nn.Conv2d(
                in_channels  = output_channels,
                out_channels = output_channels,
                kernel_size  = 3,
                padding      = 1,
                bias         = bias,
            )
        )
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)

        if input_channels != output_channels or stride != 1:
            self.shortcut = nn.Conv2d(
                in_channels  = input_channels,
                out_channels = output_channels,
                kernel_size  = 1,
                stride       = stride,
                bias         = bias,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + self.shortcut(x)


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


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        feature_sizes:  list[int],
        dropout:        float = 0.0,
        activation:     str   = "relu",
        normalization:  str   = "batch",
        bias:           bool  = False,
    ):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        channels = input_channels
        for feature_size in feature_sizes:
            self.conv_blocks.append(
                ConvBlock(
                    input_channels  = channels,
                    output_channels = feature_size,
                    dropout         = dropout,
                    activation      = activation,
                    normalization   = normalization,
                    bias            = bias,
                )
            )
            self.downsample_layers.append(nn.MaxPool2d(kernel_size=2))
            channels = feature_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skip_connections: list[torch.Tensor] = []
        for conv_block, downsample in zip(self.conv_blocks, self.downsample_layers):
            x = conv_block(x)
            skip_connections.append(x)
            x = downsample(x)
        return x, skip_connections


class Decoder(nn.Module):
    def __init__(
        self,
        feature_sizes: list[int],
        dropout:       float = 0.0,
        activation:    str   = "relu",
        normalization: str   = "batch",
        bias:          bool  = False,
        upsample_mode: str   = "convtranspose",
    ):
        super().__init__()
        self.upsample_layers = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        for index in range(len(feature_sizes) - 1):
            self.upsample_layers.append(
                build_upsample(
                    mode         = upsample_mode,
                    in_channels  = feature_sizes[index],
                    out_channels = feature_sizes[index + 1],
                )
            )
            self.conv_blocks.append(
                ConvBlock(
                    input_channels  = feature_sizes[index + 1] * 2,
                    output_channels = feature_sizes[index + 1],
                    dropout         = dropout,
                    activation      = activation,
                    normalization   = normalization,
                    bias            = bias,
                )
            )

    def forward(self, x: torch.Tensor, skip_connections: list[torch.Tensor]) -> torch.Tensor:
        for upsample, conv_block, skip in zip(self.upsample_layers, self.conv_blocks, skip_connections):
            x = upsample(x)
            x = match_spatial_size(source=x, reference=skip)
            x = torch.cat([skip, x], dim=1)
            x = conv_block(x)
        return x
