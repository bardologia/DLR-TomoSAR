from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import UNetConfig, build_activation, build_norm2d, build_upsample, initialize_weights


# Double 3x3 convolution block: Conv -> Norm -> Act -> Conv -> Norm -> Act (+ optional dropout)
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


# Resizes source tensor to match the spatial dimensions of reference (handles size mismatches)
def match_spatial_size(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if source.shape[2:] != reference.shape[2:]:
        return functional.interpolate(
            input         = source,
            size          = reference.shape[2:],
            mode          = "bilinear",
            align_corners = False,
        )
    return source


# Encoder: series of ConvBlocks followed by MaxPool downsampling; stores skip connections
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


# Decoder: upsample -> concat with skip connection -> ConvBlock at each level
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
                    input_channels  = feature_sizes[index],
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


# Standard U-Net: symmetric encoder-decoder with skip connections (Ronneberger et al., 2015)
class UNet(nn.Module):
    def __init__(self, config: UNetConfig | None = None):
        super().__init__()
        if config is None:
            config = UNetConfig()
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

        self.output_head = nn.Conv2d(
            in_channels  = feature_sizes[0],
            out_channels = config.out_channels,
            kernel_size  = 1,
        )

        initialize_weights(module=self, mode=config.init_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Contracting path: extract features at multiple scales
        x, skip_connections = self.encoder(x)
        # Bridge: deepest feature representation
        x                   = self.bottleneck(x)
        # Expanding path: recover spatial resolution using skip connections
        x                   = self.decoder(x, skip_connections[::-1])
        # Final 1x1 conv to map to output classes
        return self.output_head(x)
