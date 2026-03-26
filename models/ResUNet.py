from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import ResUNetConfig, build_activation, build_norm2d, build_upsample, initialize_weights


# Pre-activation residual block: Norm -> Act -> Conv -> Norm -> Act -> Conv + shortcut
class ResidualConvBlock(nn.Module):
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
            build_norm2d(normalization, input_channels),
            build_activation(activation),
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
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)

        if input_channels != output_channels:
            self.shortcut = nn.Conv2d(
                in_channels  = input_channels,
                out_channels = output_channels,
                kernel_size  = 1,
                bias         = bias,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path + identity/projected shortcut (residual connection)
        return self.layers(x) + self.shortcut(x)


# Resizes source tensor to match the spatial dimensions of reference
def match_spatial_size(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if source.shape[2:] != reference.shape[2:]:
        return functional.interpolate(
            input         = source,
            size          = reference.shape[2:],
            mode          = "bilinear",
            align_corners = False,
        )
    return source


# Residual U-Net: U-Net with residual connections inside each conv block (Zhang et al., 2018)
class ResUNet(nn.Module):
    def __init__(self, config: ResUNetConfig | None = None):
        super().__init__()
        if config is None:
            config = ResUNetConfig()
        self.config = config

        if len(config.features) == 0:
            raise ValueError("features must contain at least one channel size")

        feature_sizes       = config.features
        bottleneck_channels = feature_sizes[-1] * config.bottleneck_factor

        self.encoder_blocks    = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        channels = config.in_channels
        for feature_size in feature_sizes:
            self.encoder_blocks.append(
                ResidualConvBlock(
                    input_channels  = channels,
                    output_channels = feature_size,
                    dropout         = config.dropout,
                    activation      = config.activation,
                    normalization   = config.normalization,
                    bias            = config.conv_bias,
                )
            )
            self.downsample_layers.append(nn.MaxPool2d(kernel_size=2))
            channels = feature_size

        self.bottleneck = ResidualConvBlock(
            input_channels  = feature_sizes[-1],
            output_channels = bottleneck_channels,
            dropout         = config.dropout,
            activation      = config.activation,
            normalization   = config.normalization,
            bias            = config.conv_bias,
        )

        reversed_features = [bottleneck_channels] + feature_sizes[::-1]
        self.upsample_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
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

        self.output_head = nn.Conv2d(
            in_channels  = feature_sizes[0],
            out_channels = config.out_channels,
            kernel_size  = 1,
        )

        initialize_weights(module=self, mode=config.init_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder: extract features with residual blocks + MaxPool downsampling
        skip_connections: list[torch.Tensor] = []
        for encoder_block, downsample in zip(self.encoder_blocks, self.downsample_layers):
            x = encoder_block(x)
            skip_connections.append(x)
            x = downsample(x)

        # Bottleneck: deepest residual block
        x = self.bottleneck(x)

        # Decoder: upsample -> concat skip -> residual block
        for upsample, decoder_block, skip in zip(
            self.upsample_layers, self.decoder_blocks, reversed(skip_connections)
        ):
            x = upsample(x)
            x = match_spatial_size(source=x, reference=skip)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        return self.output_head(x)
