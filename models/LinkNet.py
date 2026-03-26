from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import LinkNetConfig, build_activation, build_norm2d, initialize_weights


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


# Encoder block with stride-2 conv for downsampling and a residual shortcut path
class ResidualEncoderBlock(nn.Module):
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
                stride       = 2,
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
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.main_path = nn.Sequential(*layers)

        self.shortcut_path = nn.Sequential(
            nn.Conv2d(
                in_channels  = input_channels,
                out_channels = output_channels,
                kernel_size  = 1,
                stride       = 2,
                bias         = bias,
            ),
            build_norm2d(normalization, output_channels),
        )
        self.activation = build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add main path and shortcut (residual connection)
        return self.activation(self.main_path(x) + self.shortcut_path(x))


# Decoder block: 1x1 compress -> ConvTranspose2d upsample -> 1x1 expand (bottleneck design)
class BottleneckDecoderBlock(nn.Module):
    def __init__(
        self,
        input_channels:    int,
        output_channels:   int,
        compression_ratio: int   = 4,
        activation:        str   = "relu",
        normalization:     str   = "batch",
        bias:              bool  = False,
    ):
        super().__init__()
        compressed_channels = max(1, input_channels // compression_ratio)
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels  = input_channels,
                out_channels = compressed_channels,
                kernel_size  = 1,
                bias         = bias,
            ),
            build_norm2d(normalization, compressed_channels),
            build_activation(activation),
            nn.ConvTranspose2d(
                in_channels    = compressed_channels,
                out_channels   = compressed_channels,
                kernel_size    = 3,
                stride         = 2,
                padding        = 1,
                output_padding = 1,
                bias           = bias,
            ),
            build_norm2d(normalization, compressed_channels),
            build_activation(activation),
            nn.Conv2d(
                in_channels  = compressed_channels,
                out_channels = output_channels,
                kernel_size  = 1,
                bias         = bias,
            ),
            build_norm2d(normalization, output_channels),
            build_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# LinkNet: lightweight architecture with additive skip connections (Chaurasia & Culurciello, 2017)
class LinkNet(nn.Module):
    def __init__(self, config: LinkNetConfig | None = None):
        super().__init__()
        if config is None:
            config = LinkNetConfig()
        self.config = config

        if len(config.features) == 0:
            raise ValueError("features must contain at least one channel size")
        if config.initial_kernel_size <= 0:
            raise ValueError("initial_kernel_size must be a positive integer")
        if config.decoder_bottleneck_ratio <= 0:
            raise ValueError("decoder_bottleneck_ratio must be a positive integer")

        feature_sizes = config.features
        kernel_size = config.initial_kernel_size
        padding = kernel_size // 2

        self.initial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels  = config.in_channels,
                out_channels = feature_sizes[0],
                kernel_size  = kernel_size,
                stride       = 1,
                padding      = padding,
                bias         = config.conv_bias,
            ),
            build_norm2d(config.normalization, feature_sizes[0]),
            build_activation(config.activation),
        )

        self.encoder_stages = nn.ModuleList()
        channels = feature_sizes[0]
        for feature_size in feature_sizes:
            self.encoder_stages.append(
                ResidualEncoderBlock(
                    input_channels  = channels,
                    output_channels = feature_size,
                    dropout         = config.dropout,
                    activation      = config.activation,
                    normalization   = config.normalization,
                    bias            = config.conv_bias,
                )
            )
            channels = feature_size

        self.decoder_stages = nn.ModuleList()
        for index in range(len(feature_sizes) - 1, 0, -1):
            self.decoder_stages.append(
                BottleneckDecoderBlock(
                    input_channels    = feature_sizes[index],
                    output_channels   = feature_sizes[index - 1],
                    compression_ratio = config.decoder_bottleneck_ratio,
                    activation        = config.activation,
                    normalization     = config.normalization,
                    bias              = config.conv_bias,
                )
            )
        self.decoder_stages.append(
            BottleneckDecoderBlock(
                input_channels    = feature_sizes[0],
                output_channels   = feature_sizes[0],
                compression_ratio = config.decoder_bottleneck_ratio,
                activation        = config.activation,
                normalization     = config.normalization,
                bias              = config.conv_bias,
            )
        )

        self.output_head = nn.Conv2d(
            in_channels  = feature_sizes[0],
            out_channels = config.out_channels,
            kernel_size  = 1,
        )

        initialize_weights(module=self, mode=config.init_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_input = x
        # Initial feature extraction
        x = self.initial_conv(x)

        # Encoder: residual blocks with stride-2 downsampling
        encoder_outputs: list[torch.Tensor] = []
        for encoder_stage in self.encoder_stages:
            x = encoder_stage(x)
            encoder_outputs.append(x)

        # Decoder: bottleneck upsample + element-wise addition of encoder features
        for stage_index, decoder_stage in enumerate(self.decoder_stages):
            current_skip = encoder_outputs[-(stage_index + 1)]
            decoder_input = current_skip if stage_index == 0 else x
            decoded = decoder_stage(decoder_input)

            if stage_index + 1 < len(encoder_outputs):
                target_skip = encoder_outputs[-(stage_index + 2)]
                decoded = match_spatial_size(source=decoded, reference=target_skip)
                x = decoded + target_skip
            else:
                x = decoded

        x = match_spatial_size(source=x, reference=original_input)
        return self.output_head(x)
