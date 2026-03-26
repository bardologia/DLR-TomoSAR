from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import AttentionUNetConfig, build_activation, build_norm2d, build_upsample, initialize_weights


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


# Attention Gate: learns to suppress irrelevant skip features using gate signal from decoder
class AttentionGate(nn.Module):
    def __init__(
        self,
        gate_channels:         int,
        skip_channels:         int,
        intermediate_channels: int,
        normalization:         str  = "batch",
        bias:                  bool = False,
    ):
        super().__init__()
        self.gate_projection = nn.Sequential(
            nn.Conv2d(
                in_channels  = gate_channels,
                out_channels = intermediate_channels,
                kernel_size  = 1,
                bias         = bias,
            ),
            build_norm2d(normalization, intermediate_channels),
        )
        self.skip_projection = nn.Sequential(
            nn.Conv2d(
                in_channels  = skip_channels,
                out_channels = intermediate_channels,
                kernel_size  = 1,
                bias         = bias,
            ),
            build_norm2d(normalization, intermediate_channels),
        )
        self.attention_score = nn.Sequential(
            nn.Conv2d(
                in_channels  = intermediate_channels,
                out_channels = 1,
                kernel_size  = 1,
                bias         = bias,
            ),
            build_norm2d(normalization, 1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate_signal: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        # Project gate and skip into shared intermediate space
        projected_gate = self.gate_projection(gate_signal)
        projected_skip = self.skip_projection(skip_connection)

        if projected_gate.shape[2:] != projected_skip.shape[2:]:
            projected_gate = functional.interpolate(
                input         = projected_gate,
                size          = projected_skip.shape[2:],
                mode          = "bilinear",
                align_corners = False,
            )

        # Compute attention coefficients via sigmoid and reweight skip features
        attention_weights = self.attention_score(self.relu(projected_gate + projected_skip))
        return skip_connection * attention_weights


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


# Attention U-Net: U-Net with attention gates on skip connections (Oktay et al., 2018)
class AttentionUNet(nn.Module):
    def __init__(self, config: AttentionUNetConfig | None = None):
        super().__init__()
        if config is None:
            config = AttentionUNetConfig()
        self.config = config

        if len(config.features) == 0:
            raise ValueError("features must contain at least one channel size")
        if config.attention_intermediate_ratio <= 0:
            raise ValueError("attention_intermediate_ratio must be greater than 0")

        feature_sizes = config.features
        bottleneck_channels = feature_sizes[-1] * config.bottleneck_factor

        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        channels = config.in_channels
        for feature_size in feature_sizes:
            self.encoder_blocks.append(
                ConvBlock(
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

        self.bottleneck = ConvBlock(
            input_channels  = feature_sizes[-1],
            output_channels = bottleneck_channels,
            dropout         = config.dropout,
            activation      = config.activation,
            normalization   = config.normalization,
            bias            = config.conv_bias,
        )

        reversed_features = [bottleneck_channels] + feature_sizes[::-1]
        self.upsample_layers = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for index in range(len(reversed_features) - 1):
            decoder_channels = reversed_features[index + 1]
            intermediate_channels = max(1, int(decoder_channels * config.attention_intermediate_ratio))

            self.upsample_layers.append(
                build_upsample(
                    mode         = config.upsample_mode,
                    in_channels  = reversed_features[index],
                    out_channels = decoder_channels,
                )
            )
            self.attention_gates.append(
                AttentionGate(
                    gate_channels         = decoder_channels,
                    skip_channels         = decoder_channels,
                    intermediate_channels = intermediate_channels,
                    normalization         = config.normalization,
                    bias                  = config.conv_bias,
                )
            )
            self.decoder_blocks.append(
                ConvBlock(
                    input_channels  = decoder_channels * 2,
                    output_channels = decoder_channels,
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
        original_input = x
        skip_connections: list[torch.Tensor] = []
        # Encoder: extract multi-scale features
        for encoder_block, downsample in zip(self.encoder_blocks, self.downsample_layers):
            x = encoder_block(x)
            skip_connections.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder: upsample, apply attention gate to skip, concat, and refine
        for upsample, attention_gate, decoder_block, skip in zip(
            self.upsample_layers,
            self.attention_gates,
            self.decoder_blocks,
            reversed(skip_connections),
        ):
            x = upsample(x)
            x = match_spatial_size(source=x, reference=skip)
            skip = attention_gate(gate_signal=x, skip_connection=skip)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        x = match_spatial_size(source=x, reference=original_input)
        return self.output_head(x)
