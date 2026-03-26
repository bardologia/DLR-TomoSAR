from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import UNetPlusPlusConfig, build_activation, build_norm2d, initialize_weights


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


# UNet++: nested U-Net with dense skip connections (Zhou et al., 2018)
class UNetPlusPlus(nn.Module):
    def __init__(self, config: UNetPlusPlusConfig | None = None):
        super().__init__()
        if config is None:
            config = UNetPlusPlusConfig()
        self.config = config

        feature_sizes = config.features
        if len(feature_sizes) != 4:
            raise ValueError("features must contain exactly 4 channel sizes")
        self.deep_supervision = config.deep_supervision

        level_0, level_1, level_2, level_3 = feature_sizes
        bottleneck_width = level_3 * config.bottleneck_factor
        dropout = config.dropout
        act = config.activation
        norm = config.normalization
        bias = config.conv_bias

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Backbone encoder nodes (column 0): progressively deeper features
        self.encoder_0_0 = ConvBlock(config.in_channels, level_0, dropout, act, norm, bias)
        self.encoder_1_0 = ConvBlock(level_0, level_1, dropout, act, norm, bias)
        self.encoder_2_0 = ConvBlock(level_1, level_2, dropout, act, norm, bias)
        self.encoder_3_0 = ConvBlock(level_2, level_3, dropout, act, norm, bias)
        self.encoder_4_0 = ConvBlock(level_3, bottleneck_width, dropout, act, norm, bias)

        # Dense intermediate nodes: each aggregates skip connections from all prior columns
        self.dense_0_1 = ConvBlock(level_0 + level_1, level_0, dropout, act, norm, bias)
        self.dense_1_1 = ConvBlock(level_1 + level_2, level_1, dropout, act, norm, bias)
        self.dense_2_1 = ConvBlock(level_2 + level_3, level_2, dropout, act, norm, bias)
        self.dense_3_1 = ConvBlock(level_3 + bottleneck_width, level_3, dropout, act, norm, bias)

        self.dense_0_2 = ConvBlock(level_0 * 2 + level_1, level_0, dropout, act, norm, bias)
        self.dense_1_2 = ConvBlock(level_1 * 2 + level_2, level_1, dropout, act, norm, bias)
        self.dense_2_2 = ConvBlock(level_2 * 2 + level_3, level_2, dropout, act, norm, bias)

        self.dense_0_3 = ConvBlock(level_0 * 3 + level_1, level_0, dropout, act, norm, bias)
        self.dense_1_3 = ConvBlock(level_1 * 3 + level_2, level_1, dropout, act, norm, bias)

        self.dense_0_4 = ConvBlock(level_0 * 4 + level_1, level_0, dropout, act, norm, bias)

        if self.deep_supervision:
            self.output_heads = nn.ModuleList(
                [nn.Conv2d(in_channels=level_0, out_channels=config.out_channels, kernel_size=1) for _ in range(4)]
            )
        else:
            self.output_head = nn.Conv2d(
                in_channels  = level_0,
                out_channels = config.out_channels,
                kernel_size  = 1,
            )

        initialize_weights(module=self, mode=config.init_mode)

    def _upsample_and_match(self, source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        upsampled = self.upsample(source)
        return match_spatial_size(source=upsampled, reference=reference)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        # Column 0: encoder backbone (downsampling path)
        node_0_0 = self.encoder_0_0(x)
        node_1_0 = self.encoder_1_0(self.pool(node_0_0))
        node_2_0 = self.encoder_2_0(self.pool(node_1_0))
        node_3_0 = self.encoder_3_0(self.pool(node_2_0))
        node_4_0 = self.encoder_4_0(self.pool(node_3_0))

        # Column 1: first set of dense skip connections
        up_1_0 = self._upsample_and_match(source=node_1_0, reference=node_0_0)
        node_0_1 = self.dense_0_1(torch.cat([node_0_0, up_1_0], dim=1))
        up_2_0 = self._upsample_and_match(source=node_2_0, reference=node_1_0)
        node_1_1 = self.dense_1_1(torch.cat([node_1_0, up_2_0], dim=1))
        up_3_0 = self._upsample_and_match(source=node_3_0, reference=node_2_0)
        node_2_1 = self.dense_2_1(torch.cat([node_2_0, up_3_0], dim=1))
        up_4_0 = self._upsample_and_match(source=node_4_0, reference=node_3_0)
        node_3_1 = self.dense_3_1(torch.cat([node_3_0, up_4_0], dim=1))

        # Column 2: accumulates features from columns 0 and 1
        up_1_1 = self._upsample_and_match(source=node_1_1, reference=node_0_0)
        node_0_2 = self.dense_0_2(torch.cat([node_0_0, node_0_1, up_1_1], dim=1))
        up_2_1 = self._upsample_and_match(source=node_2_1, reference=node_1_0)
        node_1_2 = self.dense_1_2(torch.cat([node_1_0, node_1_1, up_2_1], dim=1))
        up_3_1 = self._upsample_and_match(source=node_3_1, reference=node_2_0)
        node_2_2 = self.dense_2_2(torch.cat([node_2_0, node_2_1, up_3_1], dim=1))

        # Column 3: accumulates features from columns 0, 1, and 2
        up_1_2 = self._upsample_and_match(source=node_1_2, reference=node_0_0)
        node_0_3 = self.dense_0_3(torch.cat([node_0_0, node_0_1, node_0_2, up_1_2], dim=1))
        up_2_2 = self._upsample_and_match(source=node_2_2, reference=node_1_0)
        node_1_3 = self.dense_1_3(torch.cat([node_1_0, node_1_1, node_1_2, up_2_2], dim=1))

        # Column 4: final densely-connected node with all prior features
        up_1_3 = self._upsample_and_match(source=node_1_3, reference=node_0_0)
        node_0_4 = self.dense_0_4(torch.cat([node_0_0, node_0_1, node_0_2, node_0_3, up_1_3], dim=1))

        # Output: deep supervision returns predictions from multiple columns, otherwise final node only
        if self.deep_supervision:
            return [
                self.output_heads[0](node_0_1),
                self.output_heads[1](node_0_2),
                self.output_heads[2](node_0_3),
                self.output_heads[3](node_0_4),
            ]
        return self.output_head(node_0_4)
