from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import UNetPlusPlusConfig, build_activation, build_norm2d, initialize_weights


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


class UNetPlusPlus(nn.Module):
    def __init__(self, config: UNetPlusPlusConfig | None = None):
        super().__init__()
        if config is None:
            config = UNetPlusPlusConfig()
        self.config = config

        feature_sizes = config.features
        assert len(feature_sizes) == 4
        self.deep_supervision = config.deep_supervision

        level_0, level_1, level_2, level_3 = feature_sizes
        bottleneck_width = level_3 * config.bottleneck_factor
        dropout = config.dropout
        act = config.activation
        norm = config.normalization
        bias = config.conv_bias

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.encoder_0_0 = ConvBlock(config.in_channels, level_0, dropout, act, norm, bias)
        self.encoder_1_0 = ConvBlock(level_0, level_1, dropout, act, norm, bias)
        self.encoder_2_0 = ConvBlock(level_1, level_2, dropout, act, norm, bias)
        self.encoder_3_0 = ConvBlock(level_2, level_3, dropout, act, norm, bias)
        self.encoder_4_0 = ConvBlock(level_3, bottleneck_width, dropout, act, norm, bias)

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
                [nn.Conv2d(level_0, config.out_channels, kernel_size=1) for _ in range(4)]
            )
        else:
            self.output_head = nn.Conv2d(level_0, config.out_channels, kernel_size=1)

        initialize_weights(self, config.init_mode)

    def _upsample_and_match(self, source, reference):
        return match_spatial_size(self.upsample(source), reference)

    def forward(self, x):
        node_0_0 = self.encoder_0_0(x)
        node_1_0 = self.encoder_1_0(self.pool(node_0_0))
        node_2_0 = self.encoder_2_0(self.pool(node_1_0))
        node_3_0 = self.encoder_3_0(self.pool(node_2_0))
        node_4_0 = self.encoder_4_0(self.pool(node_3_0))

        node_0_1 = self.dense_0_1(torch.cat([node_0_0, self._upsample_and_match(node_1_0, node_0_0)], 1))
        node_1_1 = self.dense_1_1(torch.cat([node_1_0, self._upsample_and_match(node_2_0, node_1_0)], 1))
        node_2_1 = self.dense_2_1(torch.cat([node_2_0, self._upsample_and_match(node_3_0, node_2_0)], 1))
        node_3_1 = self.dense_3_1(torch.cat([node_3_0, self._upsample_and_match(node_4_0, node_3_0)], 1))

        node_0_2 = self.dense_0_2(torch.cat([node_0_0, node_0_1, self._upsample_and_match(node_1_1, node_0_0)], 1))
        node_1_2 = self.dense_1_2(torch.cat([node_1_0, node_1_1, self._upsample_and_match(node_2_1, node_1_0)], 1))
        node_2_2 = self.dense_2_2(torch.cat([node_2_0, node_2_1, self._upsample_and_match(node_3_1, node_2_0)], 1))

        node_0_3 = self.dense_0_3(torch.cat([node_0_0, node_0_1, node_0_2, self._upsample_and_match(node_1_2, node_0_0)], 1))
        node_1_3 = self.dense_1_3(torch.cat([node_1_0, node_1_1, node_1_2, self._upsample_and_match(node_2_2, node_1_0)], 1))

        node_0_4 = self.dense_0_4(torch.cat([node_0_0, node_0_1, node_0_2, node_0_3, self._upsample_and_match(node_1_3, node_0_0)], 1))

        if self.deep_supervision:
            return [
                self.output_heads[0](node_0_1),
                self.output_heads[1](node_0_2),
                self.output_heads[2](node_0_3),
                self.output_heads[3](node_0_4),
            ]
        return self.output_head(node_0_4)
