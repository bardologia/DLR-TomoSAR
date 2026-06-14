from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from configuration.model.models_config import U2NetLiteConfig
from ..blocks                          import build_activation, build_norm2d, initialize_weights


class RSUConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dilation: int, activation: str, normalization: str, bias: bool):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=bias),
            build_norm2d(normalization, output_channels),
            build_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class RSU(nn.Module):
    def __init__(self, height: int, input_channels: int, mid_channels: int, output_channels: int, activation: str, normalization: str, bias: bool):
        super().__init__()
        if height < 2:
            raise ValueError(f"RSU height must be at least 2; got {height}")
        self.height = height

        self.conv_in = RSUConv(input_channels, output_channels, 1, activation, normalization, bias)

        self.encoder_convs = nn.ModuleList([RSUConv(output_channels, mid_channels, 1, activation, normalization, bias)])
        for _ in range(height - 2):
            self.encoder_convs.append(RSUConv(mid_channels, mid_channels, 1, activation, normalization, bias))

        self.bottom_conv = RSUConv(mid_channels, mid_channels, 2, activation, normalization, bias)

        self.decoder_convs = nn.ModuleList()
        for index in range(height - 1):
            out_channels = output_channels if index == height - 2 else mid_channels
            self.decoder_convs.append(RSUConv(mid_channels * 2, out_channels, 1, activation, normalization, bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        encoder_outputs = []
        x = x_in
        for index, conv in enumerate(self.encoder_convs):
            x = conv(x)
            encoder_outputs.append(x)
            if index < self.height - 2:
                x = functional.max_pool2d(x, kernel_size=2, ceil_mode=True)

        x = self.bottom_conv(x)

        for index, conv in enumerate(self.decoder_convs):
            skip = encoder_outputs[-(index + 1)]
            if x.shape[2:] != skip.shape[2:]:
                x = functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = conv(torch.cat([x, skip], dim=1))

        return x + x_in


class RSUDilated(nn.Module):
    def __init__(self, input_channels: int, mid_channels: int, output_channels: int, activation: str, normalization: str, bias: bool):
        super().__init__()
        self.conv_in = RSUConv(input_channels, output_channels, 1, activation, normalization, bias)

        self.encoder_convs = nn.ModuleList([
            RSUConv(output_channels, mid_channels, 1, activation, normalization, bias),
            RSUConv(mid_channels,    mid_channels, 2, activation, normalization, bias),
            RSUConv(mid_channels,    mid_channels, 4, activation, normalization, bias),
        ])

        self.bottom_conv = RSUConv(mid_channels, mid_channels, 8, activation, normalization, bias)

        self.decoder_convs = nn.ModuleList([
            RSUConv(mid_channels * 2, mid_channels,    4, activation, normalization, bias),
            RSUConv(mid_channels * 2, mid_channels,    2, activation, normalization, bias),
            RSUConv(mid_channels * 2, output_channels, 1, activation, normalization, bias),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        encoder_outputs = []
        x = x_in
        for conv in self.encoder_convs:
            x = conv(x)
            encoder_outputs.append(x)

        x = self.bottom_conv(x)

        for index, conv in enumerate(self.decoder_convs):
            skip = encoder_outputs[-(index + 1)]
            x    = conv(torch.cat([x, skip], dim=1))

        return x + x_in


class U2NetLite(nn.Module):
    def __init__(self, config: U2NetLiteConfig | None = None):
        super().__init__()
        if config is None:
            config = U2NetLiteConfig()
        self.config = config

        if len(config.features) != 4:
            raise ValueError(f"features must contain exactly four channel sizes; got {len(config.features)}")

        f0, f1, f2, f3 = config.features
        heights        = config.rsu_heights

        def mid(channels: int) -> int:
            return max(channels // 4, 8)

        self.encoder_stages = nn.ModuleList([
            RSU(heights[0], config.in_channels, mid(f0), f0, config.activation, config.normalization, config.conv_bias),
            RSU(heights[1], f0,                 mid(f1), f1, config.activation, config.normalization, config.conv_bias),
            RSU(heights[2], f1,                 mid(f2), f2, config.activation, config.normalization, config.conv_bias),
        ])

        self.bridge = RSUDilated(f2, mid(f3), f3, config.activation, config.normalization, config.conv_bias)

        self.decoder_stages = nn.ModuleList([
            RSU(heights[2], f3 + f2, mid(f2), f2, config.activation, config.normalization, config.conv_bias),
            RSU(heights[1], f2 + f1, mid(f1), f1, config.activation, config.normalization, config.conv_bias),
            RSU(heights[0], f1 + f0, mid(f0), f0, config.activation, config.normalization, config.conv_bias),
        ])

        self.dropout     = nn.Dropout2d(config.dropout) if config.dropout > 0 else nn.Identity()
        self.output_head = nn.Conv2d(f0, config.out_channels, kernel_size=1)

        initialize_weights(module=self, mode=config.init_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs = []
        for stage in self.encoder_stages:
            x = stage(x)
            encoder_outputs.append(x)
            x = functional.max_pool2d(x, kernel_size=2, ceil_mode=True)

        x = self.bridge(x)

        for stage, skip in zip(self.decoder_stages, reversed(encoder_outputs)):
            x = functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = stage(torch.cat([x, skip], dim=1))

        x = self.dropout(x)
        return self.output_head(x)
