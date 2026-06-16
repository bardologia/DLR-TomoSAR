from __future__ import annotations

import torch
import torch.nn as nn

from configuration.model.backbone_models_config import ConvNeXtUNetConfig
from ..blocks                          import DropPath, build_activation, initialize_weights
from ..blocks                          import match_spatial_size


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, ffn_ratio: float, drop_path: float, ffn_activation: str, layer_scale_init: float):
        super().__init__()
        hidden = int(channels * ffn_ratio)

        self.dwconv     = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm       = nn.LayerNorm(channels)
        self.fc1        = nn.Linear(channels, hidden)
        self.activation = build_activation(ffn_activation)
        self.fc2        = nn.Linear(hidden, channels)
        self.drop_path  = DropPath(drop_path)

        self.gamma = nn.Parameter(layer_scale_init * torch.ones(channels)) if layer_scale_init > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        return residual + self.drop_path(x)


class ConvNeXtStage(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, n_blocks: int, ffn_ratio: float, drop_path_rates: list[float], ffn_activation: str, layer_scale_init: float):
        super().__init__()
        self.projection = nn.Conv2d(input_channels, output_channels, kernel_size=1) if input_channels != output_channels else nn.Identity()
        self.blocks     = nn.Sequential(*[
            ConvNeXtBlock(output_channels, ffn_ratio, drop_path_rates[index], ffn_activation, layer_scale_init)
            for index in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.projection(x))


class ConvNeXtUNet(nn.Module):
    def __init__(self, config: ConvNeXtUNetConfig | None = None):
        super().__init__()
        if config is None:
            config = ConvNeXtUNetConfig()
        self.config = config

        if len(config.features) == 0:
            raise ValueError("features must contain at least one channel size")

        feature_sizes       = config.features
        n_stages            = len(feature_sizes)
        bottleneck_channels = feature_sizes[-1] * config.bottleneck_factor

        total_blocks    = config.blocks_per_stage * (2 * n_stages + 1)
        drop_path_rates = [config.stochastic_depth_rate * index / max(total_blocks - 1, 1) for index in range(total_blocks)]

        def take_rates() -> list[float]:
            nonlocal rate_index
            rates       = drop_path_rates[rate_index:rate_index + config.blocks_per_stage]
            rate_index += config.blocks_per_stage
            return rates

        rate_index = 0

        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, feature_sizes[0], kernel_size=3, padding=1, bias=config.conv_bias),
            ChannelLayerNorm(feature_sizes[0]),
        )

        self.encoder_stages    = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        channels = feature_sizes[0]
        for feature_size in feature_sizes:
            self.encoder_stages.append(ConvNeXtStage(channels, feature_size, config.blocks_per_stage, config.ffn_ratio, take_rates(), config.ffn_activation, config.layer_scale_init))
            self.downsample_layers.append(nn.Sequential(
                ChannelLayerNorm(feature_size),
                nn.Conv2d(feature_size, feature_size, kernel_size=2, stride=2),
            ))
            channels = feature_size

        self.bottleneck = ConvNeXtStage(feature_sizes[-1], bottleneck_channels, config.blocks_per_stage, config.ffn_ratio, take_rates(), config.ffn_activation, config.layer_scale_init)

        reversed_features    = [bottleneck_channels] + feature_sizes[::-1]
        self.upsample_layers = nn.ModuleList()
        self.decoder_stages  = nn.ModuleList()
        for index in range(len(reversed_features) - 1):
            self.upsample_layers.append(nn.ConvTranspose2d(reversed_features[index], reversed_features[index + 1], kernel_size=2, stride=2))
            self.decoder_stages.append(ConvNeXtStage(reversed_features[index + 1] * 2, reversed_features[index + 1], config.blocks_per_stage, config.ffn_ratio, take_rates(), config.ffn_activation, config.layer_scale_init))

        self.output_head = nn.Conv2d(feature_sizes[0], config.out_channels, kernel_size=1)

        initialize_weights(module=self, mode=config.init_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        skip_connections: list[torch.Tensor] = []
        for stage, downsample in zip(self.encoder_stages, self.downsample_layers):
            x = stage(x)
            skip_connections.append(x)
            x = downsample(x)

        x = self.bottleneck(x)

        for upsample, stage, skip in zip(self.upsample_layers, self.decoder_stages, reversed(skip_connections)):
            x = upsample(x)
            x = match_spatial_size(source=x, reference=skip)
            x = torch.cat([skip, x], dim=1)
            x = stage(x)

        return self.output_head(x)
