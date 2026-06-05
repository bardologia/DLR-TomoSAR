from __future__ import annotations

import torch
import torch.nn as nn

from configuration.models_config import ResUNetConfig, ResUNetMultiHeadConfig, ResUNetPerGaussianConfig, UNetSkipConfig
from .blocks import build_upsample, initialize_weights
from .blocks import PixelMLP, ResidualConvBlock, match_spatial_size


class ResUNetBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if len(config.features) == 0:
            raise ValueError("features must contain at least one channel size")

        feature_sizes       = config.features
        bottleneck_channels = feature_sizes[-1] * config.bottleneck_factor

        self.encoder_blocks = nn.ModuleList()
        channels = config.in_channels
        for index, feature_size in enumerate(feature_sizes):
            self.encoder_blocks.append(
                ResidualConvBlock(
                    input_channels  = channels,
                    output_channels = feature_size,
                    dropout         = config.dropout,
                    activation      = config.activation,
                    normalization   = config.normalization,
                    bias            = config.conv_bias,
                    stride          = 1 if index == 0 else 2,
                    first_unit      = index == 0,
                )
            )
            channels = feature_size

        self.bottleneck = ResidualConvBlock(
            input_channels  = feature_sizes[-1],
            output_channels = bottleneck_channels,
            dropout         = config.dropout,
            activation      = config.activation,
            normalization   = config.normalization,
            bias            = config.conv_bias,
            stride          = 2,
        )

        reversed_features    = [bottleneck_channels] + feature_sizes[::-1]
        self.upsample_layers = nn.ModuleList()
        self.decoder_blocks  = nn.ModuleList()
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

        self.embedding_channels = feature_sizes[0]
        self.hidden_channels    = max(self.embedding_channels // 2, 16)

    def _resolve_gaussian_layout(self) -> None:
        n_params = self.config.params_per_gaussian
        if self.config.out_channels % n_params != 0:
            raise ValueError(f"out_channels ({self.config.out_channels}) must be divisible by params_per_gaussian ({n_params})")
        self.n_gaussians = self.config.out_channels // n_params
        self.n_params    = n_params

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections: list[torch.Tensor] = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)

        x = self.bottleneck(x)

        for upsample, decoder_block, skip in zip(self.upsample_layers, self.decoder_blocks, reversed(skip_connections)):
            x = upsample(x)
            x = match_spatial_size(source=x, reference=skip)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        return x


class ResUNet(ResUNetBackbone):
    def __init__(self, config: ResUNetConfig | None = None):
        super().__init__(config if config is not None else ResUNetConfig())

        self.output_head = nn.Conv2d(
            in_channels  = self.embedding_channels,
            out_channels = self.config.out_channels,
            kernel_size  = 1,
        )

        initialize_weights(module=self, mode=self.config.init_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encode_decode(x)
        return self.output_head(embedding)


class ResUNetMultiHead(ResUNetBackbone):
    def __init__(self, config: ResUNetMultiHeadConfig | None = None):
        super().__init__(config if config is not None else ResUNetMultiHeadConfig())
        self._resolve_gaussian_layout()

        self.head_amp   = PixelMLP(self.embedding_channels, self.hidden_channels, self.n_gaussians, self.config.activation)
        self.head_mu    = PixelMLP(self.embedding_channels, self.hidden_channels, self.n_gaussians, self.config.activation)
        self.head_sigma = PixelMLP(self.embedding_channels, self.hidden_channels, self.n_gaussians, self.config.activation)

        initialize_weights(module=self, mode=self.config.init_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encode_decode(x)

        amp   = self.head_amp(embedding)
        mu    = self.head_mu(embedding)
        sigma = self.head_sigma(embedding)

        B, K, H, W = amp.shape
        out = torch.stack([amp, mu, sigma], dim=2)
        out = out.view(B, K * 3, H, W)
        return out


class ResUNetPerGaussian(ResUNetBackbone):
    def __init__(self, config: ResUNetPerGaussianConfig | None = None):
        super().__init__(config if config is not None else ResUNetPerGaussianConfig())
        self._resolve_gaussian_layout()

        self.gaussian_heads = nn.ModuleList([
            PixelMLP(self.embedding_channels, self.hidden_channels, self.n_params, self.config.activation)
            for _ in range(self.n_gaussians)
        ])

        initialize_weights(module=self, mode=self.config.init_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encode_decode(x)

        head_outputs = [head(embedding) for head in self.gaussian_heads]

        B, _, H, W = head_outputs[0].shape
        out = torch.stack(head_outputs, dim=1)
        out = out.view(B, self.n_gaussians * self.n_params, H, W)
        return out


class UNetSkip(nn.Module):
    def __init__(self, config: UNetSkipConfig | None = None):
        super().__init__()
        if config is None:
            config = UNetSkipConfig()
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

        reversed_features    = [bottleneck_channels] + feature_sizes[::-1]
        self.upsample_layers = nn.ModuleList()
        self.decoder_blocks  = nn.ModuleList()
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
        skip_connections: list[torch.Tensor] = []
        for encoder_block, downsample in zip(self.encoder_blocks, self.downsample_layers):
            x = encoder_block(x)
            skip_connections.append(x)
            x = downsample(x)

        x = self.bottleneck(x)

        for upsample, decoder_block, skip in zip(
            self.upsample_layers, self.decoder_blocks, reversed(skip_connections)
        ):
            x = upsample(x)
            x = match_spatial_size(source=x, reference=skip)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        out = self.output_head(x)
        return out
