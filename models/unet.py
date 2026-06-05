from __future__ import annotations

import torch
import torch.nn as nn

from configuration.models_config import UNetConfig, UNetMultiHeadConfig, UNetPerGaussianConfig
from .blocks import initialize_weights
from .blocks import ConvBlock, Decoder, Encoder, PixelMLP


class UNetBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
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

        self.embedding_channels = feature_sizes[0]
        self.hidden_channels    = max(self.embedding_channels // 2, 16)

    def _resolve_gaussian_layout(self) -> None:
        n_params = self.config.params_per_gaussian
        if self.config.out_channels % n_params != 0:
            raise ValueError(
                f"out_channels ({self.config.out_channels}) must be divisible by "
                f"params_per_gaussian ({n_params})"
            )
        self.n_gaussians = self.config.out_channels // n_params
        self.n_params    = n_params

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        return self.decoder(x, skip_connections[::-1])


class UNet(UNetBackbone):
    def __init__(self, config: UNetConfig | None = None):
        super().__init__(config if config is not None else UNetConfig())

        self.output_head = nn.Conv2d(
            in_channels  = self.embedding_channels,
            out_channels = self.config.out_channels,
            kernel_size  = 1,
        )

        initialize_weights(module=self, mode=self.config.init_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encode_decode(x)
        return self.output_head(embedding)


class UNetMultiHead(UNetBackbone):
    def __init__(self, config: UNetMultiHeadConfig | None = None):
        super().__init__(config if config is not None else UNetMultiHeadConfig())
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


class UNetPerGaussian(UNetBackbone):
    def __init__(self, config: UNetPerGaussianConfig | None = None):
        super().__init__(config if config is not None else UNetPerGaussianConfig())
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
