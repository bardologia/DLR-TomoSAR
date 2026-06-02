from __future__ import annotations

import torch
import torch.nn as nn

from configuration.models_config import UNetPerGaussianConfig, build_activation, initialize_weights
from .UNet import ConvBlock, Encoder, Decoder
from .UNet_multihead import PixelMLP


class UNetPerGaussian(nn.Module):

    def __init__(self, config: UNetPerGaussianConfig | None = None):
        super().__init__()
        if config is None:
            config = UNetPerGaussianConfig()
        self.config = config

        n_params = config.params_per_gaussian
        if config.out_channels % n_params != 0:
            raise ValueError(
                f"out_channels ({config.out_channels}) must be divisible by "
                f"params_per_gaussian ({n_params})"
            )
        self.n_gaussians  = config.out_channels // n_params
        self.n_params     = n_params

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

        embedding_channels = feature_sizes[0]
        hidden_channels    = max(embedding_channels // 2, 16)

        # One head per Gaussian, each predicts n_params channels
        self.gaussian_heads = nn.ModuleList([
            PixelMLP(embedding_channels, hidden_channels, n_params, config.activation)
            for _ in range(self.n_gaussians)
        ])

        initialize_weights(module=self, mode=config.init_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        embedding = self.decoder(x, skip_connections[::-1])

        # Each head: (B, n_params, H, W)  →  stack along channel dim
        head_outputs = [head(embedding) for head in self.gaussian_heads]

        # Interleave so output layout is [amp_0, mu_0, sig_0, amp_1, mu_1, sig_1, ...]
        B, _, H, W = head_outputs[0].shape
        out = torch.stack(head_outputs, dim=1)   # (B, K, n_params, H, W)
        out = out.view(B, self.n_gaussians * self.n_params, H, W)
        return out
