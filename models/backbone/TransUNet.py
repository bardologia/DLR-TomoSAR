from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from configuration.model.backbone_models_config import TransUNetConfig
from ..blocks                          import ConvBlock, PatchEmbedding, TransformerBlock, build_upsample, initialize_weights, match_spatial_size, tokens_to_feature_map


class TransUNet(nn.Module):
    def __init__(self, config: TransUNetConfig | None = None):
        super().__init__()
        if config is None:
            config = TransUNetConfig()
        self.config = config

        if len(config.cnn_features) == 0:
            raise ValueError("cnn_features must contain at least one channel size")
        if config.patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")

        cnn_features        = config.cnn_features
        bottleneck_channels = cnn_features[-1] * config.bottleneck_factor

        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        channels = config.in_channels
        for feature_size in cnn_features:
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

        self.pre_transformer_conv = ConvBlock(
            input_channels  = cnn_features[-1],
            output_channels = bottleneck_channels,
            dropout         = config.dropout,
            activation      = config.activation,
            normalization   = config.normalization,
            bias            = config.conv_bias,
        )

        self.patch_embedding = PatchEmbedding(
            in_channels   = bottleneck_channels,
            embedding_dim = bottleneck_channels,
            patch_size    = config.patch_size,
        )

        drop_path_rates = [x.item() for x in torch.linspace(0, config.stochastic_depth_rate, config.transformer_layers)]
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim     = bottleneck_channels,
                num_heads         = config.transformer_heads,
                mlp_ratio         = config.transformer_mlp_ratio,
                dropout           = config.dropout,
                attention_dropout = config.attention_dropout,
                ffn_activation    = config.ffn_activation,
                drop_path_rate    = drop_path_rates[i],
            )
            for i in range(config.transformer_layers)
        ])
        self.transformer_norm = nn.LayerNorm(bottleneck_channels)

        cnn_downsample          = 2 ** len(cnn_features)
        self.expected_grid_size = config.image_size // cnn_downsample // config.patch_size
        num_patches             = self.expected_grid_size ** 2
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, num_patches, bottleneck_channels)
        )
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        reversed_features    = [bottleneck_channels] + cnn_features[::-1]
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
                ConvBlock(
                    input_channels  = reversed_features[index + 1] * 2,
                    output_channels = reversed_features[index + 1],
                    dropout         = config.dropout,
                    activation      = config.activation,
                    normalization   = config.normalization,
                    bias            = config.conv_bias,
                )
            )

        self.output_head = nn.Conv2d(
            in_channels  = cnn_features[0],
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

        x = self.pre_transformer_conv(x)

        tokens, grid_height, grid_width = self.patch_embedding(x)

        pos_embed = self.positional_embedding
        if grid_height != self.expected_grid_size or grid_width != self.expected_grid_size:
            pos_embed = pos_embed.transpose(1, 2).view(
                1, -1, self.expected_grid_size, self.expected_grid_size
            )
            pos_embed = functional.interpolate(
                input         = pos_embed,
                size          = (grid_height, grid_width),
                mode          = "bilinear",
                align_corners = False,
            )
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
        tokens = tokens + pos_embed

        for transformer_block in self.transformer_blocks:
            tokens = transformer_block(tokens)
        tokens = self.transformer_norm(tokens)
        x      = tokens_to_feature_map(tokens, grid_height, grid_width)

        for upsample, decoder_block, skip in zip(
            self.upsample_layers, self.decoder_blocks, reversed(skip_connections)
        ):
            x = upsample(x)
            x = match_spatial_size(source=x, reference=skip)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        out  = self.output_head(x)
        return out
