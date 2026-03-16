from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import TransUNetConfig


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query_key_value = nn.Linear(embedding_dim, embedding_dim * 3)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, sequence_length, embedding_dim = x.shape
        qkv = self.query_key_value(x).reshape(
            batch_size, sequence_length, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        attention_weights = (query @ key.transpose(-2, -1)) * self.scale
        attention_weights = attention_weights.softmax(dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        attended = (attention_weights @ value).transpose(1, 2).reshape(
            batch_size, sequence_length, embedding_dim
        )
        return self.output_projection(attended)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadSelfAttention(embedding_dim, num_heads, dropout)
        self.norm_2 = nn.LayerNorm(embedding_dim)
        hidden_dim = int(embedding_dim * mlp_ratio)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.norm_1(x))
        x = x + self.feed_forward(self.norm_2(x))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels, embedding_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.projection(x)
        grid_height, grid_width = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, grid_height, grid_width


def match_spatial_size(source, reference):
    if source.shape[2:] != reference.shape[2:]:
        return functional.interpolate(
            source,
            size=reference.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
    return source


def tokens_to_feature_map(tokens, height, width):
    batch_size, _, channels = tokens.shape
    return tokens.transpose(1, 2).view(batch_size, channels, height, width)


class TransUNet(nn.Module):
    def __init__(self, config: TransUNetConfig | None = None):
        super().__init__()
        if config is None:
            config = TransUNetConfig()
        self.config = config

        cnn_features = config.cnn_features
        bottleneck_channels = cnn_features[-1] * config.bottleneck_factor

        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        channels = config.in_channels
        for feature_size in cnn_features:
            self.encoder_blocks.append(ConvBlock(channels, feature_size, config.dropout))
            self.downsample_layers.append(nn.MaxPool2d(2))
            channels = feature_size

        self.pre_transformer_conv = ConvBlock(cnn_features[-1], bottleneck_channels, config.dropout)

        self.patch_embedding = PatchEmbedding(
            in_channels=bottleneck_channels,
            embedding_dim=bottleneck_channels,
            patch_size=config.patch_size,
        )

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=bottleneck_channels,
                num_heads=config.transformer_heads,
                mlp_ratio=config.transformer_mlp_ratio,
                dropout=config.dropout,
            )
            for _ in range(config.transformer_layers)
        ])
        self.transformer_norm = nn.LayerNorm(bottleneck_channels)

        reversed_features = [bottleneck_channels] + cnn_features[::-1]
        self.upsample_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for index in range(len(reversed_features) - 1):
            self.upsample_layers.append(
                nn.ConvTranspose2d(reversed_features[index], reversed_features[index + 1], kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(
                ConvBlock(reversed_features[index], reversed_features[index + 1], config.dropout)
            )

        self.output_head = nn.Conv2d(cnn_features[0], config.out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for encoder_block, downsample in zip(self.encoder_blocks, self.downsample_layers):
            x = encoder_block(x)
            skip_connections.append(x)
            x = downsample(x)

        x = self.pre_transformer_conv(x)

        tokens, grid_height, grid_width = self.patch_embedding(x)
        for transformer_block in self.transformer_blocks:
            tokens = transformer_block(tokens)
        tokens = self.transformer_norm(tokens)
        x = tokens_to_feature_map(tokens, grid_height, grid_width)

        if x.shape[2:] != skip_connections[-1].shape[2:]:
            x = match_spatial_size(x, skip_connections[-1])

        for upsample, decoder_block, skip in zip(
            self.upsample_layers, self.decoder_blocks, reversed(skip_connections)
        ):
            x = upsample(x)
            x = match_spatial_size(x, skip)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        return self.output_head(x)
