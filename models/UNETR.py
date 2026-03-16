from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import UNETRConfig


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


class UNETR(nn.Module):
    def __init__(self, config: UNETRConfig | None = None):
        super().__init__()
        if config is None:
            config = UNETRConfig()
        self.config = config

        self.patch_embedding = PatchEmbedding(
            in_channels=config.in_channels,
            embedding_dim=config.embedding_dim,
            patch_size=config.patch_size,
        )

        self.positional_embedding = nn.Parameter(
            torch.zeros(1, (config.image_size // config.patch_size) ** 2, config.embedding_dim)
        )
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=config.embedding_dim,
                num_heads=config.transformer_heads,
                mlp_ratio=config.transformer_mlp_ratio,
                dropout=config.dropout,
            )
            for _ in range(config.transformer_layers)
        ])
        self.transformer_norm = nn.LayerNorm(config.embedding_dim)

        total_layers = config.transformer_layers
        self.skip_layer_indices = [
            total_layers // 4 - 1,
            total_layers // 2 - 1,
            (3 * total_layers) // 4 - 1,
            total_layers - 1,
        ]

        decoder_features = config.decoder_features
        self.skip_projections = nn.ModuleList()
        for decoder_feature_size in decoder_features:
            self.skip_projections.append(
                nn.Sequential(
                    nn.Conv2d(config.embedding_dim, decoder_feature_size, kernel_size=1, bias=False),
                    nn.BatchNorm2d(decoder_feature_size),
                    nn.ReLU(inplace=True),
                )
            )

        self.input_skip_conv = ConvBlock(config.in_channels, decoder_features[-1], config.dropout)

        self.upsample_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for index in range(len(decoder_features)):
            if index == 0:
                input_to_upsample = decoder_features[0]
            else:
                input_to_upsample = decoder_features[index - 1]

            self.upsample_layers.append(
                nn.ConvTranspose2d(input_to_upsample, decoder_features[index], kernel_size=2, stride=2)
            )

            if index < len(decoder_features) - 1:
                concat_channels = decoder_features[index] + decoder_features[index + 1]
            else:
                concat_channels = decoder_features[index] + decoder_features[-1]

            self.decoder_blocks.append(
                ConvBlock(concat_channels, decoder_features[index], config.dropout)
            )

        self.final_upsample = nn.ConvTranspose2d(
            decoder_features[-1], decoder_features[-1], kernel_size=config.patch_size // (2 ** len(decoder_features)),
            stride=config.patch_size // (2 ** len(decoder_features)),
        )
        self.output_head = nn.Conv2d(decoder_features[-1], config.out_channels, kernel_size=1)

    def forward(self, x):
        original_input = x

        tokens, grid_height, grid_width = self.patch_embedding(x)
        tokens = tokens + self.positional_embedding[:, :tokens.shape[1], :]

        collected_skip_features = []
        for layer_index, transformer_block in enumerate(self.transformer_blocks):
            tokens = transformer_block(tokens)
            if layer_index in self.skip_layer_indices:
                feature_map = tokens_to_feature_map(
                    self.transformer_norm(tokens) if layer_index == len(self.transformer_blocks) - 1 else tokens,
                    grid_height, grid_width,
                )
                skip_index = self.skip_layer_indices.index(layer_index)
                projected = self.skip_projections[skip_index](feature_map)
                collected_skip_features.append(projected)

        x = collected_skip_features[0]

        for index in range(len(self.upsample_layers)):
            x = self.upsample_layers[index](x)

            if index + 1 < len(collected_skip_features):
                skip = collected_skip_features[index + 1]
            else:
                skip = self.input_skip_conv(original_input)

            x = match_spatial_size(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = self.decoder_blocks[index](x)

        x = self.final_upsample(x)
        x = match_spatial_size(x, original_input)
        return self.output_head(x)
