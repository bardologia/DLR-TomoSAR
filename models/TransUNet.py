from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import TransUNetConfig, build_activation, build_norm2d, build_upsample, DropPath, initialize_weights


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


# Standard multi-head self-attention with scaled dot-product (Vaswani et al., 2017)
class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim:     int,
        num_heads:         int,
        dropout:           float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query_key_value = nn.Linear(embedding_dim, embedding_dim * 3)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = x.shape
        qkv = self.query_key_value(x)
        qkv = qkv.reshape(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        attention_weights = (query @ key.transpose(-2, -1)) * self.scale
        attention_weights = attention_weights.softmax(dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        attended = (attention_weights @ value).transpose(1, 2)
        attended = attended.reshape(batch_size, sequence_length, embedding_dim)
        projected = self.output_projection(attended)
        return self.output_dropout(projected)


# Transformer block: LayerNorm -> MHSA -> DropPath -> LayerNorm -> FFN -> DropPath
class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim:     int,
        num_heads:         int,
        mlp_ratio:         float = 4.0,
        dropout:           float = 0.0,
        attention_dropout: float = 0.0,
        ffn_activation:    str   = "gelu",
        drop_path_rate:    float = 0.0,
    ):
        super().__init__()
        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadSelfAttention(
            embedding_dim     = embedding_dim,
            num_heads         = num_heads,
            dropout           = dropout,
            attention_dropout = attention_dropout,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm_2 = nn.LayerNorm(embedding_dim)
        hidden_dim = int(embedding_dim * mlp_ratio)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            build_activation(ffn_activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_1(x)
        x = x + self.drop_path(self.attention(normalized))
        normalized = self.norm_2(x)
        x = x + self.drop_path(self.feed_forward(normalized))
        return x


# Splits feature map into non-overlapping patches and projects them to token embeddings
class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels:   int,
        embedding_dim: int,
        patch_size:    int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = embedding_dim,
            kernel_size  = patch_size,
            stride       = patch_size,
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.projection(x)
        grid_height, grid_width = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, grid_height, grid_width


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


# Reshapes flat token sequence back into a 2D feature map (B, C, H, W)
def tokens_to_feature_map(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch_size, _, channels = tokens.shape
    return tokens.transpose(1, 2).view(batch_size, channels, height, width)


# TransUNet: CNN encoder + Transformer bottleneck + CNN decoder (Chen et al., 2021)
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

        cnn_features = config.cnn_features
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

        cnn_downsample = 2 ** len(cnn_features)
        self.expected_grid_size = config.image_size // cnn_downsample // config.patch_size
        num_patches = self.expected_grid_size ** 2
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, num_patches, bottleneck_channels)
        )
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        reversed_features = [bottleneck_channels] + cnn_features[::-1]
        self.upsample_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
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
        # CNN encoder: extract multi-scale features and store skip connections
        skip_connections: list[torch.Tensor] = []
        for encoder_block, downsample in zip(self.encoder_blocks, self.downsample_layers):
            x = encoder_block(x)
            skip_connections.append(x)
            x = downsample(x)

        # Pre-transformer convolution to match embedding dimension
        x = self.pre_transformer_conv(x)

        # Tokenize: split bottleneck features into patch tokens
        tokens, grid_height, grid_width = self.patch_embedding(x)

        # Add learnable positional embeddings (interpolate if grid size differs)
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

        # Transformer encoder: global self-attention across all patch tokens
        for transformer_block in self.transformer_blocks:
            tokens = transformer_block(tokens)
        tokens = self.transformer_norm(tokens)
        # Reshape tokens back to 2D feature map
        x = tokens_to_feature_map(tokens, grid_height, grid_width)

        if x.shape[2:] != skip_connections[-1].shape[2:]:
            x = match_spatial_size(source=x, reference=skip_connections[-1])

        # CNN decoder: upsample -> concat skip -> ConvBlock at each level
        for upsample, decoder_block, skip in zip(
            self.upsample_layers, self.decoder_blocks, reversed(skip_connections)
        ):
            x = upsample(x)
            x = match_spatial_size(source=x, reference=skip)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        return self.output_head(x)
