from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import UNETRConfig, build_activation, build_norm2d, DropPath, initialize_weights


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


# Projects transformer tokens to CNN feature space with progressive ConvTranspose upsampling
class ProgressiveProjectionHead(nn.Module):
    def __init__(
        self,
        input_channels:  int,
        output_channels: int,
        upsample_steps:  int,
        activation:      str   = "relu",
        normalization:   str   = "batch",
        bias:            bool  = False,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels  = input_channels,
                out_channels = output_channels,
                kernel_size  = 1,
                bias         = bias,
            ),
            build_norm2d(normalization, output_channels),
            build_activation(activation),
        ]
        for _ in range(upsample_steps):
            layers.extend([
                nn.ConvTranspose2d(
                    in_channels  = output_channels,
                    out_channels = output_channels,
                    kernel_size  = 2,
                    stride       = 2,
                    bias         = bias,
                ),
                build_norm2d(normalization, output_channels),
                build_activation(activation),
            ])
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
        self.head_dim  = embedding_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.query_key_value   = nn.Linear(embedding_dim, embedding_dim * 3)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = x.shape
        qkv = self.query_key_value(x)
        qkv = qkv.reshape(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        attention_weights = (query @ key.transpose(-2, -1)) * self.scale
        attention_weights = attention_weights.softmax(dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        attended  = (attention_weights @ value).transpose(1, 2)
        attended  = attended.reshape(batch_size, sequence_length, embedding_dim)
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
        self.norm_1    = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadSelfAttention(
            embedding_dim     = embedding_dim,
            num_heads         = num_heads,
            dropout           = dropout,
            attention_dropout = attention_dropout,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm_2    = nn.LayerNorm(embedding_dim)
        
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


# Splits input image into non-overlapping patches and projects them to token embeddings
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


# UNETR: pure-transformer encoder with CNN decoder and multi-scale skip connections (Hatamizadeh et al., 2022)
class UNETR(nn.Module):
    def __init__(self, config: UNETRConfig | None = None):
        super().__init__()
        config = config or UNETRConfig()
        self.config = config

        if config.patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        if config.image_size % config.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if config.transformer_layers < 4:
            raise ValueError("transformer_layers must be at least 4 to collect four skip connections")
        if len(config.decoder_features) != 4:
            raise ValueError("decoder_features must contain exactly four feature sizes")

        self.patch_embedding = PatchEmbedding(
            in_channels   = config.in_channels,
            embedding_dim = config.embedding_dim,
            patch_size    = config.patch_size,
        )

        self.positional_embedding = nn.Parameter(torch.zeros(1, (config.image_size // config.patch_size) ** 2, config.embedding_dim))
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        drop_path_rates = [x.item() for x in torch.linspace(0, config.stochastic_depth_rate, config.transformer_layers)]
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim     = config.embedding_dim,
                num_heads         = config.transformer_heads,
                mlp_ratio         = config.transformer_mlp_ratio,
                dropout           = config.dropout,
                attention_dropout = config.attention_dropout,
                ffn_activation    = config.ffn_activation,
                drop_path_rate    = drop_path_rates[i],
            )
            for i in range(config.transformer_layers)
        ])
        self.transformer_norm = nn.LayerNorm(config.embedding_dim)

        # Indices of transformer layers from which to extract skip connections (evenly spaced)
        total_layers = config.transformer_layers
        self.skip_layer_indices = [
            total_layers // 4 - 1,
            total_layers // 2 - 1,
            (3 * total_layers) // 4 - 1,
            total_layers - 1,
        ]

        decoder_features      = config.decoder_features
        # Projection heads: progressively upsample intermediate transformer features to match decoder scales
        self.transformer_skip_heads = nn.ModuleList([
            ProgressiveProjectionHead(
                input_channels  = config.embedding_dim,
                output_channels = decoder_features[-1],
                upsample_steps  = 3,
                activation      = config.activation,
                normalization   = config.normalization,
                bias            = config.conv_bias,
            ),
            ProgressiveProjectionHead(
                input_channels  = config.embedding_dim,
                output_channels = decoder_features[-2],
                upsample_steps  = 2,
                activation      = config.activation,
                normalization   = config.normalization,
                bias            = config.conv_bias,
            ),
            ProgressiveProjectionHead(
                input_channels  = config.embedding_dim,
                output_channels = decoder_features[-3],
                upsample_steps  = 1,
                activation      = config.activation,
                normalization   = config.normalization,
                bias            = config.conv_bias,
            ),
        ])
        self.bottleneck_projection = ProgressiveProjectionHead(
            input_channels  = config.embedding_dim,
            output_channels = decoder_features[0],
            upsample_steps  = 0,
            activation      = config.activation,
            normalization   = config.normalization,
            bias            = config.conv_bias,
        )

        # Separate conv to process original input as the finest-resolution skip connection
        self.input_skip_conv = ConvBlock(
            input_channels  = config.in_channels,
            output_channels = decoder_features[-1],
            dropout         = config.dropout,
            activation      = config.activation,
            normalization   = config.normalization,
            bias            = config.conv_bias,
        )
        
        self.upsample_layers = nn.ModuleList()
        self.decoder_blocks  = nn.ModuleList()

        upsample_output_channels = decoder_features[1:] + [decoder_features[-1]]
        skip_channels = [decoder_features[1], decoder_features[2], decoder_features[3], decoder_features[3]]

        for index, output_channels in enumerate(upsample_output_channels):
            input_to_upsample = decoder_features[index]

            self.upsample_layers.append(
                nn.ConvTranspose2d(
                    in_channels  = input_to_upsample,
                    out_channels = output_channels,
                    kernel_size  = 2,
                    stride       = 2,
                )
            )

            self.decoder_blocks.append(
                ConvBlock(
                    input_channels  = output_channels + skip_channels[index],
                    output_channels = output_channels,
                    dropout         = config.dropout,
                    activation      = config.activation,
                    normalization   = config.normalization,
                    bias            = config.conv_bias,
                )
            )

        remaining_scale = config.patch_size // (2 ** len(decoder_features))
        if remaining_scale > 1:
            self.final_upsample = nn.ConvTranspose2d(
                in_channels  = decoder_features[-1],
                out_channels = decoder_features[-1],
                kernel_size  = remaining_scale,
                stride       = remaining_scale,
            )
        else:
            self.final_upsample = nn.Identity()
        self.output_head = nn.Conv2d(
            in_channels  = decoder_features[-1],
            out_channels = config.out_channels,
            kernel_size  = 1,
        )

        initialize_weights(module=self, mode=config.init_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_input = x

        # Tokenize input image into patch embeddings + add positional encoding
        tokens, grid_height, grid_width = self.patch_embedding(x)
        tokens = tokens + self.positional_embedding[:, :tokens.shape[1], :]

        # Transformer encoder: collect intermediate skip features at evenly spaced layers
        transformer_skip_maps = []
        for layer_index, transformer_block in enumerate(self.transformer_blocks):
            tokens = transformer_block(tokens)
            if layer_index in self.skip_layer_indices:
                skip_tokens = self.transformer_norm(tokens) if layer_index == len(self.transformer_blocks) - 1 else tokens
                transformer_skip_maps.append(tokens_to_feature_map(skip_tokens, grid_height, grid_width))

        # Progressively upsample intermediate skip maps to match decoder resolution levels
        progressively_upsampled_skips = [
            skip_head(feature_map)
            for skip_head, feature_map in zip(self.transformer_skip_heads, transformer_skip_maps[:-1])
        ]
        progressively_upsampled_skips = progressively_upsampled_skips[::-1]

        # Project bottleneck (deepest transformer output) to decoder feature space
        x = self.bottleneck_projection(transformer_skip_maps[-1])

        # CNN decoder: upsample -> concat with skip -> ConvBlock at each level
        for index in range(len(self.upsample_layers)):
            x = self.upsample_layers[index](x)

            if index < len(progressively_upsampled_skips):
                skip = progressively_upsampled_skips[index]
            else:
                skip = self.input_skip_conv(original_input)

            x = match_spatial_size(source=x, reference=skip)
            x = torch.cat([x, skip], dim=1)
            x = self.decoder_blocks[index](x)

        x = self.final_upsample(x)
        x = match_spatial_size(source=x, reference=original_input)
        return self.output_head(x)
