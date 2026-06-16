from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional



class EmbeddingNorm:
    EMBEDDING_NORMS = ("none", "l2", "layernorm")

    def normalize_embedding(self, z: torch.Tensor) -> torch.Tensor:
        kind = self.config.embedding_norm
        if kind == "none":
            return z
        if kind == "l2":
            return functional.normalize(z, dim=1, eps=1e-6)

        mean = z.mean(dim=1, keepdim=True)
        var  = z.var(dim=1, keepdim=True, unbiased=False)
        return (z - mean) / torch.sqrt(var + 1e-6)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob     = 1.0 - self.drop_prob
        shape         = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor_(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


def build_activation(name: str) -> nn.Module:
    factories = {
        "relu"        : lambda: nn.ReLU(inplace=True),
        "leaky_relu"  : lambda: nn.LeakyReLU(inplace=True),
        "gelu"        : lambda: nn.GELU(),
        "elu"         : lambda: nn.ELU(inplace=True),
        "silu"        : lambda: nn.SiLU(inplace=True),
    }
    if name not in factories:
        raise ValueError(f"Unknown activation '{name}'. Available: {list(factories.keys())}")
    return factories[name]()


def build_norm2d(name: str, num_features: int) -> nn.Module:
    if name == "batch":
        return nn.BatchNorm2d(num_features)
    if name == "instance":
        return nn.InstanceNorm2d(num_features, affine=True)
    if name == "group":
        num_groups = min(32, num_features)
        while num_features % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups, num_features)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"Unknown normalization '{name}'. Available: batch, instance, group, none")


def build_upsample(mode: str, in_channels: int, out_channels: int, scale_factor: int = 2) -> nn.Module:
    if mode == "convtranspose":
        return nn.ConvTranspose2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = scale_factor,
            stride       = scale_factor,
        )
    if mode == "bilinear":
        return nn.Sequential(
            nn.Upsample(
                scale_factor  = scale_factor,
                mode          = "bilinear",
                align_corners = False,
            ),
            nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = 1,
                bias         = False,
            ),
        )
    raise ValueError(f"Unknown upsample mode '{mode}'. Available: convtranspose, bilinear")


def initialize_weights(module: nn.Module, mode: str) -> None:
    if mode == "default":
        return
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if mode == "kaiming":
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif mode == "xavier":
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            if mode == "kaiming":
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif mode == "xavier":
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d, nn.LayerNorm)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)


def match_spatial_size(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if source.shape[2:] != reference.shape[2:]:
        return functional.interpolate(
            input         = source,
            size          = reference.shape[2:],
            mode          = "bilinear",
            align_corners = False,
        )
    return source


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


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        input_channels:  int,
        output_channels: int,
        dropout:         float = 0.0,
        activation:      str   = "relu",
        normalization:   str   = "batch",
        bias:            bool  = False,
        stride:          int   = 1,
        first_unit:      bool  = False,
    ):
        super().__init__()
        layers = []
        if not first_unit:
            layers.append(build_norm2d(normalization, input_channels))
            layers.append(build_activation(activation))

        layers.append(
            nn.Conv2d(
                in_channels  = input_channels,
                out_channels = output_channels,
                kernel_size  = 3,
                stride       = stride,
                padding      = 1,
                bias         = bias,
            )
        )
        layers.append(build_norm2d(normalization, output_channels))
        layers.append(build_activation(activation))
        layers.append(
            nn.Conv2d(
                in_channels  = output_channels,
                out_channels = output_channels,
                kernel_size  = 3,
                padding      = 1,
                bias         = bias,
            )
        )
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)

        if input_channels != output_channels or stride != 1:
            self.shortcut = nn.Conv2d(
                in_channels  = input_channels,
                out_channels = output_channels,
                kernel_size  = 1,
                stride       = stride,
                bias         = bias,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + self.shortcut(x)


class PixelMLP(nn.Module):
    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int,
        out_channels:    int,
        activation:      str = "relu",
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True),
            build_activation(activation),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GaussianHeadsMixin:
    def _resolve_gaussian_layout(self) -> None:
        n_params = self.config.params_per_gaussian
        if self.config.out_channels % n_params != 0:
            raise ValueError(f"out_channels ({self.config.out_channels}) must be divisible by params_per_gaussian ({n_params})")
        self.n_gaussians = self.config.out_channels // n_params
        self.n_params    = n_params

    def _build_triple_heads(self) -> None:
        self.head_amp   = PixelMLP(self.embedding_channels, self.hidden_channels, self.n_gaussians, self.config.activation)
        self.head_mu    = PixelMLP(self.embedding_channels, self.hidden_channels, self.n_gaussians, self.config.activation)
        self.head_sigma = PixelMLP(self.embedding_channels, self.hidden_channels, self.n_gaussians, self.config.activation)

    def _triple_head_forward(self, embedding: torch.Tensor) -> torch.Tensor:
        amp   = self.head_amp(embedding)
        mu    = self.head_mu(embedding)
        sigma = self.head_sigma(embedding)

        B, K, H, W = amp.shape
        out = torch.stack([amp, mu, sigma], dim=2)
        return out.view(B, K * 3, H, W)

    def _build_per_gaussian_heads(self) -> None:
        self.gaussian_heads = nn.ModuleList([
            PixelMLP(self.embedding_channels, self.hidden_channels, self.n_params, self.config.activation)
            for _ in range(self.n_gaussians)
        ])

    def _per_gaussian_forward(self, embedding: torch.Tensor) -> torch.Tensor:
        head_outputs = [head(embedding) for head in self.gaussian_heads]

        B, _, H, W = head_outputs[0].shape
        out = torch.stack(head_outputs, dim=1)
        return out.view(B, self.n_gaussians * self.n_params, H, W)


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        feature_sizes:  list[int],
        dropout:        float = 0.0,
        activation:     str   = "relu",
        normalization:  str   = "batch",
        bias:           bool  = False,
    ):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        channels = input_channels
        for feature_size in feature_sizes:
            self.conv_blocks.append(
                ConvBlock(
                    input_channels  = channels,
                    output_channels = feature_size,
                    dropout         = dropout,
                    activation      = activation,
                    normalization   = normalization,
                    bias            = bias,
                )
            )
            self.downsample_layers.append(nn.MaxPool2d(kernel_size=2))
            channels = feature_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skip_connections: list[torch.Tensor] = []
        for conv_block, downsample in zip(self.conv_blocks, self.downsample_layers):
            x = conv_block(x)
            skip_connections.append(x)
            x = downsample(x)
        return x, skip_connections


class Decoder(nn.Module):
    def __init__(
        self,
        feature_sizes: list[int],
        dropout:       float = 0.0,
        activation:    str   = "relu",
        normalization: str   = "batch",
        bias:          bool  = False,
        upsample_mode: str   = "convtranspose",
    ):
        super().__init__()
        self.upsample_layers = nn.ModuleList()
        self.conv_blocks     = nn.ModuleList()
        for index in range(len(feature_sizes) - 1):
            self.upsample_layers.append(
                build_upsample(
                    mode         = upsample_mode,
                    in_channels  = feature_sizes[index],
                    out_channels = feature_sizes[index + 1],
                )
            )
            self.conv_blocks.append(
                ConvBlock(
                    input_channels  = feature_sizes[index + 1] * 2,
                    output_channels = feature_sizes[index + 1],
                    dropout         = dropout,
                    activation      = activation,
                    normalization   = normalization,
                    bias            = bias,
                )
            )

    def forward(self, x: torch.Tensor, skip_connections: list[torch.Tensor]) -> torch.Tensor:
        for upsample, conv_block, skip in zip(self.upsample_layers, self.conv_blocks, skip_connections):
            x = upsample(x)
            x = match_spatial_size(source=x, reference=skip)
            x = torch.cat([skip, x], dim=1)
            x = conv_block(x)
        return x


def scaled_dot_product(query, key, value, scale, attention_dropout, attention_bias=None):
    attention_weights = (query @ key.transpose(-2, -1)) * scale

    if attention_bias is not None:
        attention_weights = attention_weights + attention_bias

    attention_weights = attention_weights.softmax(dim=-1)
    attention_weights = attention_dropout(attention_weights)

    return attention_weights @ value


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

        attended  = scaled_dot_product(query, key, value, self.scale, self.attention_dropout).transpose(1, 2)
        attended  = attended.reshape(batch_size, sequence_length, embedding_dim)
        projected = self.output_projection(attended)
        return self.output_dropout(projected)


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
        x          = x + self.drop_path(self.attention(normalized))
        normalized = self.norm_2(x)
        x          = x + self.drop_path(self.feed_forward(normalized))
        return x


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


def tokens_to_feature_map(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch_size, _, channels = tokens.shape
    return tokens.transpose(1, 2).view(batch_size, channels, height, width)
