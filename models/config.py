from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


# Randomly drops entire residual branches during training for regularization
class DropPath(nn.Module):
    """Stochastic depth / drop path regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor_(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


# Factory: maps string name to the corresponding activation function
def build_activation(name: str) -> nn.Module:
    """Create an activation module from a string name."""
    factories = {
        "relu": lambda: nn.ReLU(inplace=True),
        "leaky_relu": lambda: nn.LeakyReLU(inplace=True),
        "gelu": lambda: nn.GELU(),
        "elu": lambda: nn.ELU(inplace=True),
        "silu": lambda: nn.SiLU(inplace=True),
    }
    if name not in factories:
        raise ValueError(f"Unknown activation '{name}'. Available: {list(factories.keys())}")
    return factories[name]()


# Factory: maps string name to the corresponding 2D normalization layer
def build_norm2d(name: str, num_features: int) -> nn.Module:
    """Create a 2D normalization layer from a string name."""
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


# Factory: creates upsampling layer (learned ConvTranspose or bilinear + 1x1 conv)
def build_upsample(
    mode:         str,
    in_channels:  int,
    out_channels: int,
    scale_factor: int = 2,
) -> nn.Module:
    """Create an upsampling module."""
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


# Applies Kaiming or Xavier initialization to Conv, Linear, and Norm layers
def initialize_weights(module: nn.Module, mode: str) -> None:
    """Apply weight initialization to all layers of a module."""
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


# ----- Model configuration dataclasses -----

@dataclass
class UNetConfig:
    in_channels: int = 1
    out_channels: int = 6
    features: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor: int = 2
    dropout: float = 0.0
    activation: str = "relu"
    normalization: str = "batch"
    upsample_mode: str = "convtranspose"
    conv_bias: bool = False
    init_mode: str = "default"


@dataclass
class ResUNetConfig:
    in_channels: int = 1
    out_channels: int = 6
    features: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor: int = 2
    dropout: float = 0.0
    activation: str = "relu"
    normalization: str = "batch"
    upsample_mode: str = "convtranspose"
    conv_bias: bool = False
    init_mode: str = "default"


@dataclass
class AttentionUNetConfig:
    in_channels: int = 1
    out_channels: int = 6
    features: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor: int = 2
    dropout: float = 0.0
    attention_intermediate_ratio: float = 0.5
    activation: str = "relu"
    normalization: str = "batch"
    upsample_mode: str = "convtranspose"
    conv_bias: bool = False
    init_mode: str = "default"


@dataclass
class UNetPlusPlusConfig:
    in_channels: int = 1
    out_channels: int = 6
    features: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor: int = 2
    dropout: float = 0.0
    deep_supervision: bool = False
    activation: str = "relu"
    normalization: str = "batch"
    upsample_mode: str = "convtranspose"
    conv_bias: bool = False
    init_mode: str = "default"


@dataclass
class FCNConfig:
    in_channels: int = 1
    out_channels: int = 6
    features: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor: int = 2
    variant: str = "8s"
    dropout: float = 0.0
    activation: str = "relu"
    normalization: str = "batch"
    conv_bias: bool = False
    init_mode: str = "default"


@dataclass
class LinkNetConfig:
    in_channels: int = 1
    out_channels: int = 6
    features: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    dropout: float = 0.0
    initial_kernel_size: int = 7
    decoder_bottleneck_ratio: int = 4
    activation: str = "relu"
    normalization: str = "batch"
    conv_bias: bool = False
    init_mode: str = "default"


@dataclass
class SwinUNetConfig:
    in_channels: int = 1
    out_channels: int = 6
    image_size: int = 256
    patch_size: int = 4
    embedding_dim: int = 96
    depths: list[int] = field(default_factory=lambda: [2, 2, 6, 2])
    num_heads: list[int] = field(default_factory=lambda: [3, 6, 12, 24])
    window_size: int = 7
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    ffn_activation: str = "gelu"
    stochastic_depth_rate: float = 0.0
    init_mode: str = "default"


@dataclass
class TransUNetConfig:
    in_channels: int = 1
    out_channels: int = 6
    image_size: int = 256
    cnn_features: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor: int = 2
    transformer_layers: int = 12
    transformer_heads: int = 8
    transformer_mlp_ratio: float = 4.0
    patch_size: int = 1
    dropout: float = 0.0
    activation: str = "relu"
    normalization: str = "batch"
    upsample_mode: str = "convtranspose"
    conv_bias: bool = False
    attention_dropout: float = 0.0
    ffn_activation: str = "gelu"
    stochastic_depth_rate: float = 0.0
    init_mode: str = "default"


@dataclass
class UNETRConfig:
    in_channels: int = 1
    out_channels: int = 6
    image_size: int = 256
    patch_size: int = 16
    embedding_dim: int = 768
    transformer_layers: int = 12
    transformer_heads: int = 12
    transformer_mlp_ratio: float = 4.0
    decoder_features: list[int] = field(default_factory=lambda: [512, 256, 128, 64])
    dropout: float = 0.0
    activation: str = "relu"
    normalization: str = "batch"
    conv_bias: bool = False
    attention_dropout: float = 0.0
    ffn_activation: str = "gelu"
    stochastic_depth_rate: float = 0.0
    init_mode: str = "default"
