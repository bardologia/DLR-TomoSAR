from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UNetConfig:
    in_channels: int = 1
    out_channels: int = 6
    features: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor: int = 2
    dropout: float = 0.0


@dataclass
class ResUNetConfig:
    in_channels: int = 1
    out_channels: int = 6
    features: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor: int = 2
    dropout: float = 0.0


@dataclass
class AttentionUNetConfig:
    in_channels: int = 1
    out_channels: int = 6
    features: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor: int = 2
    dropout: float = 0.0
    attention_intermediate_ratio: float = 0.5


@dataclass
class UNetPlusPlusConfig:
    in_channels: int = 1
    out_channels: int = 6
    features: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor: int = 2
    dropout: float = 0.0
    deep_supervision: bool = False


@dataclass
class FCNConfig:
    in_channels: int = 1
    out_channels: int = 6
    features: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor: int = 2
    dropout: float = 0.0


@dataclass
class LinkNetConfig:
    in_channels: int = 1
    out_channels: int = 6
    features: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    dropout: float = 0.0
    initial_kernel_size: int = 7
    decoder_bottleneck_ratio: int = 4


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


@dataclass
class TransUNetConfig:
    in_channels: int = 1
    out_channels: int = 6
    cnn_features: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor: int = 2
    transformer_layers: int = 12
    transformer_heads: int = 8
    transformer_mlp_ratio: float = 4.0
    patch_size: int = 1
    dropout: float = 0.0


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
