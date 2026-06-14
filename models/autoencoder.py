from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from configuration.model.models_config import AutoencoderConfig
from .blocks import build_activation, initialize_weights


class AutoencoderBlocks:
    @staticmethod
    def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
        return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

    @staticmethod
    def mlp_stack(in_ch: int, hidden: int, out_ch: int, depth: int, activation: str, dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = in_ch

        for _ in range(max(1, depth)):
            layers.append(AutoencoderBlocks.conv1x1(prev, hidden))
            layers.append(build_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            prev = hidden

        layers.append(AutoencoderBlocks.conv1x1(prev, out_ch))
        return nn.Sequential(*layers)

    @staticmethod
    def to_sequence(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        B, L, H, W = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(B * H * W, L)
        return seq, (B, H, W)

    @staticmethod
    def from_sequence(seq: torch.Tensor, dims: tuple[int, int, int]) -> torch.Tensor:
        B, H, W = dims
        L_out   = seq.shape[1]
        return seq.reshape(B, H, W, L_out).permute(0, 3, 1, 2).contiguous()


class MlpEncoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.net = AutoencoderBlocks.mlp_stack(
            in_ch      = config.profile_length,
            hidden     = config.hidden_dim,
            out_ch     = config.embedding_dim,
            depth      = config.depth,
            activation = config.activation,
            dropout    = config.dropout,
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        return self.net(curve)


class Conv1dEncoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        pad = config.seq_kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv1d(1, config.seq_channels, config.seq_kernel_size, padding=pad),
            build_activation(config.activation),
            nn.Conv1d(config.seq_channels, config.seq_channels, config.seq_kernel_size, padding=pad),
            build_activation(config.activation),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(config.seq_channels, config.embedding_dim)

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        seq, dims = AutoencoderBlocks.to_sequence(curve)
        feats     = self.body(seq.unsqueeze(1))
        z         = self.head(feats.squeeze(-1))
        return AutoencoderBlocks.from_sequence(z, dims)


class Transformer1dEncoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.embed = nn.Linear(config.profile_length, config.hidden_dim)
        layer      = nn.TransformerEncoderLayer(
            d_model         = config.hidden_dim,
            nhead           = config.num_heads,
            dim_feedforward = config.hidden_dim * 2,
            dropout         = config.dropout,
            activation      = config.activation if config.activation in ("relu", "gelu") else "gelu",
            batch_first     = True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, config.depth))
        self.head    = nn.Linear(config.hidden_dim, config.embedding_dim)

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        seq, dims = AutoencoderBlocks.to_sequence(curve)
        tokens    = self.embed(seq).unsqueeze(1)
        attended  = self.encoder(tokens).squeeze(1)
        z         = self.head(attended)
        return AutoencoderBlocks.from_sequence(z, dims)


class MlpDecoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.net = AutoencoderBlocks.mlp_stack(
            in_ch      = config.embedding_dim,
            hidden     = config.hidden_dim,
            out_ch     = config.profile_length,
            depth      = config.depth,
            activation = config.activation,
            dropout    = config.dropout,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Conv1dDecoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.length       = config.profile_length
        self.seq_channels = config.seq_channels
        pad               = config.seq_kernel_size // 2
        self.project      = nn.Linear(config.embedding_dim, config.seq_channels * config.profile_length)
        self.body         = nn.Sequential(
            nn.Conv1d(config.seq_channels, config.seq_channels, config.seq_kernel_size, padding=pad),
            build_activation(config.activation),
            nn.Conv1d(config.seq_channels, 1, config.seq_kernel_size, padding=pad),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        seq, dims = AutoencoderBlocks.to_sequence(z)
        feats     = self.project(seq).reshape(-1, self.seq_channels, self.length)
        curve     = self.body(feats).squeeze(1)
        return AutoencoderBlocks.from_sequence(curve, dims)


class Transformer1dDecoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.embed = nn.Linear(config.embedding_dim, config.hidden_dim)
        layer      = nn.TransformerEncoderLayer(
            d_model         = config.hidden_dim,
            nhead           = config.num_heads,
            dim_feedforward = config.hidden_dim * 2,
            dropout         = config.dropout,
            activation      = config.activation if config.activation in ("relu", "gelu") else "gelu",
            batch_first     = True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=max(1, config.depth))
        self.head    = nn.Linear(config.hidden_dim, config.profile_length)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        seq, dims = AutoencoderBlocks.to_sequence(z)
        tokens    = self.embed(seq).unsqueeze(1)
        attended  = self.decoder(tokens).squeeze(1)
        curve     = self.head(attended)
        return AutoencoderBlocks.from_sequence(curve, dims)


ENCODER_REGISTRY = {
    "mlp"           : MlpEncoder,
    "conv1d"        : Conv1dEncoder,
    "transformer1d" : Transformer1dEncoder,
}

DECODER_REGISTRY = {
    "mlp"           : MlpDecoder,
    "conv1d"        : Conv1dDecoder,
    "transformer1d" : Transformer1dDecoder,
}


_EMBEDDING_NORMS = ("none", "l2", "layernorm")
_CURVE_NORMS     = ("none", "log1p")


class Autoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else AutoencoderConfig()

        if self.config.encoder_kind not in ENCODER_REGISTRY:
            raise ValueError(f"Unknown encoder_kind '{self.config.encoder_kind}'. Available: {list(ENCODER_REGISTRY)}")
        if self.config.decoder_kind not in DECODER_REGISTRY:
            raise ValueError(f"Unknown decoder_kind '{self.config.decoder_kind}'. Available: {list(DECODER_REGISTRY)}")
        if self.config.embedding_norm not in _EMBEDDING_NORMS:
            raise ValueError(f"Unknown embedding_norm '{self.config.embedding_norm}'. Available: {list(_EMBEDDING_NORMS)}")
        if self.config.curve_norm not in _CURVE_NORMS:
            raise ValueError(f"Unknown curve_norm '{self.config.curve_norm}'. Available: {list(_CURVE_NORMS)}")

        self.encoder = ENCODER_REGISTRY[self.config.encoder_kind](self.config)
        self.decoder = DECODER_REGISTRY[self.config.decoder_kind](self.config)

        initialize_weights(module=self, mode=self.config.init_mode)

    def normalize_curve(self, curve: torch.Tensor) -> torch.Tensor:
        if self.config.curve_norm == "none":
            return curve
        return torch.log1p(curve.clamp(min=0.0))

    def denormalize_curve(self, curve: torch.Tensor) -> torch.Tensor:
        if self.config.curve_norm == "none":
            return curve
        return torch.expm1(curve)

    def normalize_embedding(self, z: torch.Tensor) -> torch.Tensor:
        kind = self.config.embedding_norm
        if kind == "none":
            return z
        if kind == "l2":
            return F.normalize(z, dim=1, eps=1e-6)

        mean = z.mean(dim=1, keepdim=True)
        var  = z.var(dim=1, keepdim=True, unbiased=False)
        return (z - mean) / torch.sqrt(var + 1e-6)

    def encode(self, curve: torch.Tensor) -> torch.Tensor:
        return self.normalize_embedding(self.encoder(self.normalize_curve(curve)))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reconstruct(self, curve: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z         = self.encode(curve)
        curve_hat = self.decode(z)
        return curve_hat, z

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        curve_hat, _ = self.reconstruct(curve)
        return curve_hat
