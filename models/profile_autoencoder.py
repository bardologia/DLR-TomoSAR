from __future__ import annotations

import torch
import torch.nn as nn

from configuration.jepa_config import ProfileAutoencoderConfig
from .blocks import build_activation, initialize_weights


class ProfileBlocks:
    @staticmethod
    def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
        return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

    @staticmethod
    def mlp_stack(in_ch: int, hidden: int, out_ch: int, depth: int, activation: str, dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = in_ch

        for _ in range(max(1, depth)):
            layers.append(ProfileBlocks.conv1x1(prev, hidden))
            layers.append(build_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            prev = hidden

        layers.append(ProfileBlocks.conv1x1(prev, out_ch))
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


class MlpProfileEncoder(nn.Module):
    def __init__(self, config: ProfileAutoencoderConfig) -> None:
        super().__init__()
        self.net = ProfileBlocks.mlp_stack(
            in_ch      = config.profile_length,
            hidden     = config.hidden_dim,
            out_ch     = config.embedding_dim,
            depth      = config.depth,
            activation = config.activation,
            dropout    = config.dropout,
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        return self.net(curve)


class Conv1dProfileEncoder(nn.Module):
    def __init__(self, config: ProfileAutoencoderConfig) -> None:
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
        seq, dims = ProfileBlocks.to_sequence(curve)
        feats     = self.body(seq.unsqueeze(1))
        z         = self.head(feats.squeeze(-1))
        return ProfileBlocks.from_sequence(z, dims)


class Transformer1dProfileEncoder(nn.Module):
    def __init__(self, config: ProfileAutoencoderConfig) -> None:
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
        seq, dims = ProfileBlocks.to_sequence(curve)
        tokens    = self.embed(seq).unsqueeze(1)
        attended  = self.encoder(tokens).squeeze(1)
        z         = self.head(attended)
        return ProfileBlocks.from_sequence(z, dims)


class MlpProfileDecoder(nn.Module):
    def __init__(self, config: ProfileAutoencoderConfig) -> None:
        super().__init__()
        self.net = ProfileBlocks.mlp_stack(
            in_ch      = config.embedding_dim,
            hidden     = config.hidden_dim,
            out_ch     = config.profile_length,
            depth      = config.depth,
            activation = config.activation,
            dropout    = config.dropout,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Conv1dProfileDecoder(nn.Module):
    def __init__(self, config: ProfileAutoencoderConfig) -> None:
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
        seq, dims = ProfileBlocks.to_sequence(z)
        feats     = self.project(seq).reshape(-1, self.seq_channels, self.length)
        curve     = self.body(feats).squeeze(1)
        return ProfileBlocks.from_sequence(curve, dims)


class Transformer1dProfileDecoder(MlpProfileDecoder):
    pass


PROFILE_ENCODER_REGISTRY = {
    "mlp"           : MlpProfileEncoder,
    "conv1d"        : Conv1dProfileEncoder,
    "transformer1d" : Transformer1dProfileEncoder,
}

PROFILE_DECODER_REGISTRY = {
    "mlp"           : MlpProfileDecoder,
    "conv1d"        : Conv1dProfileDecoder,
    "transformer1d" : Transformer1dProfileDecoder,
}


class ProfileParamHeads(nn.Module):
    def __init__(self, config: ProfileAutoencoderConfig) -> None:
        super().__init__()
        self.config      = config
        self.n_gaussians = config.n_gaussians
        self.ppg         = config.params_per_gaussian
        self.strategy    = config.count_strategy

        self.in_ch = config.embedding_dim
        if config.embedding_structure == "structured":
            self.in_ch = min(config.embedding_dim, max(self.n_gaussians * self.ppg, 1))

        self.param_head = ProfileBlocks.mlp_stack(
            in_ch      = self.in_ch,
            hidden     = config.hidden_dim,
            out_ch     = config.out_channels,
            depth      = 1,
            activation = config.activation,
            dropout    = config.dropout,
        )

        self.presence_head = None
        self.count_head    = None
        if self.strategy == "presence_logit":
            self.presence_head = ProfileBlocks.conv1x1(self.in_ch, self.n_gaussians)
        elif self.strategy == "count_head":
            self.count_head = ProfileBlocks.conv1x1(self.in_ch, 1)

        self._extra: dict[str, torch.Tensor] = {}

    def _param_slice(self, z: torch.Tensor) -> torch.Tensor:
        if self.config.embedding_structure == "structured":
            return z[:, : self.in_ch]
        return z

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        zp     = self._param_slice(z)
        params = self.param_head(zp)
        self._extra = {}

        if self.presence_head is not None:
            logits  = self.presence_head(zp)
            self._extra["presence_logits"] = logits
            gate    = torch.sigmoid(logits)
            B       = params.shape[0]
            spatial = params.shape[2:]
            p       = params.view(B, self.n_gaussians, self.ppg, *spatial)
            p       = torch.cat([p[:, :, :1] * gate.view(B, self.n_gaussians, 1, *spatial), p[:, :, 1:]], dim=2)
            params  = p.view(B, self.n_gaussians * self.ppg, *spatial)

        if self.count_head is not None:
            self._extra["count"] = self.count_head(zp)

        return params

    def extra(self) -> dict[str, torch.Tensor]:
        return self._extra


class ProfileAutoencoder(nn.Module):
    def __init__(self, config: ProfileAutoencoderConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else ProfileAutoencoderConfig()

        if self.config.encoder_kind not in PROFILE_ENCODER_REGISTRY:
            raise ValueError(f"Unknown encoder_kind '{self.config.encoder_kind}'. Available: {list(PROFILE_ENCODER_REGISTRY)}")
        if self.config.decoder_kind not in PROFILE_DECODER_REGISTRY:
            raise ValueError(f"Unknown decoder_kind '{self.config.decoder_kind}'. Available: {list(PROFILE_DECODER_REGISTRY)}")

        self.encoder     = PROFILE_ENCODER_REGISTRY[self.config.encoder_kind](self.config)
        self.decoder     = PROFILE_DECODER_REGISTRY[self.config.decoder_kind](self.config)
        self.param_heads = ProfileParamHeads(self.config)

        initialize_weights(module=self, mode=self.config.init_mode)

    def encode(self, curve: torch.Tensor) -> torch.Tensor:
        return self.encoder(curve)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def heads(self, z: torch.Tensor) -> torch.Tensor:
        return self.param_heads(z)

    def reconstruct(self, curve: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z         = self.encode(curve)
        curve_hat = self.decode(z)
        params    = self.heads(z)
        return params, curve_hat, z

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        params, _, _ = self.reconstruct(curve)
        return params
