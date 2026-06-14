from __future__ import annotations

import torch
import torch.nn as nn

from configuration.model.autoencoder_models_config import Conv1dAutoencoderConfig
from models.autoencoder.base import AutoencoderBase, AutoencoderBlocks
from models.blocks import build_activation


class Conv1dEncoder(nn.Module):
    def __init__(self, config: Conv1dAutoencoderConfig) -> None:
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


class Conv1dDecoder(nn.Module):
    def __init__(self, config: Conv1dAutoencoderConfig) -> None:
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


class Conv1dAutoencoder(AutoencoderBase):
    def __init__(self, config: Conv1dAutoencoderConfig | None = None) -> None:
        config = config if config is not None else Conv1dAutoencoderConfig()
        super().__init__(config, Conv1dEncoder(config), Conv1dDecoder(config))
