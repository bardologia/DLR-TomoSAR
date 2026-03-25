"""Gating (selector) networks for the dense MoE framework."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import build_activation, build_norm2d

from .config import GatingConfig, GatingType


class GatingBase(nn.Module):
    """Base class: temperature-scaled softmax over expert dim."""

    def __init__(self, num_experts: int, temperature: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature

    def to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits / self.temperature, dim=1)


class LightweightCNNGating(GatingBase):
    """Shallow conv encoder + bilinear upsample + 1x1 head."""

    def __init__(self, in_channels: int, num_experts: int, config: GatingConfig):
        super().__init__(num_experts, config.temperature)

        layers: list[nn.Module] = []
        ch = in_channels
        for feat in config.features:
            layers += [
                nn.Conv2d(ch, feat, kernel_size=3, padding=1, bias=False),
                build_norm2d(config.normalization, feat),
                build_activation(config.activation),
                nn.MaxPool2d(2),
            ]
            ch = feat

        if config.dropout > 0:
            layers.append(nn.Dropout2d(config.dropout))

        self.encoder = nn.Sequential(*layers)
        self.head    = nn.Conv2d(ch, num_experts, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w   = x.shape[2:]
        feat   = self.encoder(x)
        logits = self.head(feat)
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return self.to_probs(logits)


class EncoderOnlyGating(GatingBase):
    """Deeper multi-scale encoder + progressive upsample + 1x1 head."""

    def __init__(self, in_channels: int, num_experts: int, config: GatingConfig):
        super().__init__(num_experts, config.temperature)

        down_blocks: list[nn.Module] = []
        up_blocks  : list[nn.Module] = []

        ch = in_channels
        for feat in config.features:
            down_blocks.append(nn.Sequential(
                nn.Conv2d(ch, feat, kernel_size=3, padding=1, bias=False),
                build_norm2d(config.normalization, feat),
                build_activation(config.activation),
                nn.Conv2d(feat, feat, kernel_size=3, padding=1, bias=False),
                build_norm2d(config.normalization, feat),
                build_activation(config.activation),
                nn.MaxPool2d(2),
            ))
            ch = feat

        self.down_blocks = nn.ModuleList(down_blocks)

        rev = list(reversed(config.features))
        for i in range(len(rev) - 1):
            up_blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(rev[i], rev[i + 1], kernel_size=3, padding=1, bias=False),
                build_norm2d(config.normalization, rev[i + 1]),
                build_activation(config.activation),
            ))

        self.up_blocks = nn.ModuleList(up_blocks)

        out_feat       = rev[-1] if rev else in_channels
        self.final_up  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.head      = nn.Conv2d(out_feat, num_experts, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        out  = x

        for block in self.down_blocks:
            out = block(out)
        for block in self.up_blocks:
            out = block(out)

        out    = self.final_up(out)
        logits = self.head(out)
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return self.to_probs(logits)


class LinearProbeGating(GatingBase):
    """Single 1x1 conv baseline."""

    def __init__(self, in_channels: int, num_experts: int, config: GatingConfig):
        super().__init__(num_experts, config.temperature)
        self.head = nn.Conv2d(in_channels, num_experts, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_probs(self.head(x))


# ── factory ──────────────────────────────────────────────────────────────────

gating_registry: dict[GatingType, type[GatingBase]] = {
    GatingType.lightweight_cnn : LightweightCNNGating,
    GatingType.encoder_only    : EncoderOnlyGating,
    GatingType.linear_probe    : LinearProbeGating,
}


def build_gating(in_channels: int, num_experts: int, config: GatingConfig) -> GatingBase:
    cls = gating_registry.get(config.gating_type)
    if cls is None:
        raise ValueError(
            f"Unknown gating type '{config.gating_type}'. "
            f"Available: {list(gating_registry.keys())}"
        )
    return cls(in_channels, num_experts, config)
