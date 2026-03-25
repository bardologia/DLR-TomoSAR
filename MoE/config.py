"""Configuration for the dense Mixture-of-Experts framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── enums ────────────────────────────────────────────────────────────────────

class TrainingMode(Enum):
    end_to_end   = "end_to_end"
    gate_only    = "gate_only"
    experts_only = "experts_only"


class GatingType(Enum):
    lightweight_cnn = "lightweight_cnn"
    encoder_only    = "encoder_only"
    linear_probe    = "linear_probe"


class AggregationMode(Enum):
    soft  = "soft"
    hard  = "hard"
    top_k = "top_k"


# ── expert definition ───────────────────────────────────────────────────────

@dataclass
class ExpertDefinition:
    out_channels    : int             = 3
    backbone        : str             = "unet"
    backbone_config : dict[str, Any]  = field(default_factory=dict)
    pretrained_path : str | None      = None


# ── gating config ────────────────────────────────────────────────────────────

@dataclass
class GatingConfig:
    gating_type   : GatingType = GatingType.lightweight_cnn
    features      : list[int]  = field(default_factory=lambda: [32, 64, 128])
    dropout       : float      = 0.0
    activation    : str        = "relu"
    normalization : str        = "batch"
    temperature   : float      = 1.0


# ── loss config ──────────────────────────────────────────────────────────────

@dataclass
class MoELossConfig:
    reconstruction_weight : float = 1.0
    load_balance_weight   : float = 0.01
    entropy_weight        : float = 0.01
    reconstruction_loss   : str   = "mse"


# ── top-level config ─────────────────────────────────────────────────────────

@dataclass
class MoEConfig:
    in_channels   : int                   = 1
    experts       : list[ExpertDefinition] = field(default_factory=lambda: [
        ExpertDefinition(out_channels=3),
        ExpertDefinition(out_channels=6),
        ExpertDefinition(out_channels=9),
    ])
    gating        : GatingConfig          = field(default_factory=GatingConfig)
    loss          : MoELossConfig         = field(default_factory=MoELossConfig)
    training_mode : TrainingMode          = TrainingMode.end_to_end
    aggregation   : AggregationMode       = AggregationMode.soft
    top_k         : int                   = 1
    init_mode     : str                   = "default"
    pad_value     : float                 = 0.0

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    @property
    def expert_out_channels(self) -> list[int]:
        return [e.out_channels for e in self.experts]

    @property
    def max_out_channels(self) -> int:
        return max(self.expert_out_channels)
