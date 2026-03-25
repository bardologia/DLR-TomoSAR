"""
Gate-first dense Mixture-of-Experts for pixel-level prediction.

Pipeline: gating -> routing -> conditional expert execution -> aggregation.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AggregationMode, MoEConfig
from .experts import apply_training_mode, build_experts
from .gating import build_gating


class MoEOutput(NamedTuple):
    prediction     : torch.Tensor              # (B, C_max, H, W)
    gate_probs     : torch.Tensor              # (B, K, H, W)
    expert_outs    : dict[int, torch.Tensor]   # {k: (B, C_k, H, W)} active only
    expert_mask    : torch.Tensor              # (B, K, H, W)
    active_experts : list[int]


class DenseMoE(nn.Module):
    """Gate-first dense Mixture-of-Experts.

    The gating network evaluates the input first, decides per-pixel
    routing, then only the selected experts process the image.
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config

        self.gating  : nn.Module     = build_gating(config.in_channels, config.num_experts, config.gating)
        self.experts : nn.ModuleList = build_experts(config)

        apply_training_mode(self.experts, self.gating, config.training_mode)

    # ── routing ──────────────────────────────────────────────────────────

    def _route(self, gate_probs: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        """Return (mask, active_indices) based on aggregation mode."""
        agg = self.config.aggregation

        if agg == AggregationMode.soft:
            mask   = torch.ones_like(gate_probs)
            active = list(range(self.config.num_experts))
            return mask, active

        if agg == AggregationMode.hard:
            winner = gate_probs.argmax(dim=1)
            onehot = F.one_hot(winner, self.config.num_experts)
            mask   = onehot.permute(0, 3, 1, 2).float()
            active = sorted(winner.unique().tolist())
            return mask, active

        if agg == AggregationMode.top_k:
            k          = min(self.config.top_k, self.config.num_experts)
            _, top_idx = gate_probs.topk(k, dim=1)
            mask       = torch.zeros_like(gate_probs)
            mask.scatter_(1, top_idx, 1.0)
            active     = sorted(top_idx.unique().tolist())
            return mask, active

        raise ValueError(f"Unknown aggregation mode: {agg}")

    # ── conditional execution ────────────────────────────────────────────

    def _run_active_experts(self, x: torch.Tensor, active: list[int]) -> dict[int, torch.Tensor]:
        return {k: self.experts[k](x) for k in active}

    # ── padding ──────────────────────────────────────────────────────────

    def _pad_to_max(self, out: torch.Tensor, target_ch: int) -> torch.Tensor:
        deficit = target_ch - out.shape[1]
        if deficit <= 0:
            return out
        pad = out.new_full((out.shape[0], deficit, out.shape[2], out.shape[3]), self.config.pad_value)
        return torch.cat([out, pad], dim=1)

    # ── aggregation ──────────────────────────────────────────────────────

    def _aggregate(
        self,
        expert_outs : dict[int, torch.Tensor],
        gate_probs  : torch.Tensor,
        mask        : torch.Tensor,
    ) -> torch.Tensor:
        b, k, h, w = gate_probs.shape
        c          = self.config.max_out_channels
        agg        = self.config.aggregation

        # build (B, K, C, H, W) — zeros for inactive experts
        padded = gate_probs.new_zeros(b, k, c, h, w)
        for idx, out_k in expert_outs.items():
            padded[:, idx] = self._pad_to_max(out_k, c)

        if agg == AggregationMode.soft:
            weights = gate_probs.unsqueeze(2)
            return (padded * weights).sum(dim=1)

        if agg == AggregationMode.hard:
            winner   = gate_probs.argmax(dim=1)
            gather_i = winner.unsqueeze(1).unsqueeze(2).expand(-1, 1, c, -1, -1)
            return padded.gather(1, gather_i).squeeze(1)

        if agg == AggregationMode.top_k:
            num_k              = min(self.config.top_k, self.config.num_experts)
            topk_vals, topk_i  = gate_probs.topk(num_k, dim=1)
            topk_w             = topk_vals / (topk_vals.sum(1, keepdim=True) + 1e-8)
            gather_i           = topk_i.unsqueeze(2).expand(-1, -1, c, -1, -1)
            selected           = padded.gather(1, gather_i)
            return (selected * topk_w.unsqueeze(2)).sum(dim=1)

        raise ValueError(f"Unknown aggregation mode: {agg}")

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> MoEOutput:
        gate_probs          = self.gating(x)
        mask, active        = self._route(gate_probs)
        expert_outs         = self._run_active_experts(x, active)
        prediction          = self._aggregate(expert_outs, gate_probs, mask)

        return MoEOutput(
            prediction     = prediction,
            gate_probs     = gate_probs,
            expert_outs    = expert_outs,
            expert_mask    = mask,
            active_experts = active,
        )

    # ── convenience ──────────────────────────────────────────────────────

    def get_expert_assignment(self, x: torch.Tensor) -> torch.Tensor:
        """Per-pixel hard expert index map (B, H, W)."""
        with torch.no_grad():
            probs = self.gating(x)
        return probs.argmax(dim=1)

    def set_aggregation(self, mode: AggregationMode) -> None:
        self.config.aggregation = mode

    def set_temperature(self, temperature: float) -> None:
        self.gating.temperature = temperature
