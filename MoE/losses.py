"""
Composite loss: reconstruction + load-balance + gating entropy.

L = w_r * L_recon  +  w_b * L_balance  +  w_e * L_entropy
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import MoELossConfig
from .moe_model import MoEOutput


# ── reconstruction loss registry ─────────────────────────────────────────────

recon_losses: dict[str, type[nn.Module]] = {
    "mse"       : nn.MSELoss,
    "l1"        : nn.L1Loss,
    "smooth_l1" : nn.SmoothL1Loss,
}


def _build_recon_loss(name: str) -> nn.Module:
    cls = recon_losses.get(name)
    if cls is None:
        raise ValueError(f"Unknown reconstruction loss '{name}'. Available: {list(recon_losses.keys())}")
    return cls()


# ── regularisers ─────────────────────────────────────────────────────────────

def load_balance_loss(gate_probs: torch.Tensor) -> torch.Tensor:
    """CV-squared of per-expert mean probability. Balanced gate -> 0."""
    mean_prob = gate_probs.mean(dim=(0, 2, 3))
    return mean_prob.var() / (mean_prob.mean() ** 2 + 1e-8)


def gating_entropy_loss(gate_probs: torch.Tensor) -> torch.Tensor:
    """Mean per-pixel entropy of the gating distribution."""
    log_probs = torch.log(gate_probs.clamp(min=1e-8))
    entropy   = -(gate_probs * log_probs).sum(dim=1)
    return entropy.mean()


# ── per-expert weighted reconstruction ───────────────────────────────────────

def per_expert_reconstruction_loss(
    expert_outs : dict[int, torch.Tensor],
    target      : torch.Tensor,
    gate_probs  : torch.Tensor,
) -> torch.Tensor:
    """Gate-weighted MSE evaluated per active expert on matching channels."""
    total = target.new_tensor(0.0)

    for k, out_k in expert_outs.items():
        c_k        = out_k.shape[1]
        target_k   = target[:, :c_k, :, :]
        pixel_loss = (out_k - target_k).pow(2).mean(dim=1)
        weight_k   = gate_probs[:, k, :, :]
        total      = total + (pixel_loss * weight_k).mean()

    return total


# ── composite loss ───────────────────────────────────────────────────────────

class MoELoss(nn.Module):
    """Weighted sum of reconstruction, load-balance, and entropy terms."""

    def __init__(
        self,
        config           : MoELossConfig,
        max_out_channels : int,
        use_per_expert_loss : bool = True,
    ):
        super().__init__()
        self.config             = config
        self.max_out_channels   = max_out_channels
        self.use_per_expert_loss = use_per_expert_loss
        self.recon_loss_fn      = _build_recon_loss(config.reconstruction_loss)

    def forward(self, moe_output: MoEOutput, target: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction    = moe_output.prediction
        gate_probs    = moe_output.gate_probs
        target_matched = target[:, :self.max_out_channels, :, :]

        # reconstruction
        if self.use_per_expert_loss:
            recon = per_expert_reconstruction_loss(moe_output.expert_outs, target_matched, gate_probs)
        else:
            recon = self.recon_loss_fn(prediction, target_matched)

        # regularisation
        lb  = load_balance_loss(gate_probs)
        ent = gating_entropy_loss(gate_probs)

        # composite
        total = (
            self.config.reconstruction_weight * recon
            + self.config.load_balance_weight * lb
            + self.config.entropy_weight      * ent
        )

        return {
            "total"          : total,
            "reconstruction" : recon.detach(),
            "load_balance"   : lb.detach(),
            "entropy"        : ent.detach(),
        }
