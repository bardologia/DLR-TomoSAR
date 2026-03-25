"""
Dense Mixture-of-Experts for pixel-level multi-complexity prediction.

Gate-first pipeline:
  input → gating → routing → conditional expert execution → aggregation → output

Quick start::

    from mixture_of_experts import DenseMoE, MoEConfig

    config = MoEConfig(in_channels=1)       # 3 UNet experts: 3 / 6 / 9 channels
    model  = DenseMoE(config)
    out    = model(torch.randn(4, 1, 64, 64))
    out.prediction.shape                    # (4, 9, 64, 64)
    out.gate_probs.shape                    # (4, 3, 64, 64)

Gate-only training with pretrained experts::

    config = MoEConfig(
        in_channels   = 1,
        experts       = [
            ExpertDefinition(out_channels=3, pretrained_path="ckpt/exp3.pt"),
            ExpertDefinition(out_channels=6, pretrained_path="ckpt/exp6.pt"),
            ExpertDefinition(out_channels=9, pretrained_path="ckpt/exp9.pt"),
        ],
        training_mode = TrainingMode.gate_only,
    )
    model = DenseMoE(config)   # only gating params are trainable
"""

from .config import (
    AggregationMode,
    ExpertDefinition,
    GatingConfig,
    GatingType,
    MoEConfig,
    MoELossConfig,
    TrainingMode,
)
from .experts import apply_training_mode, build_experts
from .gating import build_gating
from .losses import (
    MoELoss,
    gating_entropy_loss,
    load_balance_loss,
    per_expert_reconstruction_loss,
)
from .moe_model import DenseMoE, MoEOutput
from .trainer import MoETrainer, TrainerConfig, build_optimizer, build_scheduler
