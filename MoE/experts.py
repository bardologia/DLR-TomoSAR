"""Expert construction, checkpoint loading, and freeze/unfreeze utilities."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models import MODEL_REGISTRY, CONFIG_REGISTRY

from .config import ExpertDefinition, MoEConfig, TrainingMode


def _build_single_expert(
    in_channels : int,
    expert_def  : ExpertDefinition,
    init_mode   : str = "default",
) -> nn.Module:
    backbone_key = expert_def.backbone.lower()
    model_cls    = MODEL_REGISTRY.get(backbone_key)
    config_cls   = CONFIG_REGISTRY.get(backbone_key)

    if model_cls is None or config_cls is None:
        raise ValueError(
            f"Unknown backbone '{expert_def.backbone}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    cfg_kwargs = {
        "in_channels"  : in_channels,
        "out_channels" : expert_def.out_channels,
        **expert_def.backbone_config,
    }
    cfg = config_cls(**cfg_kwargs)

    if hasattr(cfg, "init_mode"):
        cfg.init_mode = init_mode

    return model_cls(cfg)


def _load_pretrained(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Expert checkpoint not found: {path}")

    state_dict = torch.load(path, map_location="cpu", weights_only=True)

    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=True)


def build_experts(config: MoEConfig) -> nn.ModuleList:
    """Build all expert networks; load pretrained weights when available."""
    experts = nn.ModuleList()

    for idx, expert_def in enumerate(config.experts):
        model = _build_single_expert(config.in_channels, expert_def, config.init_mode)

        if expert_def.pretrained_path is not None:
            _load_pretrained(model, expert_def.pretrained_path)
        elif config.training_mode == TrainingMode.gate_only:
            raise ValueError(
                f"Expert {idx} has no pretrained_path, but training_mode "
                f"is gate_only — all experts must supply a checkpoint."
            )

        experts.append(model)

    return experts


def apply_training_mode(
    experts : nn.ModuleList,
    gating  : nn.Module,
    mode    : TrainingMode,
) -> None:
    """Freeze / unfreeze sub-networks according to the training mode."""
    if mode == TrainingMode.end_to_end:
        _set_requires_grad(experts, True)
        _set_requires_grad(gating, True)
    elif mode == TrainingMode.gate_only:
        _set_requires_grad(experts, False)
        _set_requires_grad(gating, True)
    elif mode == TrainingMode.experts_only:
        _set_requires_grad(experts, True)
        _set_requires_grad(gating, False)
    else:
        raise ValueError(f"Unknown training mode: {mode}")


def _set_requires_grad(module: nn.Module, value: bool) -> None:
    for p in module.parameters():
        p.requires_grad = value
