"""
match_model_sizes.py
--------------------
Counts the parameters of every registered model and finds the configuration
that makes each model's parameter count as close as possible to the UNet
baseline (default UNetConfig).

Strategy
--------
CNN-based models (UNet variants, ResUNet, AttentionUNet, UNetPlusPlus,
# LinkNet, UNetMultiHead, UNetPerGaussian):
    Scale the `features` list uniformly by a real-valued factor and round
    each element to the nearest multiple of 8.  Binary-search on the factor.

Transformer-based models (SwinUNet, TransUNet, UNETR):
    Scale `embedding_dim` (SwinUNet / UNETR) or `cnn_features` +
    `embedding_dim` derived from the last CNN feature (TransUNet).
    Binary-search on the embedding dimension directly.

Usage
-----
    cd /ste/rnd/User/vice_vi/DLR-TomoSAR
    python scripts/match_model_sizes.py
"""

from __future__ import annotations

import sys
import os

# Make sure project root is on the path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import copy
import math
from typing import Any, Callable

import torch

from configuration.models_config import (
    AttentionUNetConfig,
    LinkNetConfig,
    ResUNetConfig,
    SwinUNetConfig,
    TransUNetConfig,
    UNETRConfig,
    UNetConfig,
    UNetMultiHeadConfig,
    UNetPerGaussianConfig,
    UNetPlusPlusConfig,
)
from models import (
    AttentionUNet,
    LinkNet,
    ResUNet,
    SwinUNet,
    TransUNet,
    UNETR,
    UNet,
    UNetMultiHead,
    UNetPerGaussian,
    UNetPlusPlus,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def round8(x: float) -> int:
    """Round x to the nearest multiple of 8, minimum 8."""
    return max(8, round(x / 8) * 8)


def scale_features(base: list[int], factor: float) -> list[int]:
    return [round8(f * factor) for f in base]


def build_and_count(model_cls, config) -> int:
    try:
        model = model_cls(config)
        return count_params(model)
    except Exception:
        return -1


# ──────────────────────────────────────────────────────────────────────────────
# Binary search
# ──────────────────────────────────────────────────────────────────────────────

def binary_search_factor(
    model_cls,
    make_config: Callable[[float], Any],
    target: int,
    lo: float = 0.1,
    hi: float = 8.0,
    iters: int = 40,
) -> tuple[float, Any, int]:
    """
    Binary-search on a scalar `factor` passed to `make_config(factor)`.
    Returns (best_factor, best_config, best_param_count).
    """
    best_factor = 1.0
    best_config = make_config(1.0)
    best_count  = build_and_count(model_cls, best_config)

    for _ in range(iters):
        mid = (lo + hi) / 2.0
        cfg = make_config(mid)
        n   = build_and_count(model_cls, cfg)
        if n < 0:
            hi = mid
            continue
        if n < target:
            lo = mid
        else:
            hi = mid
        if abs(n - target) < abs(best_count - target):
            best_factor = mid
            best_config = cfg
            best_count  = n

    return best_factor, best_config, best_count


# ──────────────────────────────────────────────────────────────────────────────
# Per-model config factories
# ──────────────────────────────────────────────────────────────────────────────

BASE_FEATURES = [64, 128, 256, 512]

def _cnn_factory(cfg_cls, extra: dict | None = None):
    """Returns a make_config(factor) closure for CNN-style models."""
    extra = extra or {}
    def make_config(f: float):
        cfg = cfg_cls(features=scale_features(BASE_FEATURES, f), **extra)
        return cfg
    return make_config


def _unetplusplus_factory():
    def make_config(f: float):
        return UNetPlusPlusConfig(features=scale_features(BASE_FEATURES, f))
    return make_config


def _swin_factory():
    # Scale embedding_dim; keep depths/heads fixed, scale heads proportionally
    BASE_EMB   = 96
    BASE_HEADS = [3, 6, 12, 24]
    def make_config(f: float):
        emb = round8(BASE_EMB * f)
        # heads must divide emb at every stage
        heads = []
        for h in BASE_HEADS:
            candidate = max(1, round(h * f))
            while emb % candidate != 0 and candidate > 1:
                candidate -= 1
            heads.append(max(1, candidate))
        return SwinUNetConfig(embedding_dim=emb, num_heads=heads)
    return make_config


def _transunet_factory():
    BASE_CNN  = [64, 128, 256, 512]
    BASE_HEADS = 8
    def make_config(f: float):
        feats = scale_features(BASE_CNN, f)
        # transformer embedding = last CNN feature * bottleneck_factor isn't
        # directly configurable; the model derives it from cnn_features[-1].
        # We control size via cnn_features and transformer_layers.
        layers = max(1, round(12 * f))
        heads  = max(1, round(BASE_HEADS * f))
        # heads must divide the derived embedding; keep heads a power of 2
        heads  = max(1, 2 ** round(math.log2(heads)))
        return TransUNetConfig(cnn_features=feats, transformer_layers=layers, transformer_heads=heads)
    return make_config


def _unetr_factory():
    BASE_EMB      = 768
    BASE_DEC_FEAT = [512, 256, 128, 64]
    def make_config(f: float):
        emb       = round8(BASE_EMB * f)
        dec_feats = scale_features(BASE_DEC_FEAT, f)
        layers    = max(1, round(12 * f))
        # transformer_heads must divide emb
        heads = 12
        while emb % heads != 0 and heads > 1:
            heads -= 1
        return UNETRConfig(
            embedding_dim     = emb,
            decoder_features  = dec_feats,
            transformer_layers = layers,
            transformer_heads  = heads,
        )
    return make_config


# ──────────────────────────────────────────────────────────────────────────────
# Registry of models to tune
# ──────────────────────────────────────────────────────────────────────────────

MODELS_TO_TUNE: list[tuple[str, type, Callable]] = [
    ("unet_multihead",   UNetMultiHead,   _cnn_factory(UNetMultiHeadConfig)),
    ("unet_pergaussian", UNetPerGaussian, _cnn_factory(UNetPerGaussianConfig)),
    ("resunet",          ResUNet,         _cnn_factory(ResUNetConfig)),
    ("attention_unet",   AttentionUNet,   _cnn_factory(AttentionUNetConfig)),
    ("unetplusplus",     UNetPlusPlus,    _unetplusplus_factory()),
    ("linknet",          LinkNet,         _cnn_factory(LinkNetConfig)),
    ("swin_unet",        SwinUNet,        _swin_factory()),
    ("transunet",        TransUNet,       _transunet_factory()),
    ("unetr",            UNETR,           _unetr_factory()),
]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline_cfg   = UNetConfig()
    baseline_model = UNet(baseline_cfg)
    target         = count_params(baseline_model)

    print("=" * 70)
    print(f"  Baseline  UNet  →  {target:>12,} parameters")
    print(f"  Features  : {baseline_cfg.features}")
    print(f"  Bottleneck factor: {baseline_cfg.bottleneck_factor}")
    print("=" * 70)
    print()

    results: list[dict] = []

    for name, model_cls, make_config in MODELS_TO_TUNE:
        factor, cfg, n_params = binary_search_factor(
            model_cls, make_config, target,
            lo=0.05, hi=6.0, iters=50,
        )
        pct_diff = (n_params - target) / target * 100
        results.append(dict(name=name, factor=factor, config=cfg, params=n_params, pct=pct_diff))

    # ── Report ────────────────────────────────────────────────────────────────
    col_w = 20
    print(f"{'Model':<{col_w}}  {'Params':>12}  {'vs UNet':>8}  Key size fields")
    print("-" * 80)
    for r in results:
        cfg  = r["config"]
        name = r["name"]

        if hasattr(cfg, "features"):
            size_str = f"features={cfg.features}"
        elif hasattr(cfg, "embedding_dim"):
            size_str = f"embedding_dim={cfg.embedding_dim}"
        elif hasattr(cfg, "cnn_features"):
            size_str = f"cnn_features={cfg.cnn_features}"
        else:
            size_str = ""

        print(f"{name:<{col_w}}  {r['params']:>12,}  {r['pct']:>+7.1f}%  {size_str}")

    # ── Suggested configs ─────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  Suggested config overrides (copy into your config instantiation)")
    print("=" * 70)
    for r in results:
        cfg  = r["config"]
        name = r["name"]
        print(f"\n# {name}  ({r['params']:,} params,  {r['pct']:+.1f}% vs UNet)")

        if hasattr(cfg, "features"):
            print(f"  features          = {cfg.features}")
        if hasattr(cfg, "bottleneck_factor"):
            print(f"  bottleneck_factor = {cfg.bottleneck_factor}")
        if hasattr(cfg, "embedding_dim"):
            print(f"  embedding_dim     = {cfg.embedding_dim}")
        if hasattr(cfg, "num_heads"):
            print(f"  num_heads         = {cfg.num_heads}")
        if hasattr(cfg, "depths"):
            print(f"  depths            = {cfg.depths}")
        if hasattr(cfg, "cnn_features"):
            print(f"  cnn_features      = {cfg.cnn_features}")
        if hasattr(cfg, "transformer_layers"):
            print(f"  transformer_layers = {cfg.transformer_layers}")
        if hasattr(cfg, "transformer_heads"):
            print(f"  transformer_heads  = {cfg.transformer_heads}")
        if hasattr(cfg, "decoder_features"):
            print(f"  decoder_features   = {cfg.decoder_features}")

    print()


if __name__ == "__main__":
    main()
