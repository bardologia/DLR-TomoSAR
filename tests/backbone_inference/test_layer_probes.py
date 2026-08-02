from __future__ import annotations

import numpy as np
import pytest
import torch

from pipelines.backbone.inference.layer_probes import LayerProbeCore, RidgeProbe


H, W = 16, 12


def test_ridge_probe_recovers_a_linear_signal():
    rng = np.random.default_rng(0)
    X   = rng.normal(size=(400, 8))
    y   = X @ rng.normal(size=8) + 0.01 * rng.normal(size=400)

    assert RidgeProbe(ridge_lambda=1e-3).score(X, y) > 0.95


def test_ridge_probe_scores_noise_near_zero():
    rng = np.random.default_rng(1)
    X   = rng.normal(size=(400, 8))
    y   = rng.normal(size=400)

    assert abs(RidgeProbe().score(X, y)) < 0.2


def test_ridge_probe_needs_enough_samples():
    with pytest.raises(ValueError):
        RidgeProbe().score(np.zeros((5, 3)), np.zeros(5))


class _TwoLayerNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.useful  = torch.nn.Identity()
        self.useless = torch.nn.Identity()

    def forward(self, x):
        a = self.useful(x)
        _ = self.useless(torch.zeros_like(x))
        return a


def _batches(n: int = 3):
    rng     = np.random.default_rng(2)
    batches = []

    for _ in range(n):
        counts  = rng.integers(0, 3, size=(2, H, W)).astype(np.float32)
        images  = torch.from_numpy(np.stack([counts, rng.normal(size=(2, H, W)).astype(np.float32)], axis=1))

        gt = np.zeros((2, 6, H, W), dtype=np.float32)
        gt[:, 0] = (counts >= 1).astype(np.float32)
        gt[:, 3] = (counts >= 2).astype(np.float32)
        gt[:, 1] = 20.0
        gt[:, 4] = 40.0

        batches.append((images, gt))

    return batches


def test_probe_core_finds_the_informative_layer():
    core = LayerProbeCore(
        model             = _TwoLayerNet(),
        layers            = ["useful", "useless"],
        ppg               = 3,
        amp_thr           = 1e-3,
        samples_per_batch = 256,
        probe             = RidgeProbe(ridge_lambda=1e-3),
    )

    results = core.collect(_batches())

    assert results["useful"]["k_r2"]  > 0.9
    assert results["useless"]["k_r2"] < 0.2
    assert results["useful"]["n_channels"] == 2


def test_probe_core_without_feature_layers_raises():
    class _ScalarNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.flat = torch.nn.Identity()

        def forward(self, x):
            _ = self.flat(x.mean())
            return x

    core = LayerProbeCore(
        model             = _ScalarNet(),
        layers            = ["flat"],
        ppg               = 3,
        amp_thr           = 1e-3,
        samples_per_batch = 128,
        probe             = RidgeProbe(),
    )

    with pytest.raises(ValueError):
        core.collect(_batches(1))
