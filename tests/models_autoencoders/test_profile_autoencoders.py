from __future__ import annotations

import numpy as np
import pytest
import torch

from models.profile_autoencoder import PROFILE_AE_CONFIG_REGISTRY, get_profile_autoencoder


PROFILE_LENGTH = 16
EMBEDDING_DIM  = 8
NAMES          = sorted(PROFILE_AE_CONFIG_REGISTRY.keys())


def _tiny_config(name):
    cfg = PROFILE_AE_CONFIG_REGISTRY[name](profile_length=PROFILE_LENGTH, embedding_dim=EMBEDDING_DIM)
    if hasattr(cfg, "hidden_dim"):
        cfg.hidden_dim = 16
    if hasattr(cfg, "seq_channels"):
        cfg.seq_channels = 8
    if hasattr(cfg, "num_heads"):
        cfg.num_heads = 2
    if hasattr(cfg, "depth"):
        cfg.depth = 2
    if hasattr(cfg, "patch_size"):
        cfg.patch_size = 4
    return cfg


def _build(name):
    model, cfg = get_profile_autoencoder(name, _tiny_config(name))
    return model, cfg


def _real_batch():
    return (2, 3, 3)


def _profile_input(tomogram_full, dims):
    B, H, W = dims
    block   = np.asarray(tomogram_full[:PROFILE_LENGTH, :H, : B * W])
    mag     = np.abs(block).astype(np.float32)
    tensor  = torch.from_numpy(mag).reshape(PROFILE_LENGTH, H, B, W).permute(2, 0, 1, 3).contiguous()
    return tensor


def _finite(t):
    return torch.isfinite(t).all().item()


@pytest.mark.parametrize("name", NAMES)
def test_registry_entries_build(name):
    model, cfg = _build(name)
    assert isinstance(model, torch.nn.Module)
    assert cfg.profile_length == PROFILE_LENGTH


@pytest.mark.real_data
@pytest.mark.parametrize("name", NAMES)
def test_roundtrip_shape(name, tomogram_full):
    model, cfg = _build(name)
    model.eval()

    dims = _real_batch()
    x    = _profile_input(tomogram_full, dims)

    with torch.no_grad():
        x_hat, z = model.reconstruct(x)

    assert x_hat.shape == x.shape
    B, H, W = dims
    assert z.shape == (B, EMBEDDING_DIM, H, W)
    assert _finite(x_hat)
    assert _finite(z)


@pytest.mark.real_data
@pytest.mark.parametrize("name", NAMES)
def test_latent_dim(name, tomogram_full):
    model, _ = _build(name)
    model.eval()

    x = _profile_input(tomogram_full, _real_batch())
    with torch.no_grad():
        z = model.encode(x)

    assert z.shape[1] == EMBEDDING_DIM


@pytest.mark.real_data
@pytest.mark.parametrize("name", NAMES)
def test_backward_finite_grads(name, tomogram_full):
    model, _ = _build(name)
    model.train()

    x       = _profile_input(tomogram_full, _real_batch())
    x_hat   = model(x)
    loss    = torch.nn.functional.mse_loss(x_hat, x)
    loss.backward()

    assert _finite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(_finite(g) for g in grads)


@pytest.mark.parametrize("name", NAMES)
def test_eval_deterministic(name):
    model, _ = _build(name)
    model.eval()

    x = torch.randn(2, PROFILE_LENGTH, 3, 3)
    with torch.no_grad():
        first  = model(x)
        second = model(x)

    assert torch.allclose(first, second)


@pytest.mark.parametrize("name", NAMES)
def test_zero_input_finite(name):
    model, _ = _build(name)
    model.eval()

    x = torch.zeros(2, PROFILE_LENGTH, 3, 3)
    with torch.no_grad():
        x_hat = model(x)

    assert x_hat.shape == x.shape
    assert _finite(x_hat)


@pytest.mark.parametrize("name", NAMES)
def test_constant_input_finite(name):
    model, _ = _build(name)
    model.eval()

    x = torch.full((2, PROFILE_LENGTH, 3, 3), 0.7)
    with torch.no_grad():
        x_hat = model(x)

    assert x_hat.shape == x.shape
    assert _finite(x_hat)
