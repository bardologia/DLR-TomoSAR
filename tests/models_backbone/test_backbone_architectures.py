from __future__ import annotations

import numpy as np
import pytest
import torch

from models.backbone import BACKBONE_MODEL_REGISTRY, BACKBONE_CONFIG_REGISTRY, BACKBONE_HEADS, BACKBONE_IMAGE_SIZE_MODELS, get_backbone


DEVICE = torch.device("cpu")
WINDOW = 64
BATCH  = 2

ALL_NAMES      = sorted(BACKBONE_MODEL_REGISTRY.keys())
GAUSSIAN_HEADS = [head for head in BACKBONE_HEADS if head != "conv"]

SMALL_OVERRIDES = {
    "unet"           : {"features": [8, 16], "bottleneck_factor": 1},
    "unet_skip"      : {"features": [8, 16], "bottleneck_factor": 1},
    "resunet"        : {"features": [8, 16], "bottleneck_factor": 1},
    "attention_unet" : {"features": [8, 16], "bottleneck_factor": 1},
    "unetplusplus"   : {"features": [8, 16, 32, 64], "bottleneck_factor": 1},
    "linknet"        : {"features": [16, 32, 64, 128], "initial_kernel_size": 3},
    "swin_unet"      : {"image_size": WINDOW, "embedding_dim": 24, "depths": [2, 2, 2, 2], "num_heads": [1, 2, 4, 8], "window_size": 4},
    "transunet"      : {"image_size": WINDOW, "cnn_features": [8, 16, 32, 64], "transformer_layers": 2, "transformer_heads": 2},
    "unetr"          : {"image_size": WINDOW, "embedding_dim": 64, "transformer_layers": 4, "transformer_heads": 4, "decoder_features": [32, 16, 8, 8]},
    "deeplabv3plus"  : {"features": [16, 32, 64, 128]},
    "segformer"      : {"embedding_dims": [16, 32, 64, 128], "depths": [1, 1, 1, 1], "decoder_channels": 64},
    "convnext_unet"  : {"features": [16, 32, 64, 128], "blocks_per_stage": 1, "bottleneck_factor": 1},
    "dense_unet"     : {"growth_rate": 8, "block_layers": [2, 2, 2], "bottleneck_layers": 2},
    "hrnet"          : {"base_channels": 16, "n_branches": 3, "blocks_per_stage": 1},
    "multires_unet"  : {"features": [16, 32, 64, 128], "bottleneck_factor": 1},
    "fpn"            : {"features": [16, 32, 64, 128], "pyramid_channels": 32, "segmentation_convs": 1},
    "u2net"          : {"features": [16, 32, 64, 128], "rsu_heights": (4, 3, 2)},
    "pixel_mlp"      : {"features": [32, 32]},
    "local_cnn"      : {"features": [8, 16]},
    "nafnet"         : {"width": 8, "enc_blocks": [1, 1], "middle_blocks": 1, "dec_blocks": [1, 1]},
}


def _build(name, head: str = "conv"):
    overrides = dict(SMALL_OVERRIDES.get(name, {}))
    model, config = get_backbone(name, head=head, **overrides)
    model = model.to(DEVICE).eval()
    return model, config


def _input_window(interferograms, in_channels):
    raw  = np.asarray(interferograms[:in_channels, :WINDOW, :WINDOW])
    mag  = np.abs(raw).astype(np.float32)
    mean = mag.mean(axis=(1, 2), keepdims=True)
    std  = mag.std(axis=(1, 2), keepdims=True) + 1e-6
    norm = (mag - mean) / std
    batch = np.repeat(norm[None, ...], BATCH, axis=0)
    return torch.from_numpy(batch).to(DEVICE)


@pytest.fixture
def make_input(interferograms):
    def _factory(config):
        return _input_window(interferograms, config.in_channels)
    return _factory


@pytest.mark.parametrize("name", ALL_NAMES)
def test_construct_and_param_count(name):
    model, _ = _build(name)
    total    = sum(p.numel() for p in model.parameters())
    assert total > 0


@pytest.mark.parametrize("name", ALL_NAMES)
def test_forward_shape_preserved(name, make_input):
    model, config = _build(name)
    x             = make_input(config)

    with torch.no_grad():
        y = model(x)

    assert y.shape[0] == BATCH
    assert y.shape[1] == config.out_channels
    assert y.shape[2] == WINDOW
    assert y.shape[3] == WINDOW


@pytest.mark.parametrize("name", ALL_NAMES)
def test_forward_finite(name, make_input):
    model, config = _build(name)
    x             = make_input(config)

    with torch.no_grad():
        y = model(x)

    assert torch.isfinite(y).all()


@pytest.mark.parametrize("name", ALL_NAMES)
def test_backward_finite_gradients(name, make_input):
    model, config = _build(name)
    model.train()
    x             = make_input(config)

    y    = model(x)
    loss = y.float().pow(2).mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]

    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    assert torch.isfinite(loss).item()


@pytest.mark.parametrize("name", ALL_NAMES)
def test_deterministic_eval(name, make_input):
    model, config = _build(name)
    x             = make_input(config)

    with torch.no_grad():
        a = model(x)
        b = model(x)

    assert torch.allclose(a, b, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_cpu_roundtrip(name):
    model, _ = _build(name)

    model.to("cpu")
    devices = {p.device.type for p in model.parameters()}

    assert devices == {"cpu"}


def test_registry_covers_all_models():
    assert set(BACKBONE_MODEL_REGISTRY.keys()) == set(BACKBONE_CONFIG_REGISTRY.keys())
    assert len(ALL_NAMES) > 0
    assert BACKBONE_IMAGE_SIZE_MODELS.issubset(set(ALL_NAMES))


@pytest.mark.parametrize("head", GAUSSIAN_HEADS)
@pytest.mark.parametrize("name", ALL_NAMES)
def test_gaussian_head_forward_shape(name, head, make_input):
    model, config = _build(name, head=head)
    x             = make_input(config)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (BATCH, config.out_channels, WINDOW, WINDOW)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("head", BACKBONE_HEADS)
@pytest.mark.parametrize("name", ALL_NAMES)
def test_param_groups_cover_every_parameter(name, head):
    model, config = _build(name, head=head)

    grouped = sum(len(group["params"]) for group in config.get_param_groups(model))
    total   = sum(1 for _ in model.parameters())

    assert grouped == total


@pytest.mark.parametrize("name", ALL_NAMES)
def test_unknown_head_raises(name):
    with pytest.raises(ValueError):
        _build(name, head="dense")
