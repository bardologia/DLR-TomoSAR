from __future__ import annotations

import pytest
import torch

from models.backbone.resunet import ResUNetBackbone
from models.backbone.unet    import UNetBackbone
from models.dual import DUAL_MODEL_REGISTRY, get_dual


WINDOW = 32
BATCH  = 2

OVERRIDES = {
    "in_channels"        : 5,
    "params_channels"    : (0, 1, 2, 3, 4),
    "existence_channels" : (3, 4),
    "params_features"    : [8, 16],
    "existence_features" : [8],
    "bottleneck_factor"  : 1,
    "dropout"            : 0.0,
}


def _build(**extra):
    model, config = get_dual("dual_resunet", **{**OVERRIDES, **extra})
    return model.eval(), config


def _force_gate(model, logit_value: float) -> None:
    final_conv = model.existence_head.mlp[-1]
    torch.nn.init.zeros_(final_conv.weight)
    torch.nn.init.constant_(final_conv.bias, logit_value)


def _gate(model, x: torch.Tensor) -> torch.Tensor:
    embedding = model.trunk_existence.encode_decode(x.index_select(1, model.existence_index))
    return torch.sigmoid(model.existence_head(embedding))


def test_registry_holds_dual_resunet():
    assert set(DUAL_MODEL_REGISTRY) == {"dual_resunet"}


def test_default_trunks_are_three_and_two_levels():
    model, config = get_dual("dual_resunet")

    assert config.params_features    == [64, 128, 256]
    assert config.existence_features == [64, 128]
    assert len(model.trunk_params.encoder_blocks)    == 3
    assert len(model.trunk_existence.encoder_blocks) == 2


def test_trunk_architectures_are_selectable():
    model, config = _build(params_backbone="unet", existence_backbone="unet_skip")

    assert isinstance(model.trunk_params, UNetBackbone)
    assert isinstance(model.trunk_existence, ResUNetBackbone)
    assert model.trunk_existence.downsample_mode == "maxpool"

    with torch.no_grad():
        out = model(torch.randn(BATCH, config.in_channels, WINDOW, WINDOW))

    assert out.shape == (BATCH, config.out_channels, WINDOW, WINDOW)


def test_trunks_take_independent_widths():
    model, config = _build(existence_features=[4])

    assert model.trunk_existence.embedding_channels == 4
    assert model.existence_head.mlp[0].in_channels  == 4
    assert model.gaussian_heads[0].mlp[0].in_channels == model.trunk_params.embedding_channels

    with torch.no_grad():
        out = model(torch.randn(BATCH, config.in_channels, WINDOW, WINDOW))

    assert out.shape == (BATCH, config.out_channels, WINDOW, WINDOW)


def test_forward_shape_is_three_k_channels():
    model, config = _build()

    with torch.no_grad():
        out = model(torch.randn(BATCH, config.in_channels, WINDOW, WINDOW))

    assert out.shape == (BATCH, config.out_channels, WINDOW, WINDOW)


def test_existence_trunk_sees_only_its_channels():
    model, config = _build()

    x         = torch.randn(BATCH, config.in_channels, WINDOW, WINDOW)
    perturbed = x.clone()
    perturbed[:, :3] = torch.randn(BATCH, 3, WINDOW, WINDOW)

    with torch.no_grad():
        assert torch.equal(_gate(model, x), _gate(model, perturbed))


def test_params_trunk_sees_only_its_channels():
    model, config = _build(params_channels=(0, 1, 2))

    x         = torch.randn(BATCH, config.in_channels, WINDOW, WINDOW)
    perturbed = x.clone()
    perturbed[:, 3:] = torch.randn(BATCH, 2, WINDOW, WINDOW)

    with torch.no_grad():
        embedding_a = model.trunk_params.encode_decode(x.index_select(1, model.params_index))
        embedding_b = model.trunk_params.encode_decode(perturbed.index_select(1, model.params_index))

    assert torch.equal(embedding_a, embedding_b)


def test_params_trunk_sees_all_channels_by_default():
    model, config = _build()

    x         = torch.randn(BATCH, config.in_channels, WINDOW, WINDOW)
    perturbed = x.clone()
    perturbed[:, :3] = torch.randn(BATCH, 3, WINDOW, WINDOW)

    with torch.no_grad():
        assert not torch.equal(model(x), model(perturbed))


def test_closed_gate_pins_amplitude_to_off_level():
    model, config = _build()
    _force_gate(model, logit_value=-30.0)

    with torch.no_grad():
        model.amp_off.copy_(torch.tensor([-1.5, 0.75]))
        out = model(torch.randn(BATCH, config.in_channels, WINDOW, WINDOW))

    ppg = config.params_per_gaussian
    amp = out.reshape(BATCH, out.shape[1] // ppg, ppg, WINDOW, WINDOW)[:, :, 0]

    assert torch.allclose(amp[:, 0], torch.full_like(amp[:, 0], -1.5), atol=1e-5)
    assert torch.allclose(amp[:, 1], torch.full_like(amp[:, 1],  0.75), atol=1e-5)


def test_open_gate_passes_raw_amplitude():
    model, config = _build()
    _force_gate(model, logit_value=30.0)

    x = torch.randn(BATCH, config.in_channels, WINDOW, WINDOW)

    with torch.no_grad():
        out       = model(x)
        embedding = model.trunk_params.encode_decode(x.index_select(1, model.params_index))
        raw       = torch.stack([head(embedding) for head in model.gaussian_heads], dim=1)

    ppg = config.params_per_gaussian
    amp = out.reshape(BATCH, out.shape[1] // ppg, ppg, WINDOW, WINDOW)[:, :, 0]

    assert torch.allclose(amp, raw[:, :, 0], atol=1e-5)


def test_gate_leaves_mu_sigma_untouched():
    model, config = _build()
    _force_gate(model, logit_value=-30.0)

    x = torch.randn(BATCH, config.in_channels, WINDOW, WINDOW)

    with torch.no_grad():
        out       = model(x)
        embedding = model.trunk_params.encode_decode(x.index_select(1, model.params_index))
        raw       = torch.stack([head(embedding) for head in model.gaussian_heads], dim=1)

    ppg      = config.params_per_gaussian
    reshaped = out.reshape(BATCH, out.shape[1] // ppg, ppg, WINDOW, WINDOW)

    assert torch.allclose(reshaped[:, :, 1:], raw[:, :, 1:], atol=1e-5)


def test_gradients_reach_both_trunks_and_heads():
    model, config = _build()
    model.train()

    out  = model(torch.randn(BATCH, config.in_channels, WINDOW, WINDOW))
    loss = out.pow(2).mean()
    loss.backward()

    for module in (model.trunk_params, model.trunk_existence, model.existence_head, model.gaussian_heads):
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None for g in grads)
        assert all(torch.isfinite(g).all() for g in grads)

    assert model.amp_off.grad is not None


def test_param_groups_cover_every_parameter_exactly_once():
    model, config = _build()

    grouped = [id(p) for group in config.get_param_groups(model) for p in group["params"]]
    every   = [id(p) for p in model.parameters()]

    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(every)


def test_unknown_trunk_backbone_rejected():
    with pytest.raises(ValueError, match="params_backbone"):
        _build(params_backbone="swin_unet")


def test_empty_trunk_channels_rejected():
    with pytest.raises(ValueError, match="existence_channels"):
        _build(existence_channels=())


def test_out_of_range_trunk_channel_rejected():
    with pytest.raises(ValueError, match="out of range"):
        _build(existence_channels=(3, 5))


def test_non_set_pred_head_rejected():
    with pytest.raises(ValueError, match="set_pred"):
        _build(head="conv")
