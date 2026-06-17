from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from pipelines.jepa.training.coupling import CouplingMode, TargetProvider


def make_encoder():
    return nn.Sequential(nn.Conv2d(2, 2, kernel_size=1))


def test_coupling_mode_rejects_unknown_kind():
    with pytest.raises(ValueError):
        CouplingMode("scratch", "profile autoencoder")


def test_coupling_mode_frozen_disables_grad_and_eval():
    mode       = CouplingMode("frozen", "profile autoencoder")
    encoder    = make_encoder()
    mode.apply(encoder)

    assert mode.trainable is False
    assert all(not p.requires_grad for p in encoder.parameters())
    assert encoder.training is False


def test_coupling_mode_finetune_enables_grad():
    mode    = CouplingMode("finetune", "profile autoencoder")
    encoder = make_encoder()
    mode.apply(encoder)

    assert mode.trainable is True
    assert all(p.requires_grad for p in encoder.parameters())


def test_coupling_mode_param_groups_empty_when_frozen():
    mode    = CouplingMode("frozen", "profile autoencoder")
    encoder = make_encoder()
    mode.apply(encoder)

    assert mode.param_groups(encoder, lr=1e-3, wd=1e-4) == []


def test_coupling_mode_param_groups_carry_lr_and_wd_when_finetune():
    mode    = CouplingMode("finetune", "profile autoencoder")
    encoder = make_encoder()
    mode.apply(encoder)

    groups = mode.param_groups(encoder, lr=7e-4, wd=2e-5)

    assert len(groups)          == 1
    assert groups[0]["lr"]      == 7e-4
    assert groups[0]["weight_decay"] == 2e-5
    assert groups[0]["name"]    == "profile autoencoder"
    assert len(groups[0]["params"]) > 0


def test_target_provider_live_is_differentiable():
    encoder  = make_encoder()
    provider = TargetProvider("live", encoder)
    curve    = torch.randn(1, 2, 3, 3, requires_grad=True)

    out = provider.target(encoder, curve)

    assert out.requires_grad is True


def test_target_provider_stopgrad_detaches_target():
    encoder  = make_encoder()
    provider = TargetProvider("stopgrad", encoder)
    curve    = torch.randn(1, 2, 3, 3, requires_grad=True)

    out = provider.target(encoder, curve)

    assert out.requires_grad is False


def test_target_provider_ema_has_detached_parameters():
    encoder  = make_encoder()
    provider = TargetProvider("ema", encoder, decay=0.9)

    assert provider._ema is not None
    assert all(not p.requires_grad for p in provider._ema.parameters())
    assert provider._ema.training is False


def test_target_provider_ema_uses_internal_copy_not_online_encoder():
    encoder  = make_encoder()
    provider = TargetProvider("ema", encoder, decay=0.9)
    curve    = torch.randn(1, 2, 3, 3)

    with torch.no_grad():
        for p in encoder.parameters():
            p.add_(5.0)

    out_target = provider.target(encoder, curve)
    out_online = encoder(curve)

    assert not torch.allclose(out_target, out_online)


def test_target_provider_ema_update_math():
    encoder  = make_encoder()
    decay    = 0.25
    provider = TargetProvider("ema", encoder, decay=decay)

    ema_before    = [p.detach().clone() for p in provider._ema.parameters()]
    online_after  = []
    with torch.no_grad():
        for p in encoder.parameters():
            p.add_(2.0)
            online_after.append(p.detach().clone())

    provider.update(encoder)

    for before, online, after in zip(ema_before, online_after, provider._ema.parameters()):
        expected = before * decay + online * (1.0 - decay)
        assert torch.allclose(after, expected, atol=1e-6)


def test_target_provider_ema_copies_buffers():
    encoder = nn.Sequential(nn.BatchNorm2d(2))
    provider = TargetProvider("ema", encoder, decay=0.5)

    with torch.no_grad():
        for b in encoder.buffers():
            if b.dtype.is_floating_point:
                b.add_(3.0)

    provider.update(encoder)

    for ema_b, online_b in zip(provider._ema.buffers(), encoder.buffers()):
        assert torch.allclose(ema_b, online_b)


def test_target_provider_update_is_noop_without_ema():
    encoder  = make_encoder()
    provider = TargetProvider("stopgrad", encoder)

    provider.update(encoder)

    assert provider._ema is None


def test_target_provider_update_does_not_touch_online_grad_graph():
    encoder  = make_encoder()
    provider = TargetProvider("ema", encoder, decay=0.5)
    snapshot = copy.deepcopy(encoder.state_dict())

    provider.update(encoder)

    for key, value in encoder.state_dict().items():
        assert torch.allclose(value, snapshot[key])
