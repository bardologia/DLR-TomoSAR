from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from configuration.training.jepa      import EmbeddingLossConfig
from pipelines.jepa.training.coupling import CouplingMode, TargetProvider
from pipelines.jepa.training.loss     import Loss as EmbeddingLoss
from pipelines.jepa.training.trainer  import JepaModule, Trainer
from pipelines.profile_autoencoder.dataset.normalization import ProfileNormalizer, ProfileStats

from tests.jepa.conftest import EMBEDDING_DIM, IdentityNormStats, N_GAUSSIANS, SPATIAL, make_autoencoder


def build_module():
    autoencoder = make_autoencoder("none")
    backbone    = nn.Conv2d(2, EMBEDDING_DIM, kernel_size=1)
    return JepaModule(backbone, profile_autoencoder=autoencoder, image_autoencoder=None)


def test_jepa_module_forward_routes_through_backbone():
    module = build_module()
    images = torch.randn(2, 2, SPATIAL, SPATIAL)

    out = module(images)

    assert out.shape == (2, EMBEDDING_DIM, SPATIAL, SPATIAL)


def test_jepa_module_image_autoencoder_disabled_by_default():
    module = build_module()

    assert module.image_autoencoder   is None
    assert module.profile_autoencoder is not None


def test_validate_coupling_live_requires_finetune():
    frozen = CouplingMode("frozen", "profile autoencoder")
    cfg    = EmbeddingLossConfig(use_curve_recon=True)

    with pytest.raises(ValueError):
        Trainer.validate_coupling(frozen, "live", cfg)


def test_validate_coupling_ema_requires_finetune():
    frozen = CouplingMode("frozen", "profile autoencoder")
    cfg    = EmbeddingLossConfig(use_curve_recon=True)

    with pytest.raises(ValueError):
        Trainer.validate_coupling(frozen, "ema", cfg)


def test_validate_coupling_live_requires_curve_recon():
    finetune = CouplingMode("finetune", "profile autoencoder")
    cfg      = EmbeddingLossConfig(use_curve_recon=False)

    with pytest.raises(ValueError):
        Trainer.validate_coupling(finetune, "live", cfg)


def test_validate_coupling_stopgrad_frozen_is_valid():
    frozen = CouplingMode("frozen", "profile autoencoder")
    cfg    = EmbeddingLossConfig(use_curve_recon=True)

    Trainer.validate_coupling(frozen, "stopgrad", cfg)


def make_trainer_shim(target_kind="stopgrad", trainable=True):
    module = build_module()
    mode   = CouplingMode("finetune" if trainable else "frozen", "profile autoencoder")
    mode.apply(module.profile_autoencoder)

    x_axis             = torch.linspace(-4.0, 4.0, module.profile_autoencoder.config.profile_length)

    norm_stats         = IdentityNormStats()
    profile_normalizer = ProfileNormalizer(ProfileStats(loc=0.0, scale=1.0))
    emb_cfg            = EmbeddingLossConfig(use_embedding_mse=True, weight_embedding_mse=1.0, use_curve_recon=False)

    provider  = TargetProvider(target_kind, module.profile_autoencoder.encoder, decay=0.5)
    criterion = EmbeddingLoss(module.profile_autoencoder, provider, emb_cfg, x_axis, norm_stats, 3, profile_normalizer)

    trainer = Trainer.__new__(Trainer)
    trainer.model        = module
    trainer.device       = torch.device("cpu")
    trainer.has_profile  = True
    trainer.has_image    = False
    trainer.profile_mode = mode
    trainer.criterion    = criterion
    return trainer


def test_compute_loss_runs_on_tiny_batch():
    trainer = make_trainer_shim()
    images  = torch.randn(2, 2, SPATIAL, SPATIAL)
    gt      = torch.rand(2, N_GAUSSIANS * 3, SPATIAL, SPATIAL)

    out = trainer._compute_loss((images, gt))

    assert torch.isfinite(out["total_loss"])


def test_optimizer_step_updates_online_branch():
    trainer   = make_trainer_shim()
    params    = [p for p in trainer.model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.1)

    images = torch.randn(2, 2, SPATIAL, SPATIAL)
    gt     = torch.rand(2, N_GAUSSIANS * 3, SPATIAL, SPATIAL)

    before = [p.detach().clone() for p in trainer.model.backbone.parameters()]

    optimizer.zero_grad(set_to_none=True)
    loss = trainer._compute_loss((images, gt))["total_loss"]
    loss.backward()
    optimizer.step()

    after   = list(trainer.model.backbone.parameters())
    changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
    assert changed


def test_on_optimizer_step_updates_ema_target():
    trainer    = make_trainer_shim(target_kind="ema")
    ema_before = [p.detach().clone() for p in trainer.criterion.target_provider._ema.parameters()]

    with torch.no_grad():
        for p in trainer.model.profile_autoencoder.encoder.parameters():
            p.add_(1.0)

    trainer._on_optimizer_step()

    ema_after = list(trainer.criterion.target_provider._ema.parameters())
    moved     = any(not torch.allclose(b, a) for b, a in zip(ema_before, ema_after))
    assert moved


def test_on_optimizer_step_noop_when_frozen():
    trainer = make_trainer_shim(target_kind="stopgrad", trainable=False)

    trainer._on_optimizer_step()

    assert trainer.criterion.target_provider._ema is None


def test_checkpoint_state_dict_round_trip(tmp_path):
    module = build_module()
    images = torch.randn(2, 2, SPATIAL, SPATIAL)
    gt     = torch.rand(2, N_GAUSSIANS * 3, SPATIAL, SPATIAL)

    state = {"epoch": 1, "params": module.state_dict(), "x_axis": torch.linspace(-4, 4, 8).numpy()}
    path  = tmp_path / "best_model.pt"
    torch.save(state, path)

    loaded   = torch.load(path, map_location="cpu", weights_only=False)
    restored = build_module()
    restored.load_state_dict(loaded["params"])

    with torch.no_grad():
        assert torch.allclose(module(images), restored(images))
