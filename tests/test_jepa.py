from __future__ import annotations

import numpy as np
import pytest
import torch

from configuration.jepa_config     import EmbeddingLossConfig, ProfileAutoencoderConfig
from configuration.training_config import GaussianConfig, GeometryConfig, LossConfig
from models                        import get_model
from models.profile_autoencoder    import ProfileAutoencoder
from pipelines.jepa_pipeline.coupling import StageAMode, TargetProvider
from pipelines.jepa_pipeline.losses   import JepaLoss, ProfileAeLoss
from pipelines.training_pipeline.loss import Loss
from tools.logger  import NullLogger
from tools.tracker import NullTracker


H_ELEV, N_GAUSS, EMB_DIM = 64, 5, 24


class IdentityNorm:
    def normalize_output(self, x):
        return x

    def denormalize_output(self, x):
        return x


def make_autoencoder(**kw) -> ProfileAutoencoder:
    cfg = ProfileAutoencoderConfig(profile_length=H_ELEV, n_gaussians=N_GAUSS, embedding_dim=EMB_DIM, **kw)
    return ProfileAutoencoder(cfg)


def make_inner_loss() -> Loss:
    x_axis   = torch.linspace(0.0, 1.0, H_ELEV)
    gaussian = GaussianConfig(n_default_gaussians=N_GAUSS, x_min=0.0, x_max=1.0)
    loss_cfg = LossConfig(use_mse_curve=True, weight_mse_curve=1.0, use_param_huber=True, weight_param_huber=1.0)
    return Loss(x_axis, NullLogger(), NullTracker(), gaussian, loss_cfg, norm_stats=IdentityNorm(), geometry_cfg=GeometryConfig())


@pytest.mark.parametrize("encoder_kind", ["mlp", "conv1d", "transformer1d"])
@pytest.mark.parametrize("count_strategy", ["amplitude_threshold", "presence_logit", "count_head"])
def test_autoencoder_shapes_per_pixel(encoder_kind, count_strategy):
    m = make_autoencoder(encoder_kind=encoder_kind, decoder_kind=encoder_kind, count_strategy=count_strategy)
    curve = torch.randn(8, H_ELEV, 1, 1)
    params, curve_hat, z = m.reconstruct(curve)
    assert params.shape    == (8, 3 * N_GAUSS, 1, 1)
    assert curve_hat.shape == (8, H_ELEV, 1, 1)
    assert z.shape         == (8, EMB_DIM, 1, 1)


@pytest.mark.parametrize("encoder_kind", ["mlp", "conv1d", "transformer1d"])
def test_autoencoder_shapes_feature_map(encoder_kind):
    m = make_autoencoder(encoder_kind=encoder_kind, decoder_kind=encoder_kind)
    fmap = torch.randn(2, H_ELEV, 4, 5)
    z = m.encode(fmap)
    assert z.shape          == (2, EMB_DIM, 4, 5)
    assert m.heads(z).shape  == (2, 3 * N_GAUSS, 4, 5)
    assert m.decode(z).shape == (2, H_ELEV, 4, 5)


def test_structured_embedding_shapes():
    m = make_autoencoder(embedding_structure="structured")
    params, _, _ = m.reconstruct(torch.randn(4, H_ELEV, 1, 1))
    assert params.shape == (4, 3 * N_GAUSS, 1, 1)


def test_backbone_emits_embedding_channels():
    backbone, _ = get_model("unet", in_channels=9, out_channels=EMB_DIM)
    out = backbone(torch.randn(2, 9, 32, 32))
    assert out.shape == (2, EMB_DIM, 32, 32)


def test_autoencoder_overfits_curves():
    torch.manual_seed(0)
    x = torch.linspace(0, 1, H_ELEV)
    P = 32
    mus  = torch.rand(P, 1) * 0.8 + 0.1
    amps = torch.rand(P, 1) * 0.8 + 0.2
    sigs = torch.rand(P, 1) * 0.1 + 0.05
    curves = (amps * torch.exp(-((x[None, :] - mus) ** 2) / (2 * sigs ** 2))).reshape(P, H_ELEV, 1, 1)

    m   = make_autoencoder(hidden_dim=128, depth=3)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loss = torch.tensor(1.0)
    for _ in range(400):
        opt.zero_grad()
        loss = ((m.decode(m.encode(curves)) - curves) ** 2).mean()
        loss.backward()
        opt.step()
    assert loss.item() < 1e-2


def _assert_contract(d):
    assert set(d.keys()) == {"total_loss", "components", "weighted", "monitor"}
    assert torch.isfinite(d["total_loss"])


def test_profile_ae_loss_contract():
    m     = make_autoencoder()
    inner = make_inner_loss()
    from configuration.jepa_config import AutoencoderLossConfig
    crit  = ProfileAeLoss(inner, AutoencoderLossConfig(use_ae_curve=True))

    curve = torch.rand(6, H_ELEV, 1, 1)
    gt    = torch.rand(6, 3 * N_GAUSS, 1, 1)
    params_hat, curve_hat, _ = m.reconstruct(curve)
    out = crit(params_hat, curve_hat, curve, gt)
    _assert_contract(out)
    assert "ae_curve" in out["components"]
    assert "mse_curve" in out["components"]


def test_jepa_loss_contract_and_backward():
    m     = make_autoencoder()
    inner = make_inner_loss()
    tp    = TargetProvider("stopgrad", m.encoder)
    crit  = JepaLoss(m, inner, tp, EmbeddingLossConfig(use_embedding_mse=True), IdentityNorm())

    backbone, _ = get_model("unet", in_channels=4, out_channels=EMB_DIM)
    images = torch.randn(2, 4, 16, 16)
    gt     = torch.rand(2, 3 * N_GAUSS, 16, 16)

    z_hat = backbone(images)
    out   = crit(z_hat, gt)
    _assert_contract(out)
    assert "embedding_mse" in out["components"]
    assert "mse_curve"     in out["components"]
    out["total_loss"].backward()
    assert any(p.grad is not None for p in backbone.parameters())


def test_stage_a_mode_freeze_and_groups():
    m = make_autoencoder()
    StageAMode("frozen").apply(m)
    assert all(not p.requires_grad for p in m.parameters())
    assert StageAMode("frozen").param_groups(m, 1e-4, 1e-4) == []

    StageAMode("finetune").apply(m)
    assert all(p.requires_grad for p in m.parameters())
    groups = StageAMode("finetune").param_groups(m, 1e-5, 1e-4)
    assert len(groups) == 1 and groups[0]["name"] == "stage_a"


def test_target_provider_stopgrad_detaches():
    m  = make_autoencoder()
    tp = TargetProvider("stopgrad", m.encoder)
    z  = tp.target(m.encoder, torch.randn(3, H_ELEV, 1, 1))
    assert not z.requires_grad


def test_target_provider_ema_updates():
    m   = make_autoencoder()
    tp  = TargetProvider("ema", m.encoder, decay=0.5)
    before = [p.clone() for p in tp._ema.parameters()]
    with torch.no_grad():
        for p in m.encoder.parameters():
            p.add_(1.0)
    tp.update(m.encoder)
    after = list(tp._ema.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_profile_dataset_synthesizes_curves():
    from pipelines.jepa_pipeline.profile_dataset import ProfileDataset
    rng    = np.random.default_rng(0)
    params = np.zeros((3 * N_GAUSS, 8, 8), dtype=np.float32)
    params[0::3] = rng.uniform(0.0, 1.0, size=(N_GAUSS, 8, 8))
    params[1::3] = rng.uniform(0.0, 1.0, size=(N_GAUSS, 8, 8))
    params[2::3] = rng.uniform(0.1, 0.3, size=(N_GAUSS, 8, 8))
    x_axis = np.linspace(0, 1, H_ELEV, dtype=np.float32)

    ds = ProfileDataset([params], x_axis, normalizer=None, n_gaussians=N_GAUSS, keep_empty_frac=1.0)
    curve, gt = ds[0]
    assert curve.shape == (H_ELEV,)
    assert gt.shape    == (3 * N_GAUSS,)
