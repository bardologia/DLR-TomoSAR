from __future__ import annotations

import numpy as np
import pytest
import torch

from configuration.autoencoder_config import AutoencoderLossConfig, ProfileAutoencoderConfig
from configuration.jepa_config         import EmbeddingLossConfig
from models                        import get_model
from models.profile_autoencoder    import ProfileAutoencoder
from pipelines.jepa_pipeline.coupling          import StageAMode, TargetProvider
from pipelines.jepa_pipeline.losses            import JepaLoss
from pipelines.autoencoder_pipeline.losses     import ProfileAeLoss


H_ELEV, N_GAUSS, EMB_DIM = 64, 5, 24


class IdentityNorm:
    def normalize_output(self, x):
        return x

    def denormalize_output(self, x):
        return x


def make_autoencoder(**kw) -> ProfileAutoencoder:
    cfg = ProfileAutoencoderConfig(profile_length=H_ELEV, embedding_dim=EMB_DIM, **kw)
    return ProfileAutoencoder(cfg)


@pytest.mark.parametrize("encoder_kind", ["mlp", "conv1d", "transformer1d"])
def test_autoencoder_shapes_per_pixel(encoder_kind):
    m = make_autoencoder(encoder_kind=encoder_kind, decoder_kind=encoder_kind)
    curve = torch.randn(8, H_ELEV, 1, 1)
    curve_hat, z = m.reconstruct(curve)
    assert curve_hat.shape == (8, H_ELEV, 1, 1)
    assert z.shape         == (8, EMB_DIM, 1, 1)


@pytest.mark.parametrize("encoder_kind", ["mlp", "conv1d", "transformer1d"])
def test_autoencoder_shapes_feature_map(encoder_kind):
    m = make_autoencoder(encoder_kind=encoder_kind, decoder_kind=encoder_kind)
    fmap = torch.randn(2, H_ELEV, 4, 5)
    z = m.encode(fmap)
    assert z.shape          == (2, EMB_DIM, 4, 5)
    assert m.decode(z).shape == (2, H_ELEV, 4, 5)


@pytest.mark.parametrize("kind", ["l2", "layernorm"])
def test_embedding_is_normalized(kind):
    m = make_autoencoder(embedding_norm=kind)
    z = m.encode(torch.rand(5, H_ELEV, 1, 1)).squeeze(-1).squeeze(-1)
    if kind == "l2":
        norms = z.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    else:
        assert torch.allclose(z.mean(dim=1), torch.zeros(z.shape[0]),               atol=1e-4)
        assert torch.allclose(z.std(dim=1, unbiased=False), torch.ones(z.shape[0]),  atol=1e-3)


def test_embedding_norm_none_is_identity():
    m = make_autoencoder(embedding_norm="none")
    z = torch.randn(3, EMB_DIM, 2, 2)
    assert torch.equal(m.normalize_embedding(z), z)


def test_curve_norm_log1p_roundtrip():
    m = make_autoencoder(curve_norm="log1p")
    x = torch.rand(4, H_ELEV, 1, 1) * 50.0
    assert torch.allclose(m.denormalize_curve(m.normalize_curve(x)), x, atol=1e-4)

    m_none = make_autoencoder(curve_norm="none")
    assert torch.equal(m_none.normalize_curve(x), x)


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

    m      = make_autoencoder(hidden_dim=128, depth=3)
    target = m.normalize_curve(curves)
    opt    = torch.optim.Adam(m.parameters(), lr=1e-3)
    loss   = torch.tensor(1.0)
    for _ in range(400):
        opt.zero_grad()
        loss = ((m.decode(m.encode(curves)) - target) ** 2).mean()
        loss.backward()
        opt.step()
    assert loss.item() < 1e-2


def _assert_contract(d):
    assert set(d.keys()) == {"total_loss", "components", "weighted", "monitor"}
    assert torch.isfinite(d["total_loss"])


def test_profile_ae_loss_contract():
    m    = make_autoencoder()
    crit = ProfileAeLoss(AutoencoderLossConfig())

    curve = torch.rand(6, H_ELEV, 1, 1)
    curve_hat, _ = m.reconstruct(curve)
    out = crit(curve_hat, curve)
    _assert_contract(out)
    assert "curve_recon" in out["components"]


def test_jepa_loss_contract_and_backward():
    m      = make_autoencoder()
    tp     = TargetProvider("stopgrad", m.encoder)
    x_axis = torch.linspace(0.0, 1.0, H_ELEV)
    crit   = JepaLoss(m, tp, EmbeddingLossConfig(use_embedding_mse=True), x_axis, IdentityNorm(), 3)

    backbone, _ = get_model("unet", in_channels=4, out_channels=EMB_DIM)
    images = torch.randn(2, 4, 16, 16)
    gt     = torch.rand(2, 3 * N_GAUSS, 16, 16)

    z_hat = backbone(images)
    out   = crit(z_hat, gt)
    _assert_contract(out)
    assert "embedding_mse" in out["components"]
    assert "curve_recon"   in out["components"]
    out["total_loss"].backward()
    assert any(p.grad is not None for p in backbone.parameters())


def test_jepa_loss_stopgrad_blocks_encoder_gradient():
    m      = make_autoencoder()
    tp     = TargetProvider("stopgrad", m.encoder)
    x_axis = torch.linspace(0.0, 1.0, H_ELEV)
    crit   = JepaLoss(m, tp, EmbeddingLossConfig(use_embedding_mse=True, use_curve_recon=False), x_axis, IdentityNorm(), 3)

    z_hat = torch.randn(2, EMB_DIM, 8, 8, requires_grad=True)
    gt    = torch.rand(2, 3 * N_GAUSS, 8, 8)
    crit(z_hat, gt)["total_loss"].backward()

    assert all(p.grad is None for p in m.encoder.parameters())
    assert z_hat.grad is not None


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


def test_inference_model_outputs_curve():
    from pipelines.jepa_pipeline.predictor_trainer import JepaModule
    from pipelines.jepa_pipeline.inference         import JepaInferenceModel
    ae      = make_autoencoder()
    bb, _   = get_model("unet", in_channels=5, out_channels=EMB_DIM)
    adapter = JepaInferenceModel(JepaModule(bb, ae))
    out     = adapter(torch.randn(2, 5, 16, 16))
    assert out.shape == (2, H_ELEV, 16, 16)


def test_metrics_curve_only_excludes_param_space():
    from pathlib import Path
    from pipelines.inference_pipeline.metrics import Metrics, Result
    rng  = np.random.default_rng(0)
    pred = rng.random((H_ELEV, 6, 5)).astype(np.float32)
    gt   = rng.random((H_ELEV, 6, 5)).astype(np.float32)
    pm   = Metrics.curve_pixel_metrics(pred, gt)

    result = Result(
        pred_curves        = pred,
        gt_curves          = gt,
        pixel_mse          = pm["mse"],
        pixel_mae          = pm["mae"],
        pixel_r2           = pm["r2"],
        pixel_cosine       = pm["cos"],
        pixel_peak_err_idx = pm["peak"].astype(np.int32),
        cube_directory     = Path("."),
        azimuth_offset     = 0,
        range_offset       = 0,
    )

    m = Metrics(result, np.linspace(0, 1, H_ELEV), n_gaussians=N_GAUSS).compute(param_space=False)
    for k in ("curve_mse_gt", "pixel_mse_gt_mean", "elev_mae_gt_mean", "psnr_db_gt"):
        assert k in m, f"missing curve/physical metric {k}"
    for k in ("gauss_all_mu_mae", "mu_ordering_rate", "permutation_consensus_dominant_frac", "slot_0_mu_pred_mean"):
        assert k not in m, f"param-space metric {k} leaked into curve-only eval"


def test_profile_dataset_synthesizes_curves():
    from pipelines.autoencoder_pipeline.profile_dataset import ProfileDataset
    rng    = np.random.default_rng(0)
    params = np.zeros((3 * N_GAUSS, 8, 8), dtype=np.float32)
    params[0::3] = rng.uniform(0.0, 1.0, size=(N_GAUSS, 8, 8))
    params[1::3] = rng.uniform(0.0, 1.0, size=(N_GAUSS, 8, 8))
    params[2::3] = rng.uniform(0.1, 0.3, size=(N_GAUSS, 8, 8))
    x_axis = np.linspace(0, 1, H_ELEV, dtype=np.float32)

    ds    = ProfileDataset([params], x_axis, n_gaussians=N_GAUSS, keep_empty_frac=1.0)
    curve = ds[0]
    assert curve.shape == (H_ELEV,)


@pytest.mark.parametrize("provider", ["live", "ema"])
def test_validate_coupling_rejects_live_ema_with_frozen_ae(provider):
    from pipelines.jepa_pipeline.predictor_trainer import JepaPredictorTrainer
    with pytest.raises(ValueError):
        JepaPredictorTrainer.validate_coupling(StageAMode("frozen"), provider, EmbeddingLossConfig())


def test_validate_coupling_rejects_live_without_curve_recon():
    from pipelines.jepa_pipeline.predictor_trainer import JepaPredictorTrainer
    trainable = StageAMode("finetune")
    with pytest.raises(ValueError):
        JepaPredictorTrainer.validate_coupling(trainable, "live", EmbeddingLossConfig(use_curve_recon=False))
    JepaPredictorTrainer.validate_coupling(trainable, "live", EmbeddingLossConfig(use_curve_recon=True))


def test_validate_coupling_accepts_safe_combinations():
    from pipelines.jepa_pipeline.predictor_trainer import JepaPredictorTrainer
    JepaPredictorTrainer.validate_coupling(StageAMode("frozen"),   "stopgrad", EmbeddingLossConfig(use_curve_recon=False))
    JepaPredictorTrainer.validate_coupling(StageAMode("finetune"), "ema",      EmbeddingLossConfig(use_curve_recon=False))


def test_validate_stage_a_checkpoint_guards(tmp_path):
    from pipelines.jepa_pipeline.pipeline import JepaPipeline
    with pytest.raises(ValueError):
        JepaPipeline.validate_stage_a_checkpoint(None, "frozen")
    JepaPipeline.validate_stage_a_checkpoint(None, "finetune")
    JepaPipeline.validate_stage_a_checkpoint(None, "joint")

    missing = tmp_path / "best_model.pt"
    with pytest.raises(FileNotFoundError):
        JepaPipeline.validate_stage_a_checkpoint(str(missing), "frozen")

    missing.write_bytes(b"x")
    JepaPipeline.validate_stage_a_checkpoint(str(missing), "frozen")


def test_stage_a_weights_load_into_stage_b(tmp_path):
    cfg    = ProfileAutoencoderConfig(profile_length=H_ELEV, embedding_dim=EMB_DIM)
    ae_src = ProfileAutoencoder(cfg)
    with torch.no_grad():
        for p in ae_src.parameters():
            p.add_(torch.randn_like(p))

    ckpt_path = tmp_path / "best_model.pt"
    torch.save({"params": ae_src.state_dict()}, ckpt_path)

    ae_dst = ProfileAutoencoder(cfg)
    assert any(not torch.allclose(s, d) for s, d in zip(ae_src.parameters(), ae_dst.parameters()))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ae_dst.load_state_dict(ckpt["params"])
    for s, d in zip(ae_src.parameters(), ae_dst.parameters()):
        assert torch.allclose(s, d)


def test_jepa_curve_recon_target_is_log_space():
    from tools.gaussians                  import GaussianCurve
    from pipelines.autoencoder_pipeline.losses import curve_loss
    m      = make_autoencoder(curve_norm="log1p")
    tp     = TargetProvider("stopgrad", m.encoder)
    x_axis = torch.linspace(0.0, 1.0, H_ELEV)
    crit   = JepaLoss(m, tp, EmbeddingLossConfig(use_embedding_mse=True, use_curve_recon=True), x_axis, IdentityNorm(), 3)

    torch.manual_seed(0)
    z_hat = torch.randn(2, EMB_DIM, 4, 4)
    gt    = torch.rand(2, 3 * N_GAUSS, 4, 4)

    comp = crit(z_hat, gt)["components"]["curve_recon"]

    decoded     = m.decode(m.normalize_embedding(z_hat))
    gt_curve    = GaussianCurve.reconstruct(gt, x_axis, 3).to(decoded.dtype)
    expect_log  = curve_loss(decoded, m.normalize_curve(gt_curve), crit.emb_cfg.curve_kind, crit.emb_cfg.huber_delta, crit.emb_cfg.charbonnier_eps)
    expect_phys = curve_loss(decoded, gt_curve,                    crit.emb_cfg.curve_kind, crit.emb_cfg.huber_delta, crit.emb_cfg.charbonnier_eps)

    assert torch.allclose(comp, expect_log, atol=1e-5)
    assert not torch.allclose(comp, expect_phys, atol=1e-4)


def test_gaussian_curve_matches_mixture_numerics():
    from tools.gaussians import GaussianCurve, GaussianMixture
    rng  = np.random.default_rng(0)
    amps = rng.uniform(0.0, 1.0,  size=(N_GAUSS,)).astype(np.float32)
    mus  = rng.uniform(0.0, 1.0,  size=(N_GAUSS,)).astype(np.float32)
    sigs = rng.uniform(1e-7, 0.3, size=(N_GAUSS,)).astype(np.float32)
    x    = np.linspace(0, 1, H_ELEV, dtype=np.float32)

    mix = GaussianMixture.evaluate_batch(x, amps[None, :], mus[None, :], sigs[None, :])[0]

    params = np.empty(3 * N_GAUSS, dtype=np.float32)
    params[0::3], params[1::3], params[2::3] = amps, mus, sigs
    pt    = torch.tensor(params).reshape(1, 3 * N_GAUSS, 1, 1)
    curve = GaussianCurve.reconstruct(pt, torch.tensor(x), 3).reshape(H_ELEV).numpy()

    assert np.isfinite(curve).all()
    assert np.allclose(curve, mix, atol=1e-5)
