from __future__ import annotations

import numpy as np
import pytest
import torch

from configuration.training.general.loss import LossConfig
from pipelines.backbone.training.loss    import LOSS_TERMS, Loss

from tests.backbone_training._helpers import build_loss, gaussian_config, geometry_config, identity_normalizer, param_tensor, valid_param_tensor, x_axis_tensor

import tools


def test_loss_terms_table_is_consistent():
    names = [term.name for term in LOSS_TERMS]

    assert len(names) == len(set(names))
    assert "param_l1"         in names
    assert "covariance_match" in names

    for term in LOSS_TERMS:
        assert term.use_flag.startswith("use_")
        assert term.weight_key.startswith("weight_")
        assert term.space in ("norm", "denorm")


def test_forward_returns_finite_scalar_total():
    loss = build_loss(n_gaussians=2)
    pred = param_tensor(2, 2, 6, 6, seed=0)
    gt   = param_tensor(2, 2, 6, 6, seed=1)

    out  = loss(pred, gt)

    assert set(out.keys()) == {"total_loss", "components", "weighted", "monitor"}
    assert out["total_loss"].ndim == 0
    assert torch.isfinite(out["total_loss"]).item()


def test_forward_output_dict_keys_for_active_term():
    loss = build_loss(n_gaussians=2)
    pred = param_tensor(2, 2, 5, 5, seed=2)
    gt   = param_tensor(2, 2, 5, 5, seed=3)

    out  = loss(pred, gt)

    assert "param_l1"     in out["components"]
    assert "param_l1/amp" in out["components"]
    assert "param_l1/mu"  in out["components"]


def test_gradient_flows_to_predictions():
    loss = build_loss(n_gaussians=2)
    pred = param_tensor(2, 2, 6, 6, seed=4).requires_grad_(True)
    gt   = param_tensor(2, 2, 6, 6, seed=5)

    loss(pred, gt)["total_loss"].backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all().item()
    assert pred.grad.abs().sum().item() > 0.0


def test_identical_valid_params_give_zero_param_loss():
    cfg    = LossConfig(use_param_l1=True, weight_param_l1=1.0, param_match="none")
    loss   = build_loss(n_gaussians=2, loss_cfg=cfg)
    params = valid_param_tensor(2, 2, 6, 6, seed=6)

    out    = loss(params, params.clone())

    assert out["total_loss"].item() == pytest.approx(0.0, abs=1e-5)


def test_prepare_clamps_prediction_so_invalid_identical_params_are_nonzero():
    cfg    = LossConfig(use_param_l1=True, weight_param_l1=1.0, param_match="none")
    loss   = build_loss(n_gaussians=2, loss_cfg=cfg)
    params = param_tensor(2, 2, 6, 6, seed=30)

    out    = loss(params, params.clone())

    assert out["total_loss"].item() > 0.0


def test_weighted_components_scale_with_eff_weight():
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=2.0)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg)
    pred = param_tensor(2, 2, 6, 6, seed=7)
    gt   = param_tensor(2, 2, 6, 6, seed=8)

    out  = loss(pred, gt)

    eff      = cfg.eff("weight_param_l1")
    expected = eff * out["components"]["param_l1"].item()

    assert out["weighted"]["param_l1"].item() == pytest.approx(expected, rel=1e-5)


def test_curve_term_wiring_mse():
    cfg  = LossConfig(use_mse_curve=True, weight_mse_curve=1.0)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg)
    pred = param_tensor(2, 2, 5, 5, seed=9)
    gt   = param_tensor(2, 2, 5, 5, seed=10)

    out  = loss(pred, gt)

    assert "mse_curve" in out["components"]
    assert torch.isfinite(out["total_loss"]).item()


def test_log_all_losses_populates_monitor():
    loss = build_loss(n_gaussians=2, log_all_losses=True)
    pred = param_tensor(2, 2, 5, 5, seed=11)
    gt   = param_tensor(2, 2, 5, 5, seed=12)

    out  = loss(pred, gt)

    monitor_terms = {key.rsplit("_", 1)[0] for key in out["monitor"]}

    assert "mse_curve"   in monitor_terms
    assert "param_huber" in monitor_terms
    assert len(out["monitor"]) > len(out["components"])


def test_slot_presence_knobs_log_occupancy():
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0, presence_balance=True, active_weight=2.0, inactive_weight=0.5)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg)
    pred = valid_param_tensor(2, 2, 5, 5, seed=21)
    gt   = valid_param_tensor(2, 2, 5, 5, seed=22)

    out  = loss(pred, gt)

    assert "occupancy/gt_active_frac"   in out["monitor"]
    assert "occupancy/pred_active_frac" in out["monitor"]
    assert "occupancy/pred_active_slot0" in out["monitor"]
    assert "occupancy/pred_active_slot1" in out["monitor"]
    assert "occupancy/gt_active_slot0"   in out["monitor"]
    assert out["monitor"]["occupancy/gt_active_frac"].item() == pytest.approx(1.0)


def test_occupancy_absent_when_no_slot_presence_knobs():
    loss = build_loss(n_gaussians=2)
    pred = valid_param_tensor(2, 2, 5, 5, seed=23)
    gt   = valid_param_tensor(2, 2, 5, 5, seed=24)

    out  = loss(pred, gt)

    assert not any(key.startswith("occupancy/") for key in out["monitor"])


def test_presence_bce_without_head_raises():
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0, use_presence_bce=True, weight_presence_bce=1.0)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg)
    pred = valid_param_tensor(2, 2, 5, 5, seed=25)
    gt   = valid_param_tensor(2, 2, 5, 5, seed=26)

    with pytest.raises(ValueError, match="predict_presence"):
        loss(pred, gt)


def test_presence_bce_with_head_logs_component_and_occupancy():
    cfg          = LossConfig(use_param_l1=True, weight_param_l1=1.0, use_presence_bce=True, weight_presence_bce=1.0)
    loss         = build_loss(n_gaussians=2, loss_cfg=cfg)
    params       = valid_param_tensor(2, 2, 5, 5, seed=27)
    presence     = torch.zeros(2, 2, 5, 5, dtype=torch.float32)
    pred         = torch.cat([params, presence], dim=1)
    gt           = valid_param_tensor(2, 2, 5, 5, seed=28)

    out          = loss(pred, gt)

    assert "presence_bce" in out["components"]
    assert "presence_bce" in out["weighted"]
    assert "occupancy/pred_presence_frac" in out["monitor"]


def test_curriculum_swap_changes_active_terms():
    loss = build_loss(n_gaussians=2)

    before = loss(param_tensor(2, 2, 5, 5, seed=13), param_tensor(2, 2, 5, 5, seed=14))
    assert "param_l1" in before["components"]

    swap = LossConfig(use_mse_curve=True, weight_mse_curve=1.0, param_match="none")
    loss.set_curriculum(swap)

    assert loss.loss_generation == 1
    assert loss.match_strategy  == "none"

    after = loss(param_tensor(2, 2, 5, 5, seed=15), param_tensor(2, 2, 5, 5, seed=16))

    assert "mse_curve" in after["components"]
    assert "param_l1"  not in after["components"]


def test_geometry_coupled_covariance_term_uses_geometry():
    cfg  = LossConfig(use_covariance_match=True, weight_covariance_match=1.0)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg)

    assert loss.geometry.outer.shape[0] == loss.geometry.n_tracks
    assert loss.geometry.steering.shape[0] == loss.geometry.n_tracks

    pred = param_tensor(2, 2, 5, 5, seed=17).requires_grad_(True)
    gt   = param_tensor(2, 2, 5, 5, seed=18)

    out  = loss(pred, gt)

    assert "covariance_match" in out["components"]
    out["total_loss"].backward()
    assert torch.isfinite(pred.grad).all().item()


def test_geometry_coupled_coherence_resyn_term():
    cfg  = LossConfig(use_coherence_resyn=True, weight_coherence_resyn=1.0)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg)
    pred = param_tensor(2, 2, 5, 5, seed=19)
    gt   = param_tensor(2, 2, 5, 5, seed=20)

    out  = loss(pred, gt)

    assert "coherence_resyn" in out["components"]
    assert out["total_loss"].item() >= 0.0


def test_total_loss_is_weighted_normalised_mean():
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0, use_mse_curve=True, weight_mse_curve=1.0)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg)
    pred = param_tensor(2, 2, 5, 5, seed=21)
    gt   = param_tensor(2, 2, 5, 5, seed=22)

    out  = loss(pred, gt)

    weight_sum   = cfg.eff("weight_param_l1") + cfg.eff("weight_mse_curve")
    summed       = sum(v.item() for k, v in out["weighted"].items() if "/" not in k)
    expected     = summed / weight_sum

    assert out["total_loss"].item() == pytest.approx(expected, rel=1e-4)


def test_param_count_mismatch_pads_param_weights():
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0, param_weights=(1.0,))
    loss = build_loss(n_gaussians=3, loss_cfg=cfg)
    pred = param_tensor(2, 3, 5, 5, seed=23)
    gt   = param_tensor(2, 3, 5, 5, seed=24)

    out  = loss(pred, gt)

    assert torch.isfinite(out["total_loss"]).item()


@pytest.mark.real_data
def test_loss_on_real_tomogram_param_target(parameters):
    win        = np.asarray(parameters[:15, :8, :8]).astype(np.float32)
    n_channels = win.shape[0]
    n_gaussians = n_channels // 3

    gt   = torch.from_numpy(win)[None]
    pred = gt.clone() + 0.05 * torch.randn_like(gt)

    loss = Loss(
        x_axis       = x_axis_tensor(),
        logger       = tools.NullLogger(),
        tracker      = tools.NullTracker(),
        gaussian_cfg = gaussian_config(n_gaussians),
        loss_cfg     = LossConfig(use_param_l1=True, weight_param_l1=1.0, param_match="none"),
        norm_stats   = identity_normalizer(n_channels),
        geometry_cfg = geometry_config(),
    )

    noisy = loss(pred, gt)
    clean = loss(gt, gt.clone())

    assert torch.isfinite(noisy["total_loss"]).item()
    assert torch.isfinite(clean["total_loss"]).item()
    assert noisy["total_loss"].item() > clean["total_loss"].item()
