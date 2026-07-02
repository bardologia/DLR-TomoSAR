from __future__ import annotations

import numpy as np
import pytest
import torch

from configuration.training.general.loss import LossConfig
from pipelines.backbone.training.loss       import Loss
from pipelines.backbone.training.loss_terms import LOSS_TERMS, LossComponentCatalog

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


def test_catalog_names_match_loss_terms():
    assert LossComponentCatalog.names() == tuple(term.name for term in LOSS_TERMS)


def test_catalog_standalone_enables_only_the_named_term():
    for term in LOSS_TERMS:
        cfg = LossComponentCatalog.standalone(term.name)

        assert getattr(cfg, term.use_flag)   is True
        assert getattr(cfg, term.weight_key) == 1.0

        other_flags_off = [not getattr(cfg, other.use_flag) for other in LOSS_TERMS if other.name != term.name]
        assert all(other_flags_off)


def test_catalog_rejects_unknown_component():
    with pytest.raises(KeyError):
        LossComponentCatalog.standalone("not_a_real_loss")


def test_catalog_standalone_inherits_shared_knobs_from_base():
    from configuration.training import LossConfig, ParamMatching

    base = LossConfig(param_matching=ParamMatching.SORTED_GT, param_weights=(2.0, 3.0, 4.0), use_mse_curve=True, weight_mse_curve=5.0)
    cfg  = LossComponentCatalog.standalone("param_l1", base=base)

    assert cfg.use_param_l1   is True
    assert cfg.weight_param_l1 == 1.0
    assert cfg.param_matching == ParamMatching.SORTED_GT
    assert cfg.param_weights  == (2.0, 3.0, 4.0)
    assert cfg.use_mse_curve  is False


def test_catalog_curriculum_is_disabled_with_matching_phases():
    curriculum = LossComponentCatalog.curriculum("covariance_match")

    assert curriculum.enabled is False
    assert curriculum.warmup.use_covariance_match   is True
    assert curriculum.complete.use_covariance_match is True


def test_catalog_combined_enables_the_union_of_terms():
    cfg = LossComponentCatalog.combined(["param_l1", "covariance_match", "l1_curve"])

    assert cfg.use_param_l1         is True
    assert cfg.use_covariance_match is True
    assert cfg.use_l1_curve         is True

    other_flags_off = [not getattr(cfg, term.use_flag) for term in LOSS_TERMS if term.name not in ("param_l1", "covariance_match", "l1_curve")]
    assert all(other_flags_off)


def test_catalog_combined_rejects_empty_and_unknown():
    with pytest.raises(ValueError):
        LossComponentCatalog.combined([])

    with pytest.raises(KeyError):
        LossComponentCatalog.combined(["param_l1", "not_a_real_loss"])


def test_catalog_combined_curriculum_mirrors_combined_in_both_phases():
    curriculum = LossComponentCatalog.combined_curriculum(["param_l1", "coherence_resyn"])

    assert curriculum.enabled is False
    assert curriculum.warmup.use_param_l1          is True
    assert curriculum.warmup.use_coherence_resyn   is True
    assert curriculum.complete.use_param_l1        is True
    assert curriculum.complete.use_coherence_resyn is True


def test_forward_returns_finite_scalar_total():
    loss = build_loss(n_gaussians=2)
    pred = param_tensor(2, 2, 6, 6, seed=0)
    gt   = param_tensor(2, 2, 6, 6, seed=1)

    out  = loss(pred, gt)

    assert set(out.keys()) == {"total_loss", "components", "weighted", "monitor", "occupancy"}
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
    cfg    = LossConfig(use_param_l1=True, weight_param_l1=1.0)
    loss   = build_loss(n_gaussians=2, loss_cfg=cfg)
    params = valid_param_tensor(2, 2, 6, 6, seed=6)

    out    = loss(params, params.clone())

    assert out["total_loss"].item() == pytest.approx(0.0, abs=1e-5)


def test_prepare_clamps_prediction_so_invalid_identical_params_are_nonzero():
    cfg    = LossConfig(use_param_l1=True, weight_param_l1=1.0)
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

    expected = cfg.weight_param_l1 * out["components"]["param_l1"].item()

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

    assert "gt_active_frac"    in out["occupancy"]
    assert "pred_active_frac"  in out["occupancy"]
    assert "pred_active_slot0" in out["occupancy"]
    assert "pred_active_slot1" in out["occupancy"]
    assert "gt_active_slot0"   in out["occupancy"]
    assert out["occupancy"]["gt_active_frac"].item() == pytest.approx(1.0)


def test_occupancy_absent_when_no_slot_presence_knobs():
    loss = build_loss(n_gaussians=2)
    pred = valid_param_tensor(2, 2, 5, 5, seed=23)
    gt   = valid_param_tensor(2, 2, 5, 5, seed=24)

    out  = loss(pred, gt)

    assert out["occupancy"] == {}


def test_count_metrics_exact_when_pred_matches_gt():
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0, presence_balance=True)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg)
    gt   = valid_param_tensor(2, 2, 5, 5, seed=31)

    out  = loss(gt.clone(), gt)

    assert out["occupancy"]["count/exact_frac"].item() == pytest.approx(1.0)
    assert out["occupancy"]["count/under_frac"].item() == pytest.approx(0.0)
    assert out["occupancy"]["count/over_frac"].item()  == pytest.approx(0.0)
    assert out["occupancy"]["count/acc_gt2"].item()    == pytest.approx(1.0)
    assert "count/acc_gt1" not in out["occupancy"]


def test_count_metrics_under_when_pred_drops_a_slot():
    cfg  = LossConfig(use_param_l1=True, weight_param_l1=1.0, presence_balance=True)
    loss = build_loss(n_gaussians=2, loss_cfg=cfg)
    gt   = valid_param_tensor(2, 2, 5, 5, seed=32)
    pred = gt.clone()

    pred[:, 3] = 0.0

    out  = loss(pred, gt)

    assert out["occupancy"]["count/under_frac"].item() == pytest.approx(1.0)
    assert out["occupancy"]["count/exact_frac"].item() == pytest.approx(0.0)
    assert out["occupancy"]["count/over_frac"].item()  == pytest.approx(0.0)
    assert out["occupancy"]["count/acc_gt2"].item()    == pytest.approx(0.0)


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
    assert "pred_presence_frac" in out["occupancy"]


def test_curriculum_swap_changes_active_terms():
    loss = build_loss(n_gaussians=2)

    before = loss(param_tensor(2, 2, 5, 5, seed=13), param_tensor(2, 2, 5, 5, seed=14))
    assert "param_l1" in before["components"]

    swap = LossConfig(use_mse_curve=True, weight_mse_curve=1.0)
    loss.set_curriculum(swap)

    assert loss.loss_generation == 1

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

    weight_sum   = cfg.weight_param_l1 + cfg.weight_mse_curve
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
        loss_cfg     = LossConfig(use_param_l1=True, weight_param_l1=1.0),
        norm_stats   = identity_normalizer(n_channels),
        geometry_cfg = geometry_config(),
    )

    noisy = loss(pred, gt)
    clean = loss(gt, gt.clone())

    assert torch.isfinite(noisy["total_loss"]).item()
    assert torch.isfinite(clean["total_loss"]).item()
    assert noisy["total_loss"].item() > clean["total_loss"].item()
