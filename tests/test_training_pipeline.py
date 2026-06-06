from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from configuration.training_config import (
    EarlyStoppingConfig,
    EMAConfig,
    GaussianConfig,
    GeometryConfig,
    GradientClipperConfig,
    LossConfig,
    LossCurriculumConfig,
    OverfitConfig,
    SchedulerConfig,
    WarmupConfig,
)
from pipelines.training_pipeline.callbacks import (
    EMA,
    EarlyStopping,
    GradientClipper,
    Scheduler,
    Warmup,
)
from pipelines.training_pipeline.control import (
    Checkpoint,
    CurriculumController,
    OverfitManager,
)
from pipelines.training_pipeline.docs import (
    LossScaleProbe,
    LossScaleProbeConfig,
    LayerRecord,
    ModelInspector,
    TrainingDocs,
)
from pipelines.training_pipeline.loss import (
    Loss,
    LossComponents,
    ParamMatcher,
    PhysicsComponents,
)
from pipelines.training_pipeline.trainer import MetricAggregator, TrainStep
from tools import NullLogger, NullTracker


def seed_all(value: int = 0) -> None:
    torch.manual_seed(value)
    np.random.seed(value)


class IdentityNorm:
    def normalize_output(self, tensor):
        return tensor

    def denormalize_output(self, tensor):
        return tensor


class DebugTracker(NullTracker):
    def __init__(self):
        super().__init__()
        self.debug = True


def warmup_config_namespace(warmup_cfg: WarmupConfig) -> SimpleNamespace:
    return SimpleNamespace(warmup=warmup_cfg)


def scheduler_config_namespace(scheduler_cfg: SchedulerConfig) -> SimpleNamespace:
    return SimpleNamespace(scheduler=scheduler_cfg)


def early_stopping_config_namespace(es_cfg: EarlyStoppingConfig) -> SimpleNamespace:
    return SimpleNamespace(early_stopping=es_cfg)


def ema_config_namespace(ema_cfg: EMAConfig) -> SimpleNamespace:
    return SimpleNamespace(ema=ema_cfg)


def clipper_config_namespace(clipper_cfg: GradientClipperConfig) -> SimpleNamespace:
    return SimpleNamespace(gradient_clipper=clipper_cfg)


def overfit_config_namespace(overfit_cfg: OverfitConfig) -> SimpleNamespace:
    return SimpleNamespace(overfit=overfit_cfg)


def make_loss(loss_cfg: LossConfig, n_points: int = 21, n_gaussians: int = 2) -> Loss:
    x_axis = torch.linspace(-10.0, 10.0, n_points)
    gcfg   = GaussianConfig(
        n_default_gaussians = n_gaussians,
        x_min               = -10.0,
        x_max               = 10.0,
        amp_max             = 10.0,
    )
    return Loss(
        x_axis,
        NullLogger(),
        NullTracker(),
        gcfg,
        loss_cfg,
        norm_stats   = IdentityNorm(),
        geometry_cfg = GeometryConfig(),
    )


class TestParamMatcher:
    def test_silent_constructor_sets_strategy(self):
        matcher = ParamMatcher.silent("none")

        assert matcher.strategy == "none"

    def test_constructor_with_logger_runs(self):
        matcher = ParamMatcher("sort_gt_by_mu", logger=NullLogger())

        assert matcher.strategy == "sort_gt_by_mu"

    def test_match_torch_none_strategy_is_identity(self):
        seed_all()
        matcher = ParamMatcher.silent("none")
        pred      = torch.randn(2, 2, 3, 4, 4)
        pred_phys = torch.randn(2, 2, 3, 4, 4)
        gt        = torch.randn(2, 2, 3, 4, 4)
        gt_phys   = torch.randn(2, 2, 3, 4, 4)

        out = matcher.match_torch(pred, pred_phys, gt, gt_phys)

        assert torch.equal(out[0], pred)
        assert torch.equal(out[2], gt)
        assert torch.equal(out[3], gt_phys)

    def test_match_torch_sorts_gt_by_mu_ascending(self):
        pred      = torch.zeros(1, 2, 3, 1, 1)
        pred_phys = torch.zeros(1, 2, 3, 1, 1)
        gt        = torch.zeros(1, 2, 3, 1, 1)
        gt_phys   = torch.zeros(1, 2, 3, 1, 1)

        gt[0, 0, 1, 0, 0] = 5.0
        gt[0, 1, 1, 0, 0] = -2.0
        gt_phys[0, 0, 0, 0, 0] = 1.0
        gt_phys[0, 1, 0, 0, 0] = 1.0

        matcher = ParamMatcher.silent("sort_gt_by_mu")
        _, _, gt_sorted, _ = matcher.match_torch(pred, pred_phys, gt, gt_phys)

        assert float(gt_sorted[0, 0, 1, 0, 0]) == -2.0
        assert float(gt_sorted[0, 1, 1, 0, 0]) == 5.0

    def test_match_torch_inactive_components_pushed_last(self):
        pred      = torch.zeros(1, 2, 3, 1, 1)
        pred_phys = torch.zeros(1, 2, 3, 1, 1)
        gt        = torch.zeros(1, 2, 3, 1, 1)
        gt_phys   = torch.zeros(1, 2, 3, 1, 1)

        gt[0, 0, 1, 0, 0] = 8.0
        gt[0, 1, 1, 0, 0] = -8.0
        gt_phys[0, 0, 0, 0, 0] = 1.0
        gt_phys[0, 1, 0, 0, 0] = 0.0

        matcher = ParamMatcher.silent("sort_gt_by_mu")
        _, _, gt_sorted, _ = matcher.match_torch(pred, pred_phys, gt, gt_phys)

        assert float(gt_sorted[0, 0, 1, 0, 0]) == 8.0


class TestLossComponents:
    def test_mse_matches_torch_reference(self):
        seed_all()
        pred   = torch.randn(2, 3, 4, 4)
        target = torch.randn(2, 3, 4, 4)

        assert torch.allclose(LossComponents.mse(pred, target), ((pred - target) ** 2).mean())

    def test_mse_diff_equals_mse(self):
        seed_all()
        pred   = torch.randn(2, 3, 4, 4)
        target = torch.randn(2, 3, 4, 4)

        diff = pred - target

        assert torch.allclose(LossComponents.mse_diff(diff), LossComponents.mse(pred, target))

    def test_l1_matches_torch_reference(self):
        seed_all()
        pred   = torch.randn(2, 3, 4, 4)
        target = torch.randn(2, 3, 4, 4)

        assert torch.allclose(LossComponents.l1(pred, target), (pred - target).abs().mean())

    def test_l1_diff_equals_l1(self):
        seed_all()
        pred   = torch.randn(2, 3, 4, 4)
        target = torch.randn(2, 3, 4, 4)

        diff = pred - target

        assert torch.allclose(LossComponents.l1_diff(diff), LossComponents.l1(pred, target))

    def test_huber_diff_matches_huber(self):
        seed_all()
        pred   = torch.randn(2, 3, 4, 4)
        target = torch.randn(2, 3, 4, 4)

        diff = pred - target

        assert torch.allclose(LossComponents.huber_diff(diff, 1.0), LossComponents.huber(pred, target, 1.0), atol=1e-6)

    def test_huber_quadratic_for_small_diff(self):
        diff = torch.tensor([0.1, -0.2])

        result = LossComponents.huber_diff(diff, delta=1.0)
        expected = (0.5 * diff * diff).mean()

        assert torch.allclose(result, expected)

    def test_huber_linear_for_large_diff(self):
        diff  = torch.tensor([10.0])
        delta = 1.0

        result   = LossComponents.huber_diff(diff, delta=delta)
        expected = delta * (diff.abs() - 0.5 * delta)

        assert torch.allclose(result, expected.mean())

    def test_charbonnier_zero_diff_is_eps(self):
        diff = torch.zeros(4)
        eps  = 1e-3

        result = LossComponents.charbonnier_diff(diff, eps)

        assert torch.allclose(result, torch.tensor(eps))

    def test_charbonnier_matches_diff_helper(self):
        seed_all()
        pred   = torch.randn(2, 3)
        target = torch.randn(2, 3)

        assert torch.allclose(LossComponents.charbonnier(pred, target, 1e-3), LossComponents.charbonnier_diff(pred - target, 1e-3))

    def test_cosine_zero_for_identical(self):
        seed_all()
        vec = torch.randn(2, 5, 3, 3).abs() + 0.5

        result = LossComponents.cosine(vec, vec.clone(), axis=1)

        assert torch.allclose(result, torch.zeros(()), atol=1e-5)

    def test_cosine_in_zero_one_range(self):
        seed_all()
        pred   = torch.randn(2, 5, 3, 3)
        target = torch.randn(2, 5, 3, 3)

        result = LossComponents.cosine(pred, target, axis=1)

        assert 0.0 <= float(result) <= 2.0 + 1e-6

    def test_spectral_coherence_zero_for_identical(self):
        seed_all()
        curve = torch.randn(1, 16, 2, 2).abs() + 0.5

        result = LossComponents.spectral_coherence(curve, curve.clone(), window=4)

        assert float(result) < 1e-4

    def test_tv_zero_for_constant_field(self):
        params = torch.ones(1, 3, 5, 5)

        assert torch.allclose(LossComponents.tv(params), torch.zeros(()))

    def test_tv_positive_for_varying_field(self):
        seed_all()
        params = torch.randn(1, 3, 5, 5)

        assert float(LossComponents.tv(params)) > 0.0

    def test_gaussian_kernel_normalised_and_shape(self):
        kernel = LossComponents.gaussian_kernel(11, 1.5, torch.float32, torch.device("cpu"))

        assert kernel.shape == (1, 1, 11, 11)
        assert torch.allclose(kernel.sum(), torch.ones(()), atol=1e-5)

    def test_ssim_zero_for_identical(self):
        seed_all()
        cfg = LossConfig(use_ssim_curve=True)
        x   = torch.randn(2, 8, 6, 6)

        result = LossComponents.ssim(x, x.clone(), cfg)

        assert float(result) < 1e-3

    def test_ssim_invalid_axis_raises(self):
        seed_all()
        cfg          = LossConfig()
        cfg.ssim_axis = "diagonal"
        x            = torch.randn(2, 8, 6, 6)

        with pytest.raises(ValueError, match="ssim_axis"):
            LossComponents.ssim(x, x.clone(), cfg)

    @pytest.mark.parametrize("axis", ["elevation", "azimuth", "range"])
    def test_ssim_runs_for_each_axis(self, axis):
        seed_all()
        cfg          = LossConfig()
        cfg.ssim_axis = axis
        pred   = torch.randn(2, 8, 6, 6)
        target = torch.randn(2, 8, 6, 6)

        result = LossComponents.ssim(pred, target, cfg)

        assert torch.isfinite(result)

    def test_param_l1_total_and_per_param(self):
        seed_all()
        pred    = torch.randn(2, 2, 3, 4, 4)
        gt      = torch.randn(2, 2, 3, 4, 4)
        weights = torch.ones(1, 1, 3, 1, 1)

        total, per_param = LossComponents.param_l1(pred, gt, weights, ["amp", "mu", "sigma"])

        assert set(per_param.keys()) == {"amp", "mu", "sigma"}
        assert torch.allclose(total, (weights * (pred - gt).abs()).mean())

    def test_param_huber_nonnegative(self):
        seed_all()
        pred    = torch.randn(2, 2, 3, 4, 4)
        gt      = torch.randn(2, 2, 3, 4, 4)
        weights = torch.ones_like(pred)

        result = LossComponents.param_huber(pred, gt, weights, delta=0.5)

        assert float(result) >= 0.0


class TestPhysicsComponents:
    def test_masked_mean_full_mask(self):
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask   = torch.ones_like(values)

        assert torch.allclose(PhysicsComponents.masked_mean(values, mask), values.mean())

    def test_masked_mean_empty_mask_is_zero(self):
        values = torch.tensor([[1.0, 2.0]])
        mask   = torch.zeros_like(values)

        assert torch.allclose(PhysicsComponents.masked_mean(values, mask), torch.zeros(()))

    def test_moment_sums_shapes(self):
        seed_all()
        curves = torch.randn(2, 16, 3, 3).abs()
        x_axis = torch.linspace(-5.0, 5.0, 16)

        s0, s1, s2 = PhysicsComponents.moment_sums(curves, x_axis, dx=1.0)

        assert s0.shape == (2, 3, 3)
        assert s1.shape == (2, 3, 3)
        assert s2.shape == (2, 3, 3)

    def test_total_power_zero_for_identical(self):
        seed_all()
        curve = torch.randn(2, 16, 3, 3).abs() + 1.0

        result = PhysicsComponents.total_power(curve, curve.clone(), dx=1.0, floor=1e-3)

        assert float(result) < 1e-5

    def test_moments_zero_for_identical(self):
        seed_all()
        curve  = torch.randn(2, 16, 3, 3).abs() + 1.0
        x_axis = torch.linspace(-5.0, 5.0, 16)

        result = PhysicsComponents.moments(curve, curve.clone(), x_axis, dx=1.0, floor=1e-3, weights=(1.0, 1.0, 1.0))

        assert float(result) < 1e-5

    def test_moments_zero_weights_does_not_divide_by_zero(self):
        seed_all()
        pred   = torch.randn(2, 16, 3, 3).abs() + 1.0
        target = torch.randn(2, 16, 3, 3).abs() + 1.0
        x_axis = torch.linspace(-5.0, 5.0, 16)

        result = PhysicsComponents.moments(pred, target, x_axis, dx=1.0, floor=1e-3, weights=(0.0, 0.0, 0.0))

        assert torch.isfinite(result)

    def test_coherence_resynthesis_zero_for_identical(self):
        seed_all()
        geom   = make_loss(LossConfig(use_coherence_resyn=True), n_points=16).geometry
        curve  = torch.randn(2, 16, 3, 3).abs() + 1.0

        result = PhysicsComponents.coherence_resynthesis(curve, curve.clone(), geom.steering, dx=1.0, floor=1e-3)

        assert float(result) < 1e-4

    def test_covariance_matching_zero_for_identical(self):
        seed_all()
        geom  = make_loss(LossConfig(use_covariance_match=True), n_points=16).geometry
        curve = torch.randn(2, 16, 3, 3).abs() + 1.0

        result = PhysicsComponents.covariance_matching(curve, curve.clone(), geom.outer, dx=1.0, floor=1e-3)

        assert float(result) < 1e-5

    def test_capon_cycle_finite(self):
        seed_all()
        loss  = make_loss(LossConfig(use_capon_cycle=True), n_points=16)
        geom  = loss.geometry
        pred   = torch.randn(2, 16, 3, 3).abs() + 1.0
        target = torch.randn(2, 16, 3, 3).abs() + 1.0

        result = PhysicsComponents.capon_cycle(pred, target, geom.steering, geom.outer, dx=1.0, loading=1e-2, floor=1e-3)

        assert torch.isfinite(result)
        assert float(result) >= 0.0


class TestLossReconstruct:
    def test_reconstruct_shape(self):
        loss   = make_loss(LossConfig(), n_points=21, n_gaussians=2)
        params = torch.randn(2, 6, 3, 3)

        curves = loss.reconstruct(params)

        assert curves.shape == (2, 21, 3, 3)

    def test_reconstruct_peak_at_mu(self):
        loss   = make_loss(LossConfig(), n_points=41, n_gaussians=1)
        params = torch.zeros(1, 3, 1, 1)
        params[0, 0, 0, 0] = 2.0
        params[0, 1, 0, 0] = 0.0
        params[0, 2, 0, 0] = 1.0

        curves   = loss.reconstruct(params)
        peak_idx = int(curves[0, :, 0, 0].argmax())

        assert abs(float(loss.x_axis[peak_idx])) < 1e-5
        assert abs(float(curves[0, peak_idx, 0, 0]) - 2.0) < 1e-4

    def test_reconstruct_rejects_indivisible_channels(self):
        loss   = make_loss(LossConfig(), n_gaussians=2)
        params = torch.randn(2, 5, 3, 3)

        with pytest.raises(AssertionError):
            loss.reconstruct(params)


class TestLossCall:
    def _inputs(self, n_gaussians: int = 2):
        seed_all()
        C    = n_gaussians * 3
        pred = torch.randn(2, C, 3, 3)
        gt   = torch.randn(2, C, 3, 3)
        return pred, gt

    def test_mse_only_returns_scalar_and_components(self):
        loss = make_loss(LossConfig(use_mse_curve=True, weight_mse_curve=1.0))
        pred, gt = self._inputs()

        out = loss(pred, gt)

        assert out["total_loss"].ndim == 0
        assert "mse_curve" in out["components"]
        assert torch.isfinite(out["total_loss"])

    def test_total_loss_is_weighted_average(self):
        loss = make_loss(LossConfig(use_mse_curve=True, weight_mse_curve=2.0, use_l1_curve=True, weight_l1_curve=3.0))
        pred, gt = self._inputs()

        out = loss(pred, gt)

        eff_mse = loss.loss_cfg.eff("weight_mse_curve")
        eff_l1  = loss.loss_cfg.eff("weight_l1_curve")
        manual  = (eff_mse * out["components"]["mse_curve"] + eff_l1 * out["components"]["l1_curve"]) / (eff_mse + eff_l1)

        assert torch.allclose(out["total_loss"], manual, atol=1e-6)

    def test_no_active_terms_returns_zero(self):
        loss = make_loss(LossConfig())
        pred, gt = self._inputs()

        out = loss(pred, gt)

        assert float(out["total_loss"]) == 0.0
        assert out["components"] == {}

    def test_param_l1_emits_per_param_components(self):
        loss = make_loss(LossConfig(use_param_l1=True, weight_param_l1=1.0))
        pred, gt = self._inputs()

        out = loss(pred, gt)

        assert any(k.startswith("param_l1/") for k in out["components"])

    def test_log_all_losses_populates_monitor(self):
        loss = make_loss(LossConfig(use_mse_curve=True))
        loss.log_all_losses = True
        pred, gt = self._inputs()

        out = loss(pred, gt)

        assert len(out["monitor"]) > 0

    def test_gradient_flows_to_prediction(self):
        loss = make_loss(LossConfig(use_mse_curve=True, weight_mse_curve=1.0))
        pred, gt = self._inputs()
        pred.requires_grad_(True)

        out = loss(pred, gt)
        out["total_loss"].backward()

        assert pred.grad is not None
        assert torch.all(torch.isfinite(pred.grad))

    def test_param_huber_term_runs(self):
        loss = make_loss(LossConfig(use_param_huber=True, weight_param_huber=1.0))
        pred, gt = self._inputs()

        out = loss(pred, gt)

        assert "param_huber" in out["components"]

    def test_smoothness_tv_term_runs(self):
        loss = make_loss(LossConfig(use_smoothness_tv=True, weight_smoothness_tv=1.0))
        pred, gt = self._inputs()

        out = loss(pred, gt)

        assert "smoothness_tv" in out["components"]

    def test_set_curriculum_swaps_config(self):
        loss        = make_loss(LossConfig(use_mse_curve=True))
        complete    = LossConfig(use_l1_curve=True, param_match="none")
        loss.set_curriculum(complete)

        assert loss.loss_cfg is complete
        assert loss.matcher.strategy == "none"


class TestWarmup:
    def _make(self, **overrides) -> Warmup:
        cfg = WarmupConfig(**overrides)
        return Warmup(warmup_config_namespace(cfg), NullLogger(), NullTracker())

    def test_factor_disabled_is_one(self):
        warm = self._make(warmup_enabled=False)

        assert warm.factor() == 1.0

    def test_factor_at_start_is_start_factor_linear(self):
        warm = self._make(warmup_mode="linear", warmup_start_factor=0.1, warmup_steps=10)

        assert math.isclose(warm.factor(), 0.1, rel_tol=1e-6)

    def test_factor_after_steps_reaches_one(self):
        warm = self._make(warmup_mode="linear", warmup_steps=5)
        warm.current_step = 5

        assert warm.factor() == 1.0

    def test_factor_linear_midpoint(self):
        warm = self._make(warmup_mode="linear", warmup_start_factor=0.0, warmup_steps=10)
        warm.current_step = 5

        assert math.isclose(warm.factor(), 0.5, rel_tol=1e-6)

    def test_factor_cosine_monotonic(self):
        warm = self._make(warmup_mode="cosine", warmup_start_factor=0.0, warmup_steps=10)
        values = []
        for step in range(11):
            warm.current_step = step
            values.append(warm.factor())

        assert all(b >= a - 1e-9 for a, b in zip(values, values[1:]))

    def test_factor_exponential_with_zero_start(self):
        warm = self._make(warmup_mode="exponential", warmup_start_factor=0.0, warmup_steps=10)
        warm.current_step = 5

        assert math.isclose(warm.factor(), 0.5, rel_tol=1e-6)

    def test_factor_polynomial(self):
        warm = self._make(warmup_mode="polynomial", warmup_start_factor=0.0, warmup_steps=10, warmup_poly_power=2.0)
        warm.current_step = 5

        assert math.isclose(warm.factor(), 0.25, rel_tol=1e-6)

    def test_step_increments_and_finishes(self):
        warm = self._make(warmup_mode="linear", warmup_steps=3)

        for _ in range(3):
            warm.step()

        assert warm.is_finished()
        assert warm.current_step == 3

    def test_step_disabled_marks_finished(self):
        warm = self._make(warmup_enabled=False)

        result = warm.step()

        assert result == 1.0
        assert warm.is_finished()

    def test_reset_restores_initial_state(self):
        warm = self._make(warmup_mode="linear", warmup_steps=3)
        for _ in range(3):
            warm.step()
        warm.reset()

        assert warm.current_step == 0
        assert not warm.warmup_finished

    def test_state_dict_round_trip(self):
        warm = self._make(warmup_mode="linear", warmup_steps=5)
        warm.step()
        state = warm.state_dict()

        other = self._make(warmup_mode="linear", warmup_steps=5)
        other.load_state_dict(state)

        assert other.current_step == warm.current_step
        assert other.warmup_finished == warm.warmup_finished

    def test_is_finished_zero_steps(self):
        warm = self._make(warmup_steps=0)

        assert warm.is_finished()


class TestScheduler:
    def _make(self, base_lrs=(1e-3,), warmup=None, **scheduler_overrides) -> Scheduler:
        cfg = SchedulerConfig(**scheduler_overrides)
        return Scheduler(list(base_lrs), warmup, scheduler_config_namespace(cfg), NullLogger(), NullTracker())

    def test_constant_scheduler_keeps_lr(self):
        sched = self._make(type="constant")

        lrs = sched.step(epoch=5)

        assert math.isclose(lrs[0], 1e-3, rel_tol=1e-6)

    def test_step_decay_halves_lr(self):
        sched = self._make(type="step", step_size=10, gamma=0.5)

        lrs = sched.step(epoch=10)

        assert math.isclose(lrs[0], 1e-3 * 0.5, rel_tol=1e-6)

    def test_multi_step_applies_milestones(self):
        sched = self._make(type="multi_step", gamma=0.1, milestones=[2, 4])

        lrs = sched.step(epoch=4)

        assert math.isclose(lrs[0], 1e-3 * 0.1 * 0.1, rel_tol=1e-6)

    def test_exponential_decay(self):
        sched = self._make(type="exponential", gamma=0.9)

        lrs = sched.step(epoch=3)

        assert math.isclose(lrs[0], 1e-3 * 0.9 ** 3, rel_tol=1e-6)

    def test_linear_decay_progress(self):
        sched = self._make(type="linear", start_factor=1.0, end_factor=0.0, total_iters=10)

        lrs = sched.step(epoch=5)

        assert math.isclose(lrs[0], 1e-3 * 0.5, rel_tol=1e-6)

    def test_polynomial_decay(self):
        sched = self._make(type="polynomial", total_iters=10, power=1.0)

        lrs = sched.step(epoch=5)

        assert math.isclose(lrs[0], 1e-3 * 0.5, rel_tol=1e-6)

    def test_cosine_annealing_start_is_base(self):
        sched = self._make(type="cosine_annealing", epochs=10, eta_min=0.0)

        lrs = sched.step(epoch=0)

        assert math.isclose(lrs[0], 1e-3, rel_tol=1e-6)

    def test_cosine_annealing_end_reaches_eta_min(self):
        sched = self._make(type="cosine_annealing", epochs=10, eta_min=1e-6)

        lrs = sched.step(epoch=10)

        assert math.isclose(lrs[0], 1e-6, abs_tol=1e-9)

    def test_cosine_warm_restarts_resets_at_T0(self):
        sched = self._make(type="cosine_annealing_warm_restarts", T_0=5, T_mult=1.0, eta_min=0.0)

        lr_start   = sched.step(epoch=0)[0]
        lr_restart = sched.step(epoch=5)[0]

        assert math.isclose(lr_start, lr_restart, rel_tol=1e-6)

    def test_reduce_on_plateau_drops_after_patience(self):
        sched = self._make(type="reduce_on_plateau", factor=0.5, patience=2, threshold=1e-4)

        sched.step(epoch=0, metric=1.0)
        sched.step(epoch=1, metric=1.0)
        lrs = sched.step(epoch=2, metric=1.0)

        assert lrs[0] < 1e-3

    def test_reduce_on_plateau_no_metric_is_neutral(self):
        sched = self._make(type="reduce_on_plateau")

        lrs = sched.step(epoch=0, metric=None)

        assert math.isclose(lrs[0], 1e-3, rel_tol=1e-6)

    def test_unknown_scheduler_raises(self):
        sched = self._make(type="constant")
        sched.scheduler_type = "does_not_exist"

        with pytest.raises(ValueError, match="Unknown scheduler"):
            sched.step(epoch=1)

    def test_warmup_factor_applied_when_unfinished(self):
        warm = Warmup(warmup_config_namespace(WarmupConfig(warmup_mode="linear", warmup_start_factor=0.5, warmup_steps=10)), NullLogger(), NullTracker())
        sched = self._make(base_lrs=(1.0,), warmup=warm, type="constant")

        lrs = sched.step(epoch=0)

        assert lrs[0] < 1.0

    def test_reset_applies_epoch_offset(self):
        sched = self._make(type="cosine_annealing", epochs=10, eta_min=0.0)
        sched.reset(epoch_offset=5)

        lrs = sched.step(epoch=5)

        assert math.isclose(lrs[0], 1e-3, rel_tol=1e-6)

    def test_state_dict_round_trip(self):
        sched = self._make(type="reduce_on_plateau")
        sched.plateau_best  = 0.5
        sched.plateau_count = 1
        state = sched.state_dict()

        other = self._make(type="reduce_on_plateau")
        other.load_state_dict(state)

        assert other.plateau_best == 0.5
        assert other.plateau_count == 1


class TestEarlyStopping:
    def _make(self, **overrides) -> EarlyStopping:
        cfg = EarlyStoppingConfig(**overrides)
        return EarlyStopping(early_stopping_config_namespace(cfg), NullLogger(), NullTracker())

    def _model(self) -> torch.nn.Module:
        return torch.nn.Linear(2, 2)

    def test_first_call_sets_best(self):
        es    = self._make(patience=3)
        model = self._model()

        stop = es(1.0, model, epoch=0)

        assert not stop
        assert es.best_loss == 1.0

    def test_improvement_resets_counter(self):
        es    = self._make(patience=3, min_delta=0.0)
        model = self._model()

        es(1.0, model, epoch=0)
        es(2.0, model, epoch=1)
        stop = es(0.5, model, epoch=2)

        assert not stop
        assert es.counter == 0
        assert es.best_loss == 0.5

    def test_triggers_after_patience(self):
        es    = self._make(patience=2, min_delta=0.0)
        model = self._model()

        es(1.0, model, epoch=0)
        es(2.0, model, epoch=1)
        stop = es(2.0, model, epoch=2)

        assert stop
        assert es.triggered

    def test_reset_clears_state(self):
        es    = self._make(patience=2)
        model = self._model()
        es(1.0, model, epoch=0)
        es.reset()

        assert es.best_loss is None
        assert es.counter == 0
        assert not es.triggered

    def test_restore_best_loads_params(self):
        es    = self._make(patience=2, restore_best=True)
        model = self._model()
        es(1.0, model, epoch=0)

        with torch.no_grad():
            for p in model.parameters():
                p.add_(5.0)

        es.restore(model)
        restored = {name: p.clone() for name, p in model.state_dict().items()}

        for name, saved in es.best_params.items():
            assert torch.allclose(restored[name], saved)

    def test_state_dict_round_trip(self):
        es    = self._make(patience=2)
        model = self._model()
        es(1.0, model, epoch=0)
        state = es.state_dict()

        other = self._make(patience=2)
        other.load_state_dict(state)

        assert other.best_loss == es.best_loss
        assert other.counter == es.counter


class TestEMA:
    def _make(self, **overrides) -> EMA:
        cfg = EMAConfig(**overrides)
        return EMA(ema_config_namespace(cfg), NullLogger(), NullTracker())

    def test_disabled_init_returns_none_shadow(self):
        ema   = self._make(use_ema=False)
        model = torch.nn.Linear(2, 2)

        shadow = ema.init(model)

        assert shadow is None

    def test_init_creates_shadow_for_all_params(self):
        ema   = self._make(use_ema=True)
        model = torch.nn.Linear(2, 2)

        shadow = ema.init(model)

        assert set(shadow.keys()) == {name for name, _ in model.named_parameters()}

    def test_update_moves_shadow_toward_params(self):
        ema   = self._make(use_ema=True, ema_decay=0.5)
        model = torch.nn.Linear(2, 2)
        ema.init(model)

        before = {name: t.clone() for name, t in ema.shadow.items()}
        with torch.no_grad():
            for p in model.parameters():
                p.add_(2.0)
        ema.update(model)

        changed = any(not torch.allclose(before[name], ema.shadow[name]) for name in before)

        assert changed

    def test_apply_to_and_restore_round_trip(self):
        ema   = self._make(use_ema=True, ema_decay=0.0)
        model = torch.nn.Linear(2, 2)
        ema.init(model)

        original = {name: p.clone() for name, p in model.named_parameters()}
        with torch.no_grad():
            for p in model.parameters():
                p.add_(3.0)
        ema.update(model)

        ema.apply_to(model)
        ema.restore(model)

        for name, p in model.named_parameters():
            assert torch.allclose(p, original[name] + 3.0)

    def test_disabled_update_returns_none(self):
        ema   = self._make(use_ema=False)
        model = torch.nn.Linear(2, 2)
        ema.init(model)

        assert ema.update(model) is None

    def test_state_dict_round_trip(self):
        ema   = self._make(use_ema=True, ema_decay=0.99)
        model = torch.nn.Linear(2, 2)
        ema.init(model)
        state = ema.state_dict()

        other = self._make(use_ema=True)
        other.load_state_dict(state)

        assert other.decay == 0.99
        assert set(other.shadow.keys()) == set(ema.shadow.keys())


class TestGradientClipper:
    def _make(self, **overrides) -> GradientClipper:
        cfg = GradientClipperConfig(**overrides)
        return GradientClipper(clipper_config_namespace(cfg), NullLogger(), NullTracker())

    def _model_with_grads(self, grad_value: float = 1.0) -> torch.nn.Module:
        model = torch.nn.Linear(2, 2, bias=False)
        for p in model.parameters():
            p.grad = torch.full_like(p, grad_value)
        return model

    def test_global_norm_no_grad_returns_zero(self):
        model = torch.nn.Linear(2, 2)

        assert GradientClipper.global_norm(model) == 0.0

    def test_global_norm_matches_manual(self):
        model = self._model_with_grads(2.0)

        grads = torch.cat([p.grad.flatten() for p in model.parameters()])
        expected = float(torch.norm(grads, 2))

        assert math.isclose(GradientClipper.global_norm(model), expected, rel_tol=1e-5)

    def test_disabled_mode_returns_norm_unchanged(self):
        clipper = self._make(clip_mode="disabled")
        model   = self._model_with_grads(1.0)

        norm = clipper.maybe_clip(model, global_step=0)

        assert norm > 0.0

    def test_fixed_mode_clips_to_threshold(self):
        clipper = self._make(clip_mode="fixed", max_grad_norm=0.1)
        model   = self._model_with_grads(5.0)

        norm_after = clipper.maybe_clip(model, global_step=0)

        assert norm_after <= 0.1 + 1e-3

    def test_fixed_mode_does_not_amplify_small_grads(self):
        clipper = self._make(clip_mode="fixed", max_grad_norm=100.0)
        model   = self._model_with_grads(0.01)

        before = GradientClipper.global_norm(model)
        norm_after = clipper.maybe_clip(model, global_step=0)

        assert math.isclose(norm_after, before, rel_tol=1e-5)

    def test_adaptive_percentile_returns_norm_until_window_filled(self):
        clipper = self._make(clip_mode="adaptive_percentile", adaptive_window=5)
        model   = self._model_with_grads(1.0)

        result = clipper.maybe_clip(model, global_step=0)

        assert result > 0.0

    def test_adaptive_threshold_computes_after_window(self):
        clipper = self._make(clip_mode="adaptive_percentile", adaptive_window=3, adaptive_percentile=50.0)
        clipper.history = [1.0, 2.0, 3.0]

        threshold = clipper._compute_adaptive_threshold()

        assert math.isclose(threshold, 2.0, rel_tol=1e-5)

    def test_adaptive_mean_std_threshold(self):
        clipper = self._make(clip_mode="adaptive_mean_std", adaptive_window=4, adaptive_mean_std_k=0.0)
        clipper.history = [1.0, 2.0, 3.0, 4.0]

        threshold = clipper._compute_adaptive_threshold()

        assert math.isclose(threshold, 2.5, rel_tol=1e-5)

    def test_compute_adaptive_threshold_none_when_insufficient(self):
        clipper = self._make(clip_mode="adaptive_percentile", adaptive_window=10)
        clipper.history = [1.0, 2.0]

        assert clipper._compute_adaptive_threshold() is None

    def test_record_appends_to_history(self):
        clipper = self._make(clip_mode="fixed", log_histogram_freq=100)
        clipper.record(1.5, global_step=1)

        assert clipper.history[-1] == 1.5

    def test_check_gradients_detects_nan(self):
        clipper = self._make(clip_mode="fixed")
        model   = self._model_with_grads(1.0)
        first   = next(model.parameters())
        first.grad[0, 0] = float("nan")

        assert clipper.check_gradients(model, global_step=0)

    def test_check_gradients_clean_is_false(self):
        clipper = self._make(clip_mode="fixed")
        model   = self._model_with_grads(1.0)

        assert not clipper.check_gradients(model, global_step=0)


class TestCheckpoint:
    class FakeTrainer:
        def __init__(self):
            self.restored = None

        def capture_state(self, epoch):
            return {"epoch": epoch, "value": 42}

        def restore_state(self, checkpoint):
            self.restored = checkpoint
            return checkpoint["epoch"]

    def test_step_saves_on_improvement(self, tmp_path):
        path    = str(tmp_path / "ck" / "best.pt")
        ck      = Checkpoint(NullLogger(), NullTracker(), path)
        trainer = self.FakeTrainer()

        improved = ck.step(val_loss=1.0, epoch=0, trainer=trainer)

        assert improved
        assert ck.best_val_loss == 1.0
        assert (tmp_path / "ck" / "best.pt").exists()

    def test_step_no_improvement_returns_false(self, tmp_path):
        path    = str(tmp_path / "best.pt")
        ck      = Checkpoint(NullLogger(), NullTracker(), path)
        trainer = self.FakeTrainer()

        ck.step(val_loss=1.0, epoch=0, trainer=trainer)
        improved = ck.step(val_loss=2.0, epoch=1, trainer=trainer)

        assert not improved
        assert ck.best_epoch == 0

    def test_save_and_load_round_trip(self, tmp_path):
        path    = str(tmp_path / "best.pt")
        ck      = Checkpoint(NullLogger(), NullTracker(), path)
        trainer = self.FakeTrainer()

        ck.step(val_loss=0.5, epoch=3, trainer=trainer)

        loaded_ck = Checkpoint(NullLogger(), NullTracker(), path)
        epoch     = loaded_ck.load(trainer, path)

        assert epoch == 3
        assert loaded_ck.best_val_loss == 0.5
        assert loaded_ck.best_epoch == 3


class TestOverfitManager:
    def _make(self, **overrides) -> OverfitManager:
        cfg = OverfitConfig(**overrides)
        return OverfitManager(overfit_config_namespace(cfg), NullLogger())

    def test_disabled_passthrough_loaders(self):
        manager = self._make(enabled=False)
        train = [1, 2, 3]
        val   = [4]
        test  = [5]

        out = manager.setup_loaders(train, val, test)

        assert out == (train, val, test)

    def test_disabled_check_stop_false(self):
        manager = self._make(enabled=False)

        assert not manager.check_stop(0.0)

    def test_enabled_setup_returns_repeated_batch(self):
        manager = self._make(enabled=True, max_steps=4, batch_size=2)
        batch   = (torch.randn(5, 3), torch.randn(5, 3))
        train   = [batch, batch, batch]

        data_loader, val_loader, test_loader = manager.setup_loaders(train, list(train), list(train))

        assert len(data_loader) == min(len(train), 4)
        assert len(val_loader) == 1
        assert data_loader[0][0].shape[0] == 2

    def test_check_stop_on_max_steps(self):
        manager = self._make(enabled=True, max_steps=2, batch_size=1)
        batch   = (torch.randn(3, 2), torch.randn(3, 2))
        manager.setup_loaders([batch, batch], [batch], [batch])

        assert manager.check_stop(train_loss=1.0)

    def test_check_stop_on_threshold(self):
        manager = self._make(enabled=True, max_steps=1000, stop_threshold=1e-3, batch_size=1)
        batch   = (torch.randn(3, 2), torch.randn(3, 2))
        manager.setup_loaders([batch], [batch], [batch])

        assert manager.check_stop(train_loss=1e-6)


class TestCurriculumController:
    def _build(self, curriculum_cfg: LossCurriculumConfig):
        criterion      = SimpleNamespace(set_curriculum=lambda cfg: setattr(criterion, "swapped", cfg))
        criterion.swapped = None

        early_stopping = SimpleNamespace(reset=lambda: setattr(early_stopping, "was_reset", True))
        early_stopping.was_reset = False

        lr_scheduler   = SimpleNamespace(
            reset     = lambda epoch_offset=0: setattr(lr_scheduler, "reset_offset", epoch_offset),
            base_lrs  = [1e-3],
        )
        lr_scheduler.reset_offset = None

        warmup = SimpleNamespace(
            reset        = lambda: setattr(warmup, "was_reset", True),
            warmup_steps = 10,
            enabled      = True,
            is_finished  = lambda: False,
            factor       = lambda: 0.5,
        )
        warmup.was_reset = False

        optimizer = SimpleNamespace(param_groups=[{"params": []}], state={})

        applied = {}
        controller = CurriculumController(
            curriculum       = curriculum_cfg,
            criterion        = criterion,
            early_stopping   = early_stopping,
            lr_scheduler     = lr_scheduler,
            warmup           = warmup,
            optimizer        = optimizer,
            update_optimizer = lambda lrs: applied.update({"lrs": lrs}),
            logger           = NullLogger(),
        )
        return controller, criterion, early_stopping, lr_scheduler, warmup, applied

    def test_no_swap_when_disabled(self):
        cfg = LossCurriculumConfig(enabled=False, swap_epoch=0)
        controller, criterion, *_ = self._build(cfg)

        swapped = controller.maybe_swap(epoch=0)

        assert not swapped
        assert criterion.swapped is None

    def test_no_swap_on_wrong_epoch(self):
        cfg = LossCurriculumConfig(enabled=True, swap_epoch=5)
        controller, criterion, *_ = self._build(cfg)

        swapped = controller.maybe_swap(epoch=2)

        assert not swapped

    def test_swap_replaces_criterion_config(self):
        complete = LossConfig(use_l1_curve=True)
        cfg      = LossCurriculumConfig(enabled=True, swap_epoch=3, complete=complete)
        controller, criterion, *_ = self._build(cfg)

        swapped = controller.maybe_swap(epoch=3)

        assert swapped
        assert criterion.swapped is complete

    def test_swap_resets_early_stopping(self):
        cfg = LossCurriculumConfig(enabled=True, swap_epoch=0, reset_early_stopping=True)
        controller, _, early_stopping, *_ = self._build(cfg)

        controller.maybe_swap(epoch=0)

        assert early_stopping.was_reset

    def test_swap_resets_lr_with_offset(self):
        cfg = LossCurriculumConfig(enabled=True, swap_epoch=4, reset_lr=True)
        controller, _, _, lr_scheduler, _, applied = self._build(cfg)

        controller.maybe_swap(epoch=4)

        assert lr_scheduler.reset_offset == 4
        assert "lrs" in applied

    def test_swap_resets_warmup(self):
        cfg = LossCurriculumConfig(enabled=True, swap_epoch=0, reset_warmup=True)
        controller, _, _, _, warmup, _ = self._build(cfg)

        controller.maybe_swap(epoch=0)

        assert warmup.was_reset


class TestLayerRecord:
    def test_depth_and_param_counts(self):
        module = torch.nn.Linear(3, 4)
        record = LayerRecord("encoder.linear", module)

        assert record.depth == 2
        assert record.own_params == module.weight.numel() + module.bias.numel()
        assert record.trainable == record.own_params
        assert record.frozen == 0

    def test_frozen_params_counted(self):
        module = torch.nn.Linear(3, 4)
        for p in module.parameters():
            p.requires_grad_(False)
        record = LayerRecord("linear", module)

        assert record.frozen == module.weight.numel() + module.bias.numel()
        assert record.trainable == 0


class TestModelInspector:
    def _model(self) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Conv2d(2, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 2, 1),
        )

    def test_totals_sums_parameters(self, tmp_path):
        model     = self._model()
        inspector = ModelInspector(model, NullLogger(), tmp_path)

        totals = inspector.totals()

        expected = sum(p.numel() for p in model.parameters())

        assert totals["total"] == expected
        assert totals["trainable"] == expected
        assert totals["frozen"] == 0
        assert totals["size_mb"] > 0.0

    def test_run_records_shapes(self, tmp_path):
        model     = self._model()
        inspector = ModelInspector(model, NullLogger(), tmp_path)
        sample    = torch.randn(1, 2, 8, 8)

        inspector.run(sample)

        assert len(inspector.records) > 0
        assert any(r.visited for r in inspector.records)
        assert len(inspector.hooks) == 0

    def test_run_restores_training_mode(self, tmp_path):
        model     = self._model()
        model.train()
        inspector = ModelInspector(model, NullLogger(), tmp_path)

        inspector.run(torch.randn(1, 2, 8, 8))

        assert model.training

    def test_to_markdown_contains_model_name(self, tmp_path):
        model     = self._model()
        inspector = ModelInspector(model, NullLogger(), tmp_path)
        inspector.run(torch.randn(1, 2, 8, 8))

        doc = inspector.to_markdown(title="Doc")

        assert isinstance(doc.render(), str)
        assert "Doc" in doc.render()

    def test_save_markdown_writes_file(self, tmp_path):
        model     = self._model()
        inspector = ModelInspector(model, NullLogger(), tmp_path)
        inspector.run(torch.randn(1, 2, 8, 8))

        path = inspector.save_markdown("model.md")

        assert path.exists()

    def test_shape_of_handles_non_tensor(self, tmp_path):
        inspector = ModelInspector(self._model(), NullLogger(), tmp_path)

        assert inspector._shape_of(5) == "int"
        assert inspector._shape_of([]) == "list[0]"


class TestTrainingDocs:
    def _model(self) -> torch.nn.Module:
        return torch.nn.Sequential(torch.nn.Conv2d(2, 4, 1))

    def test_disabled_emits_nothing(self, tmp_path):
        docs = TrainingDocs(self._model(), SimpleNamespace(), NullLogger(), tmp_path, enabled=False)

        docs.emit(data_loader=[(torch.randn(1, 2, 4, 4),)], device=torch.device("cpu"))

        assert not (tmp_path / "docs").exists()

    def test_enabled_writes_documentation(self, tmp_path):
        docs   = TrainingDocs(self._model(), SimpleNamespace(), NullLogger(), tmp_path, enabled=True)
        loader = [(torch.randn(1, 2, 4, 4),)]

        docs.emit(data_loader=loader, device=torch.device("cpu"))

        assert (tmp_path / "docs").exists()


class TestLossScaleProbe:
    def test_config_defaults(self):
        cfg = LossScaleProbeConfig()

        assert cfg.enabled
        assert cfg.exit_after
        assert isinstance(cfg.enabled_losses, dict)

    def test_constructor_forces_weights_to_one(self):
        probe_cfg = LossScaleProbeConfig(enabled_losses={})
        loss_cfg  = LossConfig()
        gcfg      = GaussianConfig(n_default_gaussians=2, x_min=-10.0, x_max=10.0)

        probe = LossScaleProbe(probe_cfg, loss_cfg, gcfg, norm_stats=IdentityNorm(), logger=NullLogger())

        assert probe.loss_cfg.use_mse_curve
        assert probe.loss_cfg.weight_mse_curve == 1.0

    def test_enabled_losses_override_respected(self):
        probe_cfg = LossScaleProbeConfig(enabled_losses={"use_mse_curve": False})
        loss_cfg  = LossConfig()
        gcfg      = GaussianConfig(n_default_gaussians=2, x_min=-10.0, x_max=10.0)

        probe = LossScaleProbe(probe_cfg, loss_cfg, gcfg, norm_stats=IdentityNorm(), logger=NullLogger())

        assert not probe.loss_cfg.use_mse_curve

    def test_iqr_filter_small_input_passthrough(self):
        values = [1.0, 2.0, 3.0]

        assert LossScaleProbe._iqr_filter(values) == values

    def test_iqr_filter_removes_outliers(self):
        values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1000.0]

        filtered = LossScaleProbe._iqr_filter(values, k=1.5)

        assert 1000.0 not in filtered

    def test_run_disabled_returns_empty(self):
        probe_cfg = LossScaleProbeConfig(enabled=False)
        loss_cfg  = LossConfig()
        gcfg      = GaussianConfig(n_default_gaussians=2, x_min=-10.0, x_max=10.0)
        probe     = LossScaleProbe(probe_cfg, loss_cfg, gcfg, norm_stats=IdentityNorm(), logger=NullLogger())

        result = probe.run(train_loader=[], model=torch.nn.Identity(), device=torch.device("cpu"), x_axis=torch.linspace(-10, 10, 21))

        assert result == {}

    def test_run_computes_suggested_weights(self):
        seed_all()
        probe_cfg = LossScaleProbeConfig(enabled=True, n_batches=2, exit_after=False, enabled_losses={"use_mse_curve": True})
        loss_cfg  = LossConfig()
        gcfg      = GaussianConfig(n_default_gaussians=2, x_min=-10.0, x_max=10.0, amp_max=10.0)
        probe     = LossScaleProbe(probe_cfg, loss_cfg, gcfg, norm_stats=IdentityNorm(), logger=NullLogger())

        for flag in probe._ALL_USE_FLAGS:
            setattr(probe.loss_cfg, flag, flag == "use_mse_curve")

        model    = torch.nn.Conv2d(2, 6, 1)
        batch    = (torch.randn(2, 2, 3, 3), torch.randn(2, 6, 3, 3))
        loader   = [batch, batch]

        result = probe.run(loader, model, torch.device("cpu"), torch.linspace(-10, 10, 21))

        assert "mse_curve" in result
        assert math.isfinite(result["mse_curve"])


class TestMetricAggregator:
    def _loss_dict(self, value: float) -> dict:
        return {
            "components" : {"mse": value, "param_l1/amp": value},
            "weighted"   : {"mse": 2.0 * value},
            "monitor"    : {"mse_denorm": value},
        }

    def test_add_accumulates_and_reduces(self):
        agg = MetricAggregator()
        agg.add(self._loss_dict(1.0))
        agg.add(self._loss_dict(3.0))

        components = agg.reduce_components()
        weighted   = agg.reduce_weighted()
        monitor    = agg.reduce_monitor()

        assert math.isclose(components["mse"], 2.0)
        assert math.isclose(weighted["mse"], 4.0)
        assert math.isclose(monitor["mse_denorm"], 2.0)

    def test_reduce_empty_does_not_divide_by_zero(self):
        agg = MetricAggregator()

        assert agg.reduce_components() == {}

    def test_add_extra_skips_nan(self):
        agg = MetricAggregator()
        agg.add_extra({"a": 1.0, "b": float("nan")})

        reduced = agg.reduce_extra()

        assert math.isclose(reduced["a"], 1.0)
        assert math.isclose(reduced["b"], 0.0)

    def test_missing_monitor_key_handled(self):
        agg = MetricAggregator()
        agg.add({"components": {"x": 1.0}, "weighted": {"x": 1.0}})

        assert agg.count == 1
        assert agg.reduce_monitor() == {}


class TestTrainStep:
    def _build(self, accumulation_steps: int = 1, ema_enabled: bool = False):
        seed_all()
        logger  = NullLogger()
        tracker = NullTracker()

        clipper = GradientClipper(clipper_config_namespace(GradientClipperConfig(clip_mode="disabled")), logger, tracker)
        ema     = EMA(ema_config_namespace(EMAConfig(use_ema=ema_enabled, ema_decay=0.9)), logger, tracker)

        model = torch.nn.Conv2d(2, 6, 1)
        ema.init(model)

        device = torch.device("cpu")
        opt    = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        class SquaredCriterion:
            def __call__(self, pred, gt):
                value = ((pred - gt) ** 2).mean()
                return {"total_loss": value, "components": {"mse": value.detach()}, "weighted": {}, "monitor": {}}

        step = TrainStep(
            model              = model,
            optimizer          = opt,
            scaler             = scaler,
            criterion          = SquaredCriterion(),
            grad_clipper       = clipper,
            ema                = ema,
            device             = device,
            logger             = logger,
            tracker            = tracker,
            accumulation_steps = accumulation_steps,
            use_amp            = False,
            ema_every          = 1,
        )
        return step, model

    def test_step_returns_loss_and_dict(self):
        step, _ = self._build()
        images  = torch.randn(2, 2, 3, 3)
        gt      = torch.randn(2, 6, 3, 3)

        loss, loss_dict = step.step(images, gt, batch_idx=0, n_batches=1, global_step=0)

        assert torch.isfinite(loss)
        assert "total_loss" in loss_dict

    def test_step_updates_parameters(self):
        step, model = self._build()
        before      = model.weight.detach().clone()
        images      = torch.randn(2, 2, 3, 3)
        gt          = torch.randn(2, 6, 3, 3)

        step.step(images, gt, batch_idx=0, n_batches=1, global_step=0)

        assert not torch.allclose(before, model.weight.detach())

    def test_accumulation_defers_optimizer_step(self):
        step, model = self._build(accumulation_steps=2)
        before      = model.weight.detach().clone()
        images      = torch.randn(2, 2, 3, 3)
        gt          = torch.randn(2, 6, 3, 3)

        step.step(images, gt, batch_idx=0, n_batches=4, global_step=0)

        assert torch.allclose(before, model.weight.detach())

        step.step(images, gt, batch_idx=1, n_batches=4, global_step=1)

        assert not torch.allclose(before, model.weight.detach())

    def test_ema_updated_when_enabled(self):
        step, model = self._build(ema_enabled=True)
        shadow_before = {name: t.clone() for name, t in step.ema.shadow.items()}
        images        = torch.randn(2, 2, 3, 3)
        gt            = torch.randn(2, 6, 3, 3)

        step.step(images, gt, batch_idx=0, n_batches=1, global_step=0)

        changed = any(not torch.allclose(shadow_before[name], step.ema.shadow[name]) for name in shadow_before)

        assert changed

    def test_debug_tracker_path_runs(self):
        step, _      = self._build()
        step.tracker = DebugTracker()
        images       = torch.randn(2, 2, 3, 3)
        gt           = torch.randn(2, 6, 3, 3)

        loss, _ = step.step(images, gt, batch_idx=0, n_batches=1, global_step=0)

        assert torch.isfinite(loss)
