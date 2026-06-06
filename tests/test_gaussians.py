from __future__ import annotations

import numpy as np
import torch

from tools.gaussians import GaussianClamp, GaussianMixture, GaussianReconstructor


class TestGaussianMixture:
    def test_evaluate_batch_matches_direct_formula(self):
        height_axis = np.linspace(-30.0, 30.0, 121, dtype=np.float64)
        amps        = np.array([[1.0, 0.5]], dtype=np.float64)
        mus         = np.array([[0.0, 10.0]], dtype=np.float64)
        sigs        = np.array([[2.0, 4.0]], dtype=np.float64)

        predicted = GaussianMixture.evaluate_batch(height_axis, amps, mus, sigs)

        expected  = 1.0 * np.exp(-(height_axis - 0.0) ** 2 / (2.0 * 2.0 ** 2))
        expected += 0.5 * np.exp(-(height_axis - 10.0) ** 2 / (2.0 * 4.0 ** 2))

        assert predicted.shape == (1, 121)
        assert np.allclose(predicted[0], expected, atol=1e-6)

    def test_evaluate_batch_peak_at_mu_equals_amplitude(self):
        height_axis = np.linspace(-30.0, 30.0, 61, dtype=np.float64)
        amps        = np.array([[2.5]], dtype=np.float64)
        mus         = np.array([[0.0]], dtype=np.float64)
        sigs        = np.array([[3.0]], dtype=np.float64)

        predicted = GaussianMixture.evaluate_batch(height_axis, amps, mus, sigs)
        peak_idx  = int(np.argmax(predicted[0]))

        assert np.isclose(height_axis[peak_idx], 0.0)
        assert np.isclose(predicted[0, peak_idx], 2.5, atol=1e-6)

    def test_evaluate_batch_zero_sigma_is_finite(self):
        height_axis = np.linspace(-10.0, 10.0, 21, dtype=np.float64)
        amps        = np.array([[1.0]], dtype=np.float64)
        mus         = np.array([[0.0]], dtype=np.float64)
        sigs        = np.array([[0.0]], dtype=np.float64)

        predicted = GaussianMixture.evaluate_batch(height_axis, amps, mus, sigs)

        assert np.all(np.isfinite(predicted))

    def test_evaluate_pixel_total_is_sum_of_components(self):
        height_axis = np.linspace(-20.0, 20.0, 81, dtype=np.float64)
        params      = np.array([1.0, -5.0, 2.0, 0.7, 5.0, 3.0], dtype=np.float64)

        total, components = GaussianMixture.evaluate_pixel(params, height_axis, n_gaussians=2)

        assert len(components) == 2
        assert np.allclose(total, components[0] + components[1])

    def test_evaluate_slice_matches_evaluate_pixel(self):
        h_val  = 4.0
        params = np.array(
            [[[1.0]], [[2.0]], [[1.5]], [[0.5]], [[-3.0]], [[2.5]]],
            dtype = np.float64,
        )

        slice_value     = GaussianMixture.evaluate_slice(params, h_val, n_gaussians=2)
        pixel_params    = params[:, 0, 0]
        total, _        = GaussianMixture.evaluate_pixel(pixel_params, np.array([h_val]), n_gaussians=2)

        assert np.isclose(slice_value[0, 0], total[0], atol=1e-5)


class TestGaussianReconstructor:
    def test_reconstruct_batch_matches_component_sum(self):
        x_axis = np.linspace(-15.0, 15.0, 61, dtype=np.float64)
        gauss  = np.array([[[1.0, -2.0, 2.0], [0.6, 6.0, 3.0]]], dtype=np.float64)

        batch      = GaussianReconstructor.reconstruct_batch(gauss, x_axis)
        components = GaussianReconstructor.components(gauss[0].reshape(-1), x_axis, n_gaussians=2)

        assert batch.shape == (1, 61)
        assert batch.dtype == np.float32
        assert np.allclose(batch[0], components[0] + components[1], atol=1e-5)

    def test_reconstruct_batch_clamps_negative_amplitude(self):
        x_axis = np.linspace(-5.0, 5.0, 11, dtype=np.float64)
        gauss  = np.array([[[-1.0, 0.0, 2.0]]], dtype=np.float64)

        batch = GaussianReconstructor.reconstruct_batch(gauss, x_axis)

        assert np.allclose(batch, 0.0)

    def test_reconstruct_batch_zero_sigma_is_finite(self):
        x_axis = np.linspace(-5.0, 5.0, 11, dtype=np.float64)
        gauss  = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float64)

        batch = GaussianReconstructor.reconstruct_batch(gauss, x_axis)

        assert np.all(np.isfinite(batch))


class TestGaussianClamp:
    def test_apply_respects_bounds(self):
        x_axis  = torch.linspace(-30.0, 30.0, 121)
        params  = torch.tensor([[5.0, -100.0, 1000.0, -3.0, 100.0, 0.0]])
        amp_max = 2.0

        clamped = GaussianClamp.apply(params, x_axis, amp_max=amp_max)

        x_min   = float(x_axis.min())
        x_max   = float(x_axis.max())
        x_step  = (x_max - x_min) / (x_axis.shape[0] - 1)
        x_range = x_max - x_min

        assert clamped.shape == params.shape
        assert float(clamped[0, 0]) <= amp_max
        assert float(clamped[0, 3]) >= 0.0
        assert x_min <= float(clamped[0, 1]) <= x_max
        assert x_min <= float(clamped[0, 4]) <= x_max
        assert x_step * 0.5 <= float(clamped[0, 2]) <= x_range * 0.5
        assert x_step * 0.5 <= float(clamped[0, 5]) <= x_range * 0.5

    def test_apply_preserves_in_range_values(self):
        x_axis = torch.linspace(-30.0, 30.0, 121)
        params = torch.tensor([[1.0, 0.0, 5.0]])

        clamped = GaussianClamp.apply(params, x_axis, amp_max=2.0)

        assert torch.allclose(clamped, params)

    def test_apply_leaky_slope_keeps_gradient_path(self):
        x_axis = torch.linspace(-30.0, 30.0, 121)
        params = torch.tensor([[5.0, 0.0, 5.0]], requires_grad=True)

        clamped = GaussianClamp.apply(params, x_axis, amp_max=2.0, leaky_slope=0.1)
        clamped.sum().backward()

        assert params.grad is not None
        assert torch.all(torch.isfinite(params.grad))
