from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d

import jax
import jax.numpy as jnp

from configuration.param_extraction_config import FitMode, FitSettings
from tools.logger import Logger


def gpu_is_available() -> bool:
    return any(d.platform in ("gpu", "cuda") for d in jax.devices())


class JaxGaussianModel:
    @staticmethod
    def evaluate(height_axis: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        amplitudes = params[0::3]
        means      = params[1::3]
        sigmas     = params[2::3]
        safe_s2    = 2.0 * jnp.maximum(sigmas, 1e-6) ** 2
        diff       = height_axis[None, :] - means[:, None]
        exponent   = jnp.clip(-(diff * diff) / safe_s2[:, None], -100.0, 0.0)
        return (amplitudes[:, None] * jnp.exp(exponent)).sum(axis=0)

    @staticmethod
    def mse_loss(params: jnp.ndarray, height_axis: jnp.ndarray, profile: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
        predicted = JaxGaussianModel.evaluate(height_axis, params)
        mse       = jnp.mean((predicted - profile) ** 2)
        penalty   = jnp.sum(jnp.maximum(lower - params, 0.0) ** 2 + jnp.maximum(params - upper, 0.0) ** 2)
        return mse + 1e4 * penalty

    @staticmethod
    def nll_loss(params: jnp.ndarray, height_axis: jnp.ndarray, profile: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
        predicted = JaxGaussianModel.evaluate(height_axis, params)
        nll       = jnp.sum(predicted - profile * jnp.log(jnp.maximum(predicted, 1e-10)))
        penalty   = jnp.sum(jnp.maximum(lower - params, 0.0) ** 2 + jnp.maximum(params - upper, 0.0) ** 2)
        return nll + 1e4 * penalty


class AdamKernel:
    def __init__(self, loss_fn) -> None:
        batched_value_and_grad = jax.vmap(jax.value_and_grad(loss_fn), in_axes=(0, None, 0, None, None))
        self._run              = self._build(batched_value_and_grad)

    @staticmethod
    def _build(batched_value_and_grad):
        @partial(jax.jit, static_argnames=("n_steps",))
        def _run(init_params: jnp.ndarray, height_axis: jnp.ndarray, profiles: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray, n_steps: int = 1500, lr: float = 1e-2, b1: float = 0.9, b2: float = 0.999) -> jnp.ndarray:
            lr_  = jnp.float32(lr)
            b1_  = jnp.float32(b1)
            b2_  = jnp.float32(b2)
            eps  = jnp.float32(1e-8)

            params = jnp.clip(init_params.astype(jnp.float32), lower, upper)
            m      = jnp.zeros_like(params)
            v      = jnp.zeros_like(params)

            def _step(carry, t):
                p, m_, v_ = carry
                _, g      = batched_value_and_grad(p, height_axis, profiles, lower, upper)
                m_        = b1_ * m_ + (1.0 - b1_) * g
                v_        = b2_ * v_ + (1.0 - b2_) * g * g
                tf        = t.astype(jnp.float32) + 1.0
                p         = p - lr_ * (m_ / (1.0 - b1_ ** tf)) / (jnp.sqrt(v_ / (1.0 - b2_ ** tf)) + eps)
                p         = jnp.clip(p, lower, upper)
                return (p, m_, v_), None

            (params, _, _), _ = jax.lax.scan(_step, (params, m, v), jnp.arange(n_steps))
            return params

        return _run

    def __call__(self, init_params, height_axis, profiles, lower, upper, n_steps=1500, lr=1e-2, b1=0.9, b2=0.999):
        return self._run(init_params, height_axis, profiles, lower, upper, n_steps=n_steps, lr=lr, b1=b1, b2=b2)


class PmapAdamKernel:
    def __init__(self, loss_fn, devices: list) -> None:
        self._devices   = devices
        self._n_devices = len(devices)
        batched_vg      = jax.vmap(jax.value_and_grad(loss_fn), in_axes=(0, None, 0, None, None))
        self._run       = self._build(batched_vg, devices)

    @staticmethod
    def _build(batched_value_and_grad, devices):
        def _run_on_device(init_params: jnp.ndarray, height_axis: jnp.ndarray, profiles: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray, n_steps: int = 1500, lr: float = 1e-2, b1: float = 0.9, b2: float = 0.999) -> jnp.ndarray:
            lr_  = jnp.float32(lr)
            b1_  = jnp.float32(b1)
            b2_  = jnp.float32(b2)
            eps  = jnp.float32(1e-8)

            params = jnp.clip(init_params.astype(jnp.float32), lower, upper)
            m      = jnp.zeros_like(params)
            v      = jnp.zeros_like(params)

            def _step(carry, t):
                p, m_, v_ = carry
                _, g      = batched_value_and_grad(p, height_axis, profiles, lower, upper)
                m_        = b1_ * m_ + (1.0 - b1_) * g
                v_        = b2_ * v_ + (1.0 - b2_) * g * g
                tf        = t.astype(jnp.float32) + 1.0
                p         = p - lr_ * (m_ / (1.0 - b1_ ** tf)) / (jnp.sqrt(v_ / (1.0 - b2_ ** tf)) + eps)
                p         = jnp.clip(p, lower, upper)
                return (p, m_, v_), None

            (params, _, _), _ = jax.lax.scan(_step, (params, m, v), jnp.arange(n_steps))
            return params

        return jax.pmap(
            _run_on_device,
            in_axes                    = (0, None, 0, None, None),
            static_broadcasted_argnums = (5, 6, 7, 8),
            devices                    = devices,
        )

    def __call__(self, init_params, height_axis, profiles, lower, upper, n_steps=1500, lr=1e-2, b1=0.9, b2=0.999):
        n     = init_params.shape[0]
        pad   = (-n) % self._n_devices
        n_par = init_params.shape[1]
        H     = profiles.shape[1]

        if pad > 0:
            init_p = jnp.concatenate([init_params, jnp.zeros((pad, n_par), dtype=jnp.float32)], axis=0)
            prof_p = jnp.concatenate([profiles,    jnp.zeros((pad, H),     dtype=jnp.float32)], axis=0)
        else:
            init_p = init_params
            prof_p = profiles

        n_pad   = n + pad
        shard   = n_pad // self._n_devices
        init_s  = init_p.reshape(self._n_devices, shard, n_par)
        prof_s  = prof_p.reshape(self._n_devices, shard, H)

        out_s = self._run(init_s, height_axis, prof_s, lower, upper, n_steps, lr, b1, b2)
        
        return out_s.reshape(n_pad, n_par)[:n]


class GPUParameterExtractor:
    def __init__(self, fit_settings: FitSettings, logger: Logger, range_batch_size: int = 256, adam_steps: int = 1500, adam_lr: float = 1e-2, adam_b1: float = 0.9, adam_b2: float = 0.999, gpu_device_ids: Optional[List[int]] = None, r2_sample_cap: int = 4096) -> None:
        self.fit_settings     = fit_settings
        self.logger           = logger
        self.range_batch_size = range_batch_size
        self.adam_steps       = adam_steps
        self.adam_lr          = adam_lr
        self.adam_b1          = adam_b1
        self.adam_b2          = adam_b2
        self.r2_sample_cap    = r2_sample_cap

        is_mle  = isinstance(fit_settings.fit_config, FitMode.MLE)
        loss_fn = JaxGaussianModel.nll_loss if is_mle else JaxGaussianModel.mse_loss

        all_gpu_devices  = [d for d in jax.devices() if d.platform in ("gpu", "cuda")]
        active_devices   = ([all_gpu_devices[i] for i in gpu_device_ids] if gpu_device_ids else all_gpu_devices) if all_gpu_devices else jax.devices()

        if len(active_devices) > 1:
            self._kernel    = PmapAdamKernel(loss_fn, active_devices)
            self._n_devices = len(active_devices)
        else:
            self._kernel    = AdamKernel(loss_fn)
            self._n_devices = 1

        self.logger.subsection(f"JAX active devices : {active_devices}")
        self.logger.subsection(f"range_batch_size={range_batch_size}  adam_steps={adam_steps}  n_devices={self._n_devices}")

    def _build_finite_upper_bounds(self, lower_b: list, upper_b: list, n_params: int, height_span: float) -> list:
        amp_indices = set(range(0, n_params, 3))
        
        return [height_span * 2 if (v == np.inf and i not in amp_indices) else float(v) for i, v in enumerate(upper_b)]

    def _vectorised_initial_guess(self, abs_batch_rAzH: np.ndarray, height_axis: np.ndarray, n_gaussians: int) -> np.ndarray:
        r_count, Az, H = abs_batch_rAzH.shape
        N              = r_count * Az
        sigma_guess    = float((height_axis[-1] - height_axis[0]) / (4.0 * n_gaussians))
        working        = uniform_filter1d(abs_batch_rAzH.reshape(N, H).astype(np.float32), size=5, mode="nearest", axis=1).copy()
        params = np.zeros((N, 3 * n_gaussians), dtype=np.float32)
        
        for g in range(n_gaussians):
            peak_idx           = np.argmax(working, axis=1)
            peak_amp           = working[np.arange(N), peak_idx]
            peak_mu            = height_axis[peak_idx].astype(np.float32)
            params[:, g*3 + 0] = np.maximum(peak_amp, 1e-10)
            params[:, g*3 + 1] = peak_mu
            params[:, g*3 + 2] = sigma_guess
            dist               = np.abs(height_axis[None, :] - peak_mu[:, None])
            working[dist < 2.0 * sigma_guess] = 0.0
        
        return params

    def run(self, tomogram_path: Path, height_range: Tuple[float, float]) -> np.ndarray:
        from pipelines.param_extraction_pipeline.fitting import FittingMethods

        fit_cfg     = self.fit_settings.fit_config
        n_gaussians = self.fit_settings.number_of_gaussians
        n_params    = 3 * n_gaussians

        tomogram_mmap = np.load(str(tomogram_path), mmap_mode="r", allow_pickle=False)
        H, Az, R      = tomogram_mmap.shape
        height_axis   = np.linspace(height_range[0], height_range[1], H, dtype=np.float32)
        height_ax_j   = jnp.array(height_axis)

        self.logger.subsection(f"Tomogram : H={H}  Az={Az}  R={R}  n_gaussians={n_gaussians}")
        self.logger.subsection(f"Total pixels : {R * Az:,}  batches : {-(-R // self.range_batch_size)}")

        lower_b, upper_b  = FittingMethods._build_bounds(
            number_of_gaussians = n_gaussians,
            height_low          = float(height_axis[0]),
            height_high         = float(height_axis[-1]),
            lower_bounds_config = fit_cfg.lower_bounds,
            upper_bounds_config = fit_cfg.upper_bounds,
        )
        
        upper_b_base     = self._build_finite_upper_bounds(lower_b, upper_b, n_params, float(height_axis[-1] - height_axis[0]))
        lower_j          = jnp.array(lower_b, dtype=jnp.float32)
        amp_indices      = set(range(0, n_params, 3))
        threshold_factor = float(getattr(fit_cfg, "threshold_factor", 0.0))
        truncation_index = int(  getattr(fit_cfg, "truncation_index", H))
        output           = np.zeros((n_params, Az, R), dtype=np.float32)

        self.logger.subsection("Compiling JAX kernel (one-time)...")
        _up_w   = jnp.array([2.0 if i in amp_indices else upper_b_base[i] for i in range(n_params)], dtype=jnp.float32)
        _warm_n = max(4, self._n_devices)
        self._kernel(jnp.ones((_warm_n, n_params), dtype=jnp.float32), height_ax_j, jnp.ones((_warm_n, H), dtype=jnp.float32), lower_j, _up_w, n_steps=2)
        self.logger.subsection("Kernel compiled.")

        upper_j = jnp.array([2.0 if i in amp_indices else upper_b_base[i] for i in range(n_params)], dtype=jnp.float32,)

        def _load_and_prepare(r_start: int) -> tuple:
            r_e     = min(r_start + self.range_batch_size, R)
            r_c     = r_e - r_start
            raw     = np.abs(np.array(tomogram_mmap[:, :, r_start:r_e])).astype(np.float32)
           
            if threshold_factor > 0.0:
                raw = np.where(raw > raw.max(axis=0, keepdims=True) * threshold_factor, raw, 0.0)
           
            if truncation_index < H:
                raw[truncation_index:, :, :] = 0.0
           
            pf      = raw.transpose(2, 1, 0).reshape(r_c * Az, H).copy()
            act     = pf.max(axis=1) > 1e-7
            pmax    = pf.max(axis=1, keepdims=True)
            scale   = np.where(act[:, None], pmax, 1.0).astype(np.float32)
            norm    = pf / scale
           
            if fit_cfg.initial_guess is not None:
                ig   = np.array(fit_cfg.initial_guess, dtype=np.float32)
                init = np.zeros((r_c * Az, n_params), dtype=np.float32)
                init[act] = ig
                for ai in amp_indices:
                    init[act, ai] /= scale[act, 0]
            else:
                init = self._vectorised_initial_guess(raw.transpose(2, 1, 0), height_axis, n_gaussians)
                for ai in amp_indices:
                    init[act, ai] /= scale[act, 0]
            del raw
            return pf, act, scale, norm, init, r_e

        total_attempted = 0

        progress_bar = self.logger.track(transient=True)
        progress     = progress_bar.__enter__()
        bar_task     = progress.add_task("  [section]GPU fitting range bins[/section]", total=R)

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                r               = 0
                prefetch_future = pool.submit(_load_and_prepare, r)

                while r < R:
                    profiles_flat, active, safe_scale, profiles_norm, init_all, r_end = prefetch_future.result()
                    r_count          = r_end - r
                    total_attempted += int(active.sum())

                    next_r = r_end
                    if next_r < R:
                        prefetch_future = pool.submit(_load_and_prepare, next_r)

                    init_j   = jnp.array(init_all,     dtype=jnp.float32)
                    prof_j   = jnp.array(profiles_norm, dtype=jnp.float32)
                    fitted_j = self._kernel(init_j, height_ax_j, prof_j, lower_j, upper_j, n_steps=self.adam_steps, lr=self.adam_lr, b1=self.adam_b1, b2=self.adam_b2)
                    fitted   = np.array(fitted_j, dtype=np.float32)
                    del fitted_j

                    for ai in amp_indices:
                        fitted[:, ai] *= safe_scale[:, 0]

                    init_rescaled = init_all.copy()
                    for ai in amp_indices:
                        init_rescaled[:, ai] *= safe_scale[:, 0]
                    fitted = np.where(active[:, None], fitted, init_rescaled)

                    output[:, :, r:r_end] = fitted.reshape(r_count, Az, n_params).transpose(2, 1, 0)
                    del fitted, profiles_flat, profiles_norm, init_all, safe_scale

                    progress.advance(bar_task, advance=r_count)
                    r = r_end

        finally:
            progress_bar.__exit__(None, None, None)
            gc.collect()

        self.logger.subsection(f"GPU extraction complete — {total_attempted:,} / {R * Az:,} active pixels")
        average_quality = self._estimate_r2(output, tomogram_mmap, height_axis, threshold_factor, truncation_index)
        if np.isfinite(average_quality):
            self.logger.subsection(f"Average fit quality (R²): {average_quality:.4f}")
        else:
            self.logger.subsection("Average fit quality (R²): N/A")

        return output

    def _estimate_r2(
        self,
        output           : np.ndarray,
        tomogram_mmap    : np.ndarray,
        height_axis      : np.ndarray,
        threshold_factor : float,
        truncation_index : int,
    ) -> float:
        H, Az, R = tomogram_mmap.shape
        rng      = np.random.default_rng(0)
        idx_flat = rng.choice(Az * R, size=min(self.r2_sample_cap, Az * R), replace=False)
        az_idx   = idx_flat // R
        r_idx    = idx_flat %  R

        profiles = np.abs(tomogram_mmap[:, az_idx, r_idx]).T.astype(np.float64)

        if threshold_factor > 0.0:
            col_max  = profiles.max(axis=1, keepdims=True)
            profiles = np.where(profiles > col_max * threshold_factor, profiles, 0.0)
        if truncation_index < H:
            profiles[:, truncation_index:] = 0.0

        active = profiles.max(axis=1) > 1e-7
        if not active.any():
            return float("nan")

        y    = profiles[active]                                  
        h64  = height_axis.astype(np.float64)
        f    = output[:, az_idx, r_idx].T[active].astype(np.float64)
        amps = f[:, 0::3]
        mus  = f[:, 1::3]
        sigs = np.maximum(f[:, 2::3], 1e-6)
        diff = h64[None, None, :] - mus[:, :, None]
        pred = (amps[:, :, None] * np.exp(np.clip(-(diff ** 2) / (2.0 * (sigs[:, :, None] ** 2) + 1e-10), -100.0, 0.0))).sum(axis=1)

        active_bins = y > 0.0
        y_m    = np.where(active_bins, y,    0.0)
        pred_m = np.where(active_bins, pred, 0.0)

        ss_res = ((y_m - pred_m) ** 2).sum(axis=1)
        ss_tot = ((y_m - (y_m.sum(axis=1, keepdims=True) / np.maximum(active_bins.sum(axis=1, keepdims=True), 1))) ** 2).sum(axis=1)
       
        with np.errstate(invalid="ignore", divide="ignore"):
            r2 = np.where(ss_tot > 1e-20, 1.0 - ss_res / ss_tot, np.nan)
       
        finite = np.isfinite(r2)
        return float(r2[finite].mean()) if finite.any() else float("nan")


