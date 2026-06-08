from __future__ import annotations

import gc
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

import jax
import jax.numpy as jnp

from configuration.param_extraction_config import FitSettings
from pipelines.shared.preprocessing         import ProfilePreprocessor
from tools.gaussians                        import GaussianMixture
from tools.logger                           import Logger


class SigmaScan:
    @staticmethod
    def per_pixel_loss(sigmas, height_axis, profile, amps, mus):
        safe_s2 = 2.0 * jnp.maximum(sigmas, 1e-6) ** 2
        diff    = height_axis[None, :] - mus[:, None]
        expon   = jnp.clip(-(diff ** 2) / safe_s2[:, None], -100.0, 0.0)
        pred    = (amps[:, None] * jnp.exp(expon)).sum(axis=0)
        mse     = jnp.mean((pred - profile) ** 2)

        return mse

    @staticmethod
    def adam_scan(
        batched_vg  ,
        sigmas_init  : jnp.ndarray,
        height_axis  : jnp.ndarray,
        profiles     : jnp.ndarray,
        amps         : jnp.ndarray,
        mus          : jnp.ndarray,
        sigma_lower  : jnp.ndarray,
        sigma_upper  : jnp.ndarray,
        n_steps      : int,
        lr           : float,
        b1           : float,
        b2           : float,
    ) -> jnp.ndarray:
        b1_ = jnp.float32(b1)
        b2_ = jnp.float32(b2)
        eps = jnp.float32(1e-8)
        lr_ = jnp.float32(lr)
        s   = jnp.clip(sigmas_init.astype(jnp.float32), sigma_lower, sigma_upper)
        m   = jnp.zeros_like(s)
        v   = jnp.zeros_like(s)

        def _step(carry, t):
            s_, m_, v_ = carry
            _, g = batched_vg(s_, height_axis, profiles, amps, mus)
            m_   = b1_ * m_ + (1.0 - b1_) * g
            v_   = b2_ * v_ + (1.0 - b2_) * g * g
            tf   = t.astype(jnp.float32) + 1.0
            s_   = s_ - lr_ * (m_ / (1.0 - b1_ ** tf)) / (jnp.sqrt(v_ / (1.0 - b2_ ** tf)) + eps)
            s_   = jnp.clip(s_, sigma_lower, sigma_upper)
            return (s_, m_, v_), None

        (s_final, _, _), _ = jax.lax.scan(_step, (s, m, v), jnp.arange(n_steps))

        return s_final


class SigmaAdamKernel:
    def __init__(self) -> None:
        batched_vg = jax.vmap(jax.value_and_grad(SigmaScan.per_pixel_loss), in_axes=(0, None, 0, 0, 0))
        self._run  = self._build(batched_vg)

    @staticmethod
    def _build(batched_vg):
        @partial(jax.jit, static_argnames=("n_steps", "lr", "b1", "b2"))
        def _run(
            sigmas_init : jnp.ndarray,
            height_axis : jnp.ndarray,
            profiles    : jnp.ndarray,
            amps        : jnp.ndarray,
            mus         : jnp.ndarray,
            sigma_lower : jnp.ndarray,
            sigma_upper : jnp.ndarray,
            n_steps     : int   = 2000,
            lr          : float = 1e-2,
            b1          : float = 0.9,
            b2          : float = 0.999,
        ) -> jnp.ndarray:
            return SigmaScan.adam_scan(batched_vg, sigmas_init, height_axis, profiles, amps, mus, sigma_lower, sigma_upper, n_steps, lr, b1, b2)
        return _run

    def __call__(
        self,
        sigmas_init, height_axis, profiles, amps, mus,
        sigma_lower, sigma_upper,
        n_steps=2000, lr=1e-2, b1=0.9, b2=0.999,
    ):
        return self._run(
            sigmas_init, height_axis, profiles, amps, mus,
            sigma_lower, sigma_upper,
            n_steps, lr, b1, b2,
        )


class PmapSigmaAdamKernel:
    def __init__(self, devices: list) -> None:
        self._devices   = devices
        self._n_devices = len(devices)
        batched_vg      = jax.vmap(jax.value_and_grad(SigmaScan.per_pixel_loss), in_axes=(0, None, 0, 0, 0))
        self._run       = self._build(batched_vg, devices)

    @staticmethod
    def _build(batched_vg, devices):
        def _run_on_device(
            sigmas_init : jnp.ndarray,
            height_axis : jnp.ndarray,
            profiles    : jnp.ndarray,
            amps        : jnp.ndarray,
            mus         : jnp.ndarray,
            sigma_lower : jnp.ndarray,
            sigma_upper : jnp.ndarray,
            n_steps     : int   = 2000,
            lr          : float = 1e-2,
            b1          : float = 0.9,
            b2          : float = 0.999,
        ) -> jnp.ndarray:
            return SigmaScan.adam_scan(batched_vg, sigmas_init, height_axis, profiles, amps, mus, sigma_lower, sigma_upper, n_steps, lr, b1, b2)

        return jax.pmap(
            _run_on_device,
            in_axes                    = (0, None, 0, 0, 0, None, None),
            static_broadcasted_argnums = (7, 8, 9, 10),
            devices                    = devices,
        )

    def __call__(
        self,
        sigmas_init, height_axis, profiles, amps, mus,
        sigma_lower, sigma_upper,
        n_steps=2000, lr=1e-2, b1=0.9, b2=0.999,
    ):
        n, K = sigmas_init.shape
        H    = profiles.shape[1]
        D    = self._n_devices
        pad  = (-n) % D

        if pad > 0:
            z_K = jnp.zeros((pad, K), dtype=jnp.float32)
            z_H = jnp.zeros((pad, H), dtype=jnp.float32)
            sigmas_init = jnp.concatenate([sigmas_init, z_K], axis=0)
            profiles    = jnp.concatenate([profiles,    z_H], axis=0)
            amps        = jnp.concatenate([amps,        z_K], axis=0)
            mus         = jnp.concatenate([mus,         z_K], axis=0)

        n_pad = n + pad
        shard = n_pad // D

        out_s = self._run(
            sigmas_init.reshape(D, shard, K),
            height_axis,
            profiles   .reshape(D, shard, H),
            amps       .reshape(D, shard, K),
            mus        .reshape(D, shard, K),
            sigma_lower,
            sigma_upper,
            n_steps, lr, b1, b2,
        )

        return out_s.reshape(n_pad, K)[:n]


class PeakInitialiser:
    def __init__(self, n_workers : int = 1) -> None:
        self.n_workers = n_workers
        self._pool     = ProcessPoolExecutor(max_workers=n_workers)
        list(self._pool.map(abs, range(n_workers)))

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _prominence_worker(raw_chunk : np.ndarray, height_axis : np.ndarray, K : int, sigma_guess : float, min_dist : int, prominence_frac : float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        chunk_N, H = raw_chunk.shape
        amps       = np.zeros((chunk_N, K), dtype=np.float32)
        mus        = np.zeros((chunk_N, K), dtype=np.float32)
        sigs       = np.full ((chunk_N, K), sigma_guess, dtype=np.float32)

        for n in range(chunk_N):
            raw  = raw_chunk[n]
            pmax = raw.max()
            if pmax < 1e-10:
                idxs = np.linspace(0, H - 1, K, dtype=int)
            else:
                peaks, props = find_peaks(raw, prominence=pmax * prominence_frac, distance=min_dist)

                if len(peaks) > 0:
                    peaks = peaks[np.argsort(props["prominences"])[::-1]]

                if len(peaks) >= K:
                    idxs = peaks[:K]

                elif len(peaks) > 0:
                    residual = raw.copy()
                    extra    = []

                    for p in peaks:
                        lo = max(0, p - min_dist)
                        hi = min(H, p + min_dist + 1)
                        residual[lo:hi] = 0.0

                    for _ in range(K - len(peaks)):
                        ei = int(np.argmax(residual))
                        extra.append(ei)
                        lo = max(0, ei - min_dist)
                        hi = min(H, ei + min_dist + 1)
                        residual[lo:hi] = 0.0

                    idxs = np.concatenate([peaks, np.array(extra, dtype=int)])

                else:
                    idxs = np.linspace(0, H - 1, K, dtype=int)

            for g, idx in enumerate(idxs[:K]):
                amps[n, g] = max(float(raw[idx]), 1e-10)
                mus [n, g] = float(height_axis[idx])

        return amps, mus, sigs

    def run(self, prof_raw : np.ndarray, height_axis : np.ndarray, K : int, prominence_frac : float = 0.05, sigma_divisor : float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N, H        = prof_raw.shape
        h_span      = float(height_axis[-1] - height_axis[0])
        dh          = float(height_axis[1] - height_axis[0])
        sigma_base  = max(2.0 * dh, h_span / (8.0 * K))
        sigma_guess = sigma_base / max(float(sigma_divisor), 1e-6)
        min_dist    = max(1, int(sigma_base / dh))
        raw         = prof_raw.astype(np.float32, copy=False)

        chunk_size = max(1, -(-N // (self.n_workers * 2)))
        chunks     = [raw[i:i + chunk_size] for i in range(0, N, chunk_size)]

        worker_fn  = partial(
            self._prominence_worker,
            height_axis     = height_axis,
            K               = K,
            sigma_guess     = sigma_guess,
            min_dist        = min_dist,
            prominence_frac = prominence_frac,
        )

        chunk_results = list(self._pool.map(worker_fn, chunks))

        amps = np.concatenate([r[0] for r in chunk_results], axis=0)
        mus  = np.concatenate([r[1] for r in chunk_results], axis=0)
        sigs = np.concatenate([r[2] for r in chunk_results], axis=0)

        return amps, mus, sigs


class BestKSelector:
    def __init__(self, k_max : int, lambda_k : float, logger : Logger) -> None:
        self.k_max    = k_max
        self.lambda_k = lambda_k
        self.logger   = logger

    def _score_K(self, K : int, gpu_results : Dict[int, tuple], prof_norm_all : np.ndarray, height_axis : np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        amps_norm, mus, final_sigs = gpu_results[K]
        pred       = GaussianMixture.evaluate_batch(height_axis, amps_norm, mus, final_sigs)
        mse        = ((pred - prof_norm_all) ** 2).mean(axis=1)
        complexity = self.lambda_k * K * amps_norm.mean(axis=1)

        return K, mse, mse + complexity

    def select(
        self,
        gpu_results   : Dict[int, tuple],
        prof_norm_all : np.ndarray,
        scale_all     : np.ndarray,
        height_axis   : np.ndarray,
        n_params_out  : int,
        batch_tag     : str = "",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        N_act = prof_norm_all.shape[0]
        tag   = f"{batch_tag} | " if batch_tag else ""

        self.logger.section(f"[{tag}Phase 3 — Penalty Scoring & Best-K Selection]")
        self.logger.subsection(f"lambda_k : {self.lambda_k}")

        penalised_all = np.empty((N_act, self.k_max), dtype=np.float64)
        mse_all       = np.empty((N_act, self.k_max), dtype=np.float64)

        with self.logger.track(transient=True) as progress:
            bar = progress.add_task("  [section]Scoring K values[/section]", total=self.k_max)
            with ThreadPoolExecutor(max_workers=self.k_max) as tpool:
                for K, mse, pen in tpool.map(lambda K: self._score_K(K, gpu_results, prof_norm_all, height_axis), range(1, self.k_max + 1)):
                    mse_all      [:, K - 1] = mse
                    penalised_all[:, K - 1] = pen
                    progress.advance(bar)

        best_K_idx = penalised_all.argmin(axis=1)

        best_params = np.zeros((N_act, n_params_out), dtype=np.float32)
        for K in range(1, self.k_max + 1):
            mask = best_K_idx == (K - 1)

            if not mask.any():
                continue

            amps_norm, mus, final_sigs = gpu_results[K]
            amps_out = amps_norm * scale_all[:, None]
            idx      = np.where(mask)[0]

            best_params[np.ix_(idx, list(range(0, K * 3, 3)))] = amps_out  [idx]
            best_params[np.ix_(idx, list(range(1, K * 3, 3)))] = mus       [idx]
            best_params[np.ix_(idx, list(range(2, K * 3, 3)))] = final_sigs[idx]

        k_dist = {K: int((best_K_idx == K - 1).sum()) for K in range(1, self.k_max + 1)}
        self.logger.subsection(f"Best-K dist     : {k_dist}")
        self.logger.subsection(f"Mean MSE (best) : {float(mse_all[np.arange(N_act), best_K_idx].mean()):.5f}")

        return best_params, mse_all.astype(np.float32), penalised_all.astype(np.float32), best_K_idx.astype(np.int16)


class KernelBackendSelector:
    def __init__(self, gpu_device_ids : Optional[List[int]] = None) -> None:
        self.gpu_device_ids = gpu_device_ids

    def select(self) -> Tuple[object, int, str, list]:
        all_gpu_devices = [d for d in jax.devices() if d.platform in ("gpu", "cuda")]
        active_devices  = ([all_gpu_devices[i] for i in self.gpu_device_ids] if self.gpu_device_ids else all_gpu_devices) if all_gpu_devices else jax.devices()

        if len(active_devices) > 1:
            kernel    = PmapSigmaAdamKernel(active_devices)
            n_devices = len(active_devices)
            backend   = f"pmap  ({n_devices} GPUs)"
        else:
            kernel    = SigmaAdamKernel()
            n_devices = 1
            backend   = "jit  (1 GPU)"

        return kernel, n_devices, backend, active_devices


class SigmaFittingExtractor:
    def __init__(
        self,
        fit_settings        : FitSettings,
        logger              : Logger,
        range_batch_size    : int                 = 256,
        adam_steps          : int                 = 2000,
        adam_lr             : float               = 1e-2,
        adam_b1             : float               = 0.9,
        adam_b2             : float               = 0.999,
        k_max               : int                 = 5,
        lambda_k            : float               = 3e-3,
        prominence_frac     : float               = 0.05,
        sigma_init_divisor  : float               = 1.0,
        gpu_pixel_batch_size: int                 = 8192,
        gpu_device_ids      : Optional[List[int]] = None,
        init_workers        : Optional[int]       = None,
    ) -> None:

        self.fit_settings         = fit_settings
        self.logger               = logger
        self.range_batch_size     = range_batch_size
        self.adam_steps           = adam_steps
        self.adam_lr              = adam_lr
        self.adam_b1              = adam_b1
        self.adam_b2              = adam_b2
        self.k_max                = k_max
        self.lambda_k             = lambda_k
        self.prominence_frac      = prominence_frac
        self.sigma_init_divisor   = sigma_init_divisor
        self.gpu_pixel_batch_size = gpu_pixel_batch_size
        self._init_workers        = 80 if init_workers is None else init_workers

        self._peak_initialiser = PeakInitialiser(n_workers=self._init_workers)
        self._best_k_selector  = BestKSelector(k_max=k_max, lambda_k=lambda_k, logger=logger)

        kernel, n_devices, backend, active_devices = KernelBackendSelector(gpu_device_ids).select()

        self._kernel    = kernel
        self._n_devices = n_devices

        self.logger.section("[GPU Parameter Extractor]")
        self.logger.subsection(f"JAX active devices : {active_devices}")
        self.logger.subsection(f"Backend            : {backend}")
        self.logger.subsection(f"range_batch_size   : {range_batch_size}")
        self.logger.subsection(f"gpu_pixel_batch    : {gpu_pixel_batch_size}")
        self.logger.subsection(f"adam_steps         : {adam_steps}")
        self.logger.subsection(f"adam_lr            : {adam_lr}")
        self.logger.subsection(f"k_max              : {k_max}")
        self.logger.subsection(f"lambda_k           : {lambda_k}")
        self.logger.subsection(f"sigma_init_divisor : {sigma_init_divisor}")
        self.logger.subsection(f"init_workers       : {self._init_workers}")
        self.logger.subsection(f"n_devices          : {self._n_devices}")

    def _prepare_data(
        self,
        tomogram_path : Path,
        height_range  : Tuple[float, float],
    ) -> tuple:

        fit_cfg       = self.fit_settings.fit_config
        n_params_out  = 3 * self.k_max

        tomogram_mmap = np.load(str(tomogram_path), mmap_mode="r", allow_pickle=False)
        H, Az, R      = tomogram_mmap.shape
        height_axis   = np.linspace(height_range[0], height_range[1], H, dtype=np.float32)
        height_ax_j   = jnp.array(height_axis)

        h_span        = float(height_axis[-1] - height_axis[0])
        dh            = h_span / (H - 1)

        sigma_lower_j = jnp.float32(dh)
        sigma_upper_j = jnp.float32(h_span / 2.0)

        threshold_factor = float(fit_cfg.threshold_factor)
        truncation_index = int(  fit_cfg.truncation_index)

        self.logger.section("[Data Preparation]")
        self.logger.subsection(f"H            : {H}")
        self.logger.subsection(f"Az           : {Az}")
        self.logger.subsection(f"R            : {R}")
        self.logger.subsection(f"k_max        : {self.k_max}")
        self.logger.subsection(f"Total pixels : {R * Az:,}")
        self.logger.subsection(f"Batches      : {-(-R // self.range_batch_size)}")

        return (
            tomogram_mmap, H, Az, R,
            height_axis, height_ax_j,
            sigma_lower_j, sigma_upper_j,
            n_params_out,
            threshold_factor, truncation_index,
        )

    def _warmup_kernel(
        self,
        height_ax_j   : jnp.ndarray,
        H             : int,
        sigma_lower_j : jnp.ndarray,
        sigma_upper_j : jnp.ndarray,
    ) -> None:

        self.logger.section("[Kernel Compilation]")
        self.logger.subsection(f"Compiling sigma-only JAX kernel for K={self.k_max}")

        N_warm  = self._n_devices * max(1, 4 // self._n_devices)
        K       = self.k_max

        dummy_s = jnp.ones((N_warm, K),  dtype=jnp.float32) * 5.0
        dummy_p = jnp.ones((N_warm, H),  dtype=jnp.float32)
        dummy_a = jnp.ones((N_warm, K),  dtype=jnp.float32) * 0.5
        dummy_m = jnp.zeros((N_warm, K), dtype=jnp.float32)

        self._kernel(dummy_s, height_ax_j, dummy_p, dummy_a, dummy_m, sigma_lower_j, sigma_upper_j, 2, self.adam_lr, self.adam_b1, self.adam_b2)

        self.logger.subsection(f"Kernel compiled (K={K})")

    def _load_batch(
        self,
        tomogram_mmap    : np.ndarray,
        r_start          : int,
        R                : int,
        Az               : int,
        H                : int,
        threshold_factor : float,
        truncation_index : int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:

        r_end = min(r_start + self.range_batch_size, R)
        r_c   = r_end - r_start
        raw   = np.abs(np.array(tomogram_mmap[:, :, r_start:r_end])).astype(np.float32)
        raw   = ProfilePreprocessor.apply(raw, threshold_factor, truncation_index)

        pf     = raw.transpose(2, 1, 0).reshape(r_c * Az, H).copy()
        active = pf.max(axis=1) > 1e-3
        pmax   = pf.max(axis=1, keepdims=True)
        scale  = np.where(active[:, None], pmax, 1.0).astype(np.float32)
        norm   = pf / scale

        del raw

        return pf, norm, scale, active, r_end

    @staticmethod
    def _pad_rows(arr : np.ndarray, target : int) -> np.ndarray:
        pad = target - arr.shape[0]
        if pad == 0:
            return np.ascontiguousarray(arr, dtype=np.float32)
        return np.concatenate([arr.astype(np.float32, copy=False), np.zeros((pad, arr.shape[1]), dtype=np.float32)], axis=0)

    def _fit_all_K(
        self,
        inits         : dict,
        prof_norm_all : np.ndarray,
        scale_all     : np.ndarray,
        height_ax_j   : jnp.ndarray,
        sigma_lower_j : jnp.ndarray,
        sigma_upper_j : jnp.ndarray,
        N_act         : int,
        batch_tag     : str,
    ) -> dict:

        gpu_results = {}
        B           = self.gpu_pixel_batch_size

        self.logger.section(f"[{batch_tag} | Phase 2 — GPU Sigma Fitting]")

        for K in range(1, self.k_max + 1):
            amps_raw, mus, sigs_init = inits[K]
            amps_norm  = amps_raw / scale_all[:, None]
            final_sigs = np.empty((N_act, K), dtype=np.float32)

            for i_start in range(0, N_act, B):
                i_end   = min(i_start + B, N_act)
                n_chunk = i_end - i_start
                out_s   = self._kernel(
                    jnp.array(self._pad_rows(sigs_init    [i_start:i_end], B)),
                    height_ax_j,
                    jnp.array(self._pad_rows(prof_norm_all[i_start:i_end], B)),
                    jnp.array(self._pad_rows(amps_norm    [i_start:i_end], B)),
                    jnp.array(self._pad_rows(mus          [i_start:i_end], B)),
                    sigma_lower_j,
                    sigma_upper_j,
                    self.adam_steps,
                    self.adam_lr,
                    self.adam_b1,
                    self.adam_b2,
                )

                final_sigs[i_start:i_end] = np.array(out_s[:n_chunk], dtype=np.float32)
                del out_s

            gpu_results[K] = (amps_norm, mus, final_sigs)
            self.logger.subsection(f"K={K} done")

        gc.collect()

        return gpu_results

    def _fit_batch(
        self,
        profiles_flat : np.ndarray,
        profiles_norm : np.ndarray,
        active        : np.ndarray,
        safe_scale    : np.ndarray,
        height_axis   : np.ndarray,
        height_ax_j   : jnp.ndarray,
        sigma_lower_j : jnp.ndarray,
        sigma_upper_j : jnp.ndarray,
        n_params_out  : int,
        batch_tag     : str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        N          = profiles_flat.shape[0]
        output     = np.zeros((N, n_params_out),    dtype=np.float32)
        mse_out    = np.full ((N, self.k_max), np.nan, dtype=np.float32)
        pen_out    = np.full ((N, self.k_max), np.nan, dtype=np.float32)
        best_k_out = np.zeros(N,                    dtype=np.int16)
        active_idx = np.where(active)[0]

        if len(active_idx) == 0:
            return output, mse_out, pen_out, best_k_out

        prof_raw_all  = profiles_flat[active_idx]
        prof_norm_all = profiles_norm[active_idx].astype(np.float32)
        scale_all     = safe_scale[active_idx, 0]
        N_act         = len(active_idx)

        n_cpus = os.cpu_count() or 1
        self.logger.section(f"[{batch_tag} | Phase 1 — CPU Initialisation]")
        self.logger.subsection(f"Active pixels : {N_act}")
        self.logger.subsection(f"K             : {self.k_max} (shared init)")
        self.logger.subsection(f"Workers       : {self._init_workers} / {n_cpus} logical CPUs")

        amps_km, mus_km, sigs_km = self._peak_initialiser.run(prof_raw_all, height_axis, self.k_max, self.prominence_frac, self.sigma_init_divisor)
        inits = {K: (amps_km[:, :K].copy(), mus_km[:, :K].copy(), sigs_km[:, :K].copy()) for K in range(1, self.k_max + 1)}

        self.logger.subsection(f"Init shared for all {self.k_max} K values")

        gpu_results = self._fit_all_K(inits, prof_norm_all, scale_all, height_ax_j, sigma_lower_j, sigma_upper_j, N_act, batch_tag)

        best_params, mse_act, pen_act, best_idx_act = self._best_k_selector.select(gpu_results, prof_norm_all, scale_all, height_axis, n_params_out, batch_tag=batch_tag)

        output    [active_idx] = best_params
        mse_out   [active_idx] = mse_act
        pen_out   [active_idx] = pen_act
        best_k_out[active_idx] = best_idx_act + 1

        return output, mse_out, pen_out, best_k_out

    def _run_fitting(
        self,
        tomogram_mmap    : np.ndarray,
        height_axis      : np.ndarray,
        height_ax_j      : jnp.ndarray,
        sigma_lower_j    : jnp.ndarray,
        sigma_upper_j    : jnp.ndarray,
        threshold_factor : float,
        truncation_index : int,
        Az               : int,
        R                : int,
        H                : int,
        n_params_out     : int,
        output           : np.ndarray,
        mse_maps         : np.ndarray,
        penalised_maps   : np.ndarray,
        best_k_map       : np.ndarray,
    ) -> int:

        n_batches = -(-R // self.range_batch_size)

        self.logger.section("[Range Bin Loading]")
        self.logger.subsection("Streaming range batches (load fused with fitting)")
        self.logger.subsection(f"Range batches : {n_batches} x {self.range_batch_size} bins, phases 1-3 repeat once per batch")

        total_attempted = 0

        with self.logger.track(transient=True) as progress:
            bar_task = progress.add_task("  [section]Processing range bins[/section]", total=R,)

            with ThreadPoolExecutor(max_workers=2) as pool:
                r               = 0
                batch_index     = 0
                prefetch_future = pool.submit(self._load_batch, tomogram_mmap, r, R, Az, H, threshold_factor, truncation_index,)

                try:
                    while r < R:
                        profiles_flat, profiles_norm, safe_scale, active, r_end = prefetch_future.result()

                        r_start      = r
                        r_count      = r_end - r
                        batch_index += 1
                        batch_tag    = f"Batch {batch_index}/{n_batches}"

                        if r_end < R:
                            prefetch_future = pool.submit(self._load_batch, tomogram_mmap, r_end, R, Az, H, threshold_factor, truncation_index)

                        total_attempted += int(active.sum())

                        fitted, mse_batch, pen_batch, best_k_batch = self._fit_batch(
                            profiles_flat, profiles_norm,
                            active, safe_scale,
                            height_axis, height_ax_j,
                            sigma_lower_j, sigma_upper_j,
                            n_params_out,
                            batch_tag,
                        )

                        output        [:, :, r_start:r_end] = fitted.reshape(r_count, Az, n_params_out).transpose(2, 1, 0)
                        mse_maps      [:, :, r_start:r_end] = mse_batch.reshape(r_count, Az, self.k_max).transpose(2, 1, 0)
                        penalised_maps[:, :, r_start:r_end] = pen_batch.reshape(r_count, Az, self.k_max).transpose(2, 1, 0)
                        best_k_map    [:,    r_start:r_end] = best_k_batch.reshape(r_count, Az).T

                        del profiles_flat, profiles_norm, safe_scale, active, fitted, mse_batch, pen_batch, best_k_batch
                        gc.collect()

                        self.logger.subsection(f"{batch_tag} complete — range bins {r_start}-{r_end} of {R}")

                        progress.advance(bar_task, advance=r_count)
                        r = r_end

                except Exception:
                    prefetch_future.cancel()
                    raise

        self.logger.subsection(f"Total pixels   : {R * Az:,}")
        self.logger.subsection(f"Active pixels  : {total_attempted:,}")

        return total_attempted

    def run(
        self,
        tomogram_path : Path,
        height_range  : Tuple[float, float],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

        (
            tomogram_mmap, H, Az, R,
            height_axis, height_ax_j,
            sigma_lower_j, sigma_upper_j,
            n_params_out,
            threshold_factor, truncation_index,
        ) = self._prepare_data(tomogram_path, height_range)

        self._warmup_kernel(height_ax_j, H, sigma_lower_j, sigma_upper_j)

        output         = np.zeros((n_params_out, Az, R),        dtype=np.float32)
        mse_maps       = np.full ((self.k_max, Az, R), np.nan,  dtype=np.float32)
        penalised_maps = np.full ((self.k_max, Az, R), np.nan,  dtype=np.float32)
        best_k_map     = np.zeros((Az, R),                      dtype=np.int16)

        try:
            total_attempted = self._run_fitting(
                tomogram_mmap, height_axis, height_ax_j,
                sigma_lower_j, sigma_upper_j,
                threshold_factor, truncation_index,
                Az, R, H, n_params_out, output,
                mse_maps, penalised_maps, best_k_map,
            )
        finally:
            self._peak_initialiser.close()
            jax.clear_caches()
            gc.collect()

        self.logger.section("[Results]")
        self.logger.subsection(f"Active pixels fitted : {total_attempted:,} / {R * Az:,}")

        diagnostics = {
            "mse_per_k"       : mse_maps,
            "penalised_per_k" : penalised_maps,
            "best_k_map"      : best_k_map,
            "lambda_k"        : np.float32(self.lambda_k),
        }

        return output, diagnostics
