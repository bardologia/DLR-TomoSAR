from __future__ import annotations

import gc
import os
import warnings
warnings.filterwarnings("ignore", message=".*pynvml.*", category=FutureWarning)
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

import jax
import jax.numpy as jnp

from configuration.param_extraction_config import FitMode, FitSettings
from tools.logger import Logger


def _evaluate_gaussian(height_axis : np.ndarray, amps : np.ndarray, mus : np.ndarray, sigs : np.ndarray) -> np.ndarray:                
    pred = np.zeros((len(amps), len(height_axis)), dtype=np.float32)
    h    = height_axis[None, :]
    
    for g in range(amps.shape[1]):
        sig   = np.maximum(sigs[:, g:g+1], 1e-6)
        pred += amps[:, g:g+1] * np.exp(-((h - mus[:, g:g+1]) ** 2) / (2.0 * sig ** 2))
    
    return pred


def _prominence_worker(smoothed_chunk : np.ndarray, height_axis : np.ndarray, K : int, sigma_guess : float, min_dist : int, prominence_frac : float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    chunk_N, H = smoothed_chunk.shape
    amps = np.zeros((chunk_N, K), dtype=np.float32)
    mus  = np.zeros((chunk_N, K), dtype=np.float32)
    sigs = np.full ((chunk_N, K), sigma_guess, dtype=np.float32)

    for n in range(chunk_N):
        smo  = smoothed_chunk[n]
        pmax = smo.max()
        if pmax < 1e-10:
            idxs = np.linspace(0, H - 1, K, dtype=int)
        else:
            peaks, props = find_peaks(smo, prominence=pmax * prominence_frac, distance=min_dist)
            if len(peaks) >= K:
                idxs = peaks[np.argsort(props["prominences"])[::-1][:K]]
           
            elif len(peaks) > 0:
                residual        = smo.copy()
                residual[peaks] = 0.0
                extra           = []
                
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
            amps[n, g] = max(float(smo[idx]), 1e-10)
            mus [n, g] = float(height_axis[idx])
    
    return amps, mus, sigs


def _prominence_batch(prof_raw : np.ndarray, height_axis : np.ndarray, K : int, prominence_frac : float = 0.05, n_workers : int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, H        = prof_raw.shape
    h_span      = float(height_axis[-1] - height_axis[0])
    dh          = float(height_axis[1] - height_axis[0])
    sigma_guess = max(2.0 * dh, h_span / (8.0 * K))
    min_dist    = max(1, int(sigma_guess / dh))
    smoothed    = uniform_filter1d(prof_raw.astype(np.float32), size=5, mode="nearest", axis=1).copy()

    chunk_size = max(1, N // (n_workers * 8))
    chunks     = [smoothed[i:i + chunk_size] for i in range(0, N, chunk_size)]
    
    worker_fn  = partial(
        _prominence_worker,
        height_axis     = height_axis,
        K               = K,
        sigma_guess     = sigma_guess,
        min_dist        = min_dist,
        prominence_frac = prominence_frac,
    )

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        chunk_results = list(pool.map(worker_fn, chunks))

    amps = np.concatenate([r[0] for r in chunk_results], axis=0)
    mus  = np.concatenate([r[1] for r in chunk_results], axis=0)
    sigs = np.concatenate([r[2] for r in chunk_results], axis=0)
  
    return amps, mus, sigs


class SigmaAdamKernel:
    def __init__(self) -> None:
        batched_vg = jax.vmap(jax.value_and_grad(self._per_pixel_loss), in_axes=(0, None, 0, 0, 0))
        self._run = self._build(batched_vg)

    @staticmethod
    def _per_pixel_loss(sigmas, height_axis, profile, amps, mus):
        safe_s2 = 2.0 * jnp.maximum(sigmas, 1e-6) ** 2
        diff    = height_axis[None, :] - mus[:, None]
        expon   = jnp.clip(-(diff ** 2) / safe_s2[:, None], -100.0, 0.0)
        pred    = (amps[:, None] * jnp.exp(expon)).sum(axis=0)
        mse     = jnp.mean((pred - profile) ** 2)
     
        return mse
    
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
        batched_vg      = jax.vmap(jax.value_and_grad(SigmaAdamKernel._per_pixel_loss), in_axes=(0, None, 0, 0, 0))
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
        gpu_pixel_batch_size: int                 = 8192,
        r2_sample_cap       : int                 = 4096,
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
        self.r2_sample_cap        = r2_sample_cap
        self.gpu_pixel_batch_size = gpu_pixel_batch_size
        self._init_workers        = 80 if init_workers is None else init_workers

        all_gpu_devices = [d for d in jax.devices() if d.platform in ("gpu", "cuda")]
        active_devices  = ([all_gpu_devices[i] for i in gpu_device_ids] if gpu_device_ids else all_gpu_devices) if all_gpu_devices else jax.devices()

        if len(active_devices) > 1:
            self._kernel    = PmapSigmaAdamKernel(active_devices)
            self._n_devices = len(active_devices)
            backend         = f"pmap  ({self._n_devices} GPUs)"
        else:
            self._kernel    = SigmaAdamKernel()
            self._n_devices = 1
            backend         = "jit  (1 GPU)"

        self.logger.section("[GPU Parameter Extractor]")
        self.logger.subsection(f"JAX active devices : {active_devices}")
        self.logger.subsection(f"Backend            : {backend}")
        self.logger.subsection(f"range_batch_size   : {range_batch_size}")
        self.logger.subsection(f"gpu_pixel_batch    : {gpu_pixel_batch_size}")
        self.logger.subsection(f"adam_steps         : {adam_steps}")
        self.logger.subsection(f"adam_lr            : {adam_lr}")
        self.logger.subsection(f"k_max              : {k_max}")
        self.logger.subsection(f"lambda_k           : {lambda_k}")
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

        threshold_factor = float(getattr(fit_cfg, "threshold_factor", 0.0))
        truncation_index = int(  getattr(fit_cfg, "truncation_index", H))

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
        self.logger.subsection("Compiling sigma-only JAX kernels")

        N_warm = self._n_devices * max(1, 4 // self._n_devices)
        for K in range(1, self.k_max + 1):
            dummy_s = jnp.ones((N_warm, K), dtype=jnp.float32) * 5.0
            dummy_p = jnp.ones((N_warm, H), dtype=jnp.float32)
            dummy_a = jnp.ones((N_warm, K), dtype=jnp.float32) * 0.5
            dummy_m = jnp.zeros((N_warm, K), dtype=jnp.float32)
            
            self._kernel(
                dummy_s, height_ax_j, dummy_p, dummy_a, dummy_m,
                sigma_lower_j, sigma_upper_j,
                2, self.adam_lr, self.adam_b1, self.adam_b2,
            )
        
        self.logger.subsection(f"Shapes compiled : K=1 … {self.k_max}")

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

        if threshold_factor > 0.0:
            raw = np.where(raw > raw.max(axis=0, keepdims=True) * threshold_factor, raw, 0.0)
        
        if truncation_index < H:
            raw[truncation_index:, :, :] = 0.0

        pf     = raw.transpose(2, 1, 0).reshape(r_c * Az, H).copy()
        active = pf.max(axis=1) > 1e-7
        pmax   = pf.max(axis=1, keepdims=True)
        scale  = np.where(active[:, None], pmax, 1.0).astype(np.float32)
        norm   = pf / scale

        del raw
        
        return pf, norm, scale, active, r_end

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
    ) -> np.ndarray:                   
        
        N          = profiles_flat.shape[0]
        output     = np.zeros((N, n_params_out), dtype=np.float32)
        active_idx = np.where(active)[0]

        if len(active_idx) == 0:
            return output

        prof_raw_all  = profiles_flat[active_idx]                    
        prof_norm_all = profiles_norm[active_idx].astype(np.float32) 
        scale_all     = safe_scale[active_idx, 0]                    
        N_act         = len(active_idx)

        n_cpus = os.cpu_count() or 1
        self.logger.section("[Phase 1 — CPU Initialisation]")
        self.logger.subsection(f"Active pixels : {N_act}")
        self.logger.subsection(f"K             : {self.k_max} (shared init)")
        self.logger.subsection(f"Workers       : {self._init_workers} / {n_cpus} logical CPUs")
     
        amps_km, mus_km, sigs_km = _prominence_batch(prof_raw_all, height_axis, self.k_max, self.prominence_frac, self._init_workers)
        inits = {K: (amps_km[:, :K].copy(), mus_km[:, :K].copy(), sigs_km[:, :K].copy()) for K in range(1, self.k_max + 1)}

        self.logger.subsection(f"Init shared for all {self.k_max} K values")

        gpu_results = {}  
        B = self.gpu_pixel_batch_size

        self.logger.section("[Phase 2 — GPU Sigma Fitting]")

        for K in range(1, self.k_max + 1):
            amps_raw, mus, sigs_init = inits[K]
            amps_norm  = amps_raw / scale_all[:, None]               
            final_sigs = np.empty((N_act, K), dtype=np.float32)

            for i_start in range(0, N_act, B):
                i_end = min(i_start + B, N_act)
                out_s = self._kernel(
                    jnp.array(sigs_init  [i_start:i_end], dtype=jnp.float32),
                    height_ax_j,
                    jnp.array(prof_norm_all[i_start:i_end], dtype=jnp.float32),
                    jnp.array(amps_norm [i_start:i_end],    dtype=jnp.float32),
                    jnp.array(mus[i_start:i_end],           dtype=jnp.float32),
                    sigma_lower_j,
                    sigma_upper_j,
                    self.adam_steps,
                    self.adam_lr,
                    self.adam_b1,
                    self.adam_b2,
                )
                
                final_sigs[i_start:i_end] = np.array(out_s, dtype=np.float32)
                del out_s

            gpu_results[K] = (amps_norm, mus, final_sigs)
            self.logger.subsection(f"K={K} done")

        jax.clear_caches()
        gc.collect()
        
        self.logger.subsection("GPU memory released")
        self.logger.section("[Phase 3 — Penalty Scoring & Best-K Selection]")
        self.logger.subsection(f"lambda_k : {self.lambda_k}")

        penalised_all = np.empty((N_act, self.k_max), dtype=np.float64)
        mse_all       = np.empty((N_act, self.k_max), dtype=np.float64)

        def _score_K(K):
            amps_norm, mus, final_sigs = gpu_results[K]
            pred       = _evaluate_gaussian(height_axis, amps_norm, mus, final_sigs)
            mse        = ((pred - prof_norm_all) ** 2).mean(axis=1)
            complexity = self.lambda_k * K * amps_norm.mean(axis=1)
            
            return K, mse, mse + complexity

        with self.logger.track(transient=True) as progress:
            bar = progress.add_task("  [section]Scoring K values[/section]", total=self.k_max)
            with ThreadPoolExecutor(max_workers=self.k_max) as tpool:
                for K, mse, pen in tpool.map(_score_K, range(1, self.k_max + 1)):
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

        output[active_idx] = best_params
        return output

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
    ) -> int:
        
        self.logger.section("[Range Bin Loading]")
        self.logger.subsection(f"Loading all range batches into memory")

        all_profiles_flat : list = []
        all_profiles_norm : list = []
        all_safe_scale    : list = []
        all_active        : list = []

        progress_bar = self.logger.track(transient=True)
        progress     = progress_bar.__enter__()
        bar_task     = progress.add_task("  [section]Loading range bins[/section]", total=R,)

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                r               = 0
                prefetch_future = pool.submit(self._load_batch, tomogram_mmap, r, R, Az, H, threshold_factor, truncation_index,)

                try:
                    while r < R:
                        profiles_flat, profiles_norm, safe_scale, active, r_end = \
                            prefetch_future.result()
                        
                        r_count = r_end - r

                        if r_end < R:
                            prefetch_future = pool.submit(self._load_batch, tomogram_mmap, r_end, R, Az, H, threshold_factor, truncation_index)

                        all_profiles_flat.append(profiles_flat)
                        all_profiles_norm.append(profiles_norm)
                        all_safe_scale   .append(safe_scale)
                        all_active       .append(active)

                        progress.advance(bar_task, advance=r_count)
                        r = r_end
                
                except Exception:
                    prefetch_future.cancel()
                    raise
        finally:
            progress_bar.__exit__(None, None, None)

        profiles_flat_all = np.concatenate(all_profiles_flat, axis=0)
        profiles_norm_all = np.concatenate(all_profiles_norm, axis=0)
        safe_scale_all    = np.concatenate(all_safe_scale,    axis=0)
        active_all        = np.concatenate(all_active,        axis=0)
        
        del all_profiles_flat, all_profiles_norm, all_safe_scale, all_active

        total_attempted = int(active_all.sum())
        self.logger.subsection(f"Total pixels   : {R * Az:,}")
        self.logger.subsection(f"Active pixels  : {total_attempted:,}")

        fitted_all = self._fit_batch(
            profiles_flat_all, profiles_norm_all,
            active_all, safe_scale_all,
            height_axis, height_ax_j,
            sigma_lower_j, sigma_upper_j,
            n_params_out,
        )

        del profiles_flat_all, profiles_norm_all, safe_scale_all, active_all
        output[:, :, :] = fitted_all.reshape(R, Az, n_params_out).transpose(2, 1, 0)
        del fitted_all
        gc.collect()

        return total_attempted

    def _estimate_r2(
        self,
        output           : np.ndarray,
        tomogram_mmap    : np.ndarray,
        height_axis      : np.ndarray,
        threshold_factor : float,
        truncation_index : int,
    ) -> dict:
        
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
        nan_stats = dict(mse_norm=float("nan"), mse_denorm=float("nan"), r2_valid=False)
        if not active.any():
            return nan_stats

        y    = profiles[active]
        h64  = height_axis.astype(np.float64)
        f    = output[:, az_idx, r_idx].T[active].astype(np.float64)
        amps = f[:, 0::3]
        mus  = f[:, 1::3]
        sigs = np.maximum(f[:, 2::3], 1e-6)

        diff = h64[None, None, :] - mus[:, :, None]
        pred = (amps[:, :, None] * np.exp(np.clip(-(diff ** 2) / (2.0 * sigs[:, :, None] ** 2 + 1e-10), -100.0, 0.0))).sum(axis=1)

        pmax_y       = y.max(axis=1, keepdims=True)
        safe_pmax    = np.where(pmax_y > 1e-10, pmax_y, 1.0)
        y_norm       = y    / safe_pmax
        pred_norm    = pred / safe_pmax
        active_bins  = y > 0.0
        n_active     = np.maximum(active_bins.sum(axis=1), 1)
        mse_norm     = float(((active_bins * (y_norm  - pred_norm) ** 2).sum(axis=1) / n_active).mean())

        mse_denorm   = float(((active_bins * (y - pred) ** 2).sum(axis=1) / n_active).mean())

        pred_m  = np.where(active_bins, pred, 0.0)
        mean_y  = (y * active_bins).sum(axis=1, keepdims=True) / np.maximum(active_bins.sum(axis=1, keepdims=True), 1)
        ss_res  = (active_bins * (y - pred_m) ** 2).sum(axis=1)
        ss_tot  = (active_bins * (y - mean_y) ** 2).sum(axis=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            r2 = np.where(ss_tot > 1e-20, 1.0 - ss_res / ss_tot, np.nan)

        finite = np.isfinite(r2)
        if not finite.any():
            return dict(mse_norm=mse_norm, mse_denorm=mse_denorm, r2_valid=False)

        r2f = r2[finite]
        return dict(
            mse_norm    = mse_norm,
            mse_denorm  = mse_denorm,
            r2_valid    = True,
            r2_mean     = float(r2f.mean()),
            r2_median   = float(np.median(r2f)),
            r2_std      = float(r2f.std()),
            r2_p10      = float(np.percentile(r2f, 10)),
            r2_p25      = float(np.percentile(r2f, 25)),
            r2_p75      = float(np.percentile(r2f, 75)),
            r2_p90      = float(np.percentile(r2f, 90)),
            r2_neg_frac = float((r2f < 0).mean()),
        )

    def run(
        self,
        tomogram_path : Path,
        height_range  : Tuple[float, float],
    ) -> np.ndarray:
        
        (
            tomogram_mmap, H, Az, R,
            height_axis, height_ax_j,
            sigma_lower_j, sigma_upper_j,
            n_params_out,
            threshold_factor, truncation_index,
        ) = self._prepare_data(tomogram_path, height_range)

        self._warmup_kernel(height_ax_j, H, sigma_lower_j, sigma_upper_j)

        output = np.zeros((n_params_out, Az, R), dtype=np.float32)

        total_attempted = self._run_fitting(
            tomogram_mmap, height_axis, height_ax_j,
            sigma_lower_j, sigma_upper_j,
            threshold_factor, truncation_index,
            Az, R, H, n_params_out, output,
        )

        self.logger.section("[Results]")
        self.logger.subsection(f"Active pixels fitted : {total_attempted:,} / {R * Az:,}")
        stats = self._estimate_r2(output, tomogram_mmap, height_axis, threshold_factor, truncation_index)

        self.logger.subsection(f"MSE (normalised)     : {stats['mse_norm']:.5f}")
        self.logger.subsection(f"MSE (denormalised)   : {stats['mse_denorm']:.5f}")
        if stats['r2_valid']:
            self.logger.subsection(f"R²  mean             : {stats['r2_mean']:.4f}")
            self.logger.subsection(f"R²  median           : {stats['r2_median']:.4f}")
            self.logger.subsection(f"R²  std              : {stats['r2_std']:.4f}")
            self.logger.subsection(f"R²  p10 / p25        : {stats['r2_p10']:.4f}  /  {stats['r2_p25']:.4f}")
            self.logger.subsection(f"R²  p75 / p90        : {stats['r2_p75']:.4f}  /  {stats['r2_p90']:.4f}")
            self.logger.subsection(f"R²  < 0  fraction    : {stats['r2_neg_frac']:.3f}")
        else:
            self.logger.subsection("R²                   : N/A")

        return output
