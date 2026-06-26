from __future__ import annotations

import gc
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib            import Path
from typing             import Dict, Optional, Tuple

import numpy as np

import jax
import jax.numpy as jnp

from configuration.param_extraction import FitSettings
from tools.data.preprocessing       import ProfilePreprocessor
from tools.monitoring.logger        import Logger

from .initialiser import PeakInitialiser
from .selection   import BestKSelector, KernelBackendSelector


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
        self.activity_threshold   = fit_settings.fit_config.activity_threshold
        self.fit_amplitude        = bool(fit_settings.fit_config.fit_amplitude)
        self.fit_mean             = bool(fit_settings.fit_config.fit_mean)
        self.amp_mask             = jnp.float32(1.0 if self.fit_amplitude else 0.0)
        self.mu_mask              = jnp.float32(1.0 if self.fit_mean      else 0.0)
        self._init_workers        = 80 if init_workers is None else init_workers

        self._peak_initialiser = PeakInitialiser(n_workers=self._init_workers)
        self._best_k_selector  = BestKSelector(k_max=k_max, lambda_k=lambda_k, logger=logger)

        kernel, n_devices, backend, active_devices = KernelBackendSelector().select()

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
        self.logger.subsection(f"activity_threshold : {self.activity_threshold}")
        self.logger.subsection(f"fit_amplitude      : {self.fit_amplitude}")
        self.logger.subsection(f"fit_mean           : {self.fit_mean}")
        self.logger.subsection(f"init_workers       : {self._init_workers}")
        self.logger.subsection(f"n_devices          : {self._n_devices}")

    def _prepare_data(
        self,
        tomogram_path : Path,
        height_range  : Tuple[float, float],
    ) -> tuple:

        fit_cfg      = self.fit_settings.fit_config
        n_params_out = 3 * self.k_max

        tomogram_mmap = np.load(str(tomogram_path), mmap_mode="r", allow_pickle=False)
        H, Az, R      = tomogram_mmap.shape
        height_axis = np.linspace(height_range[0], height_range[1], H, dtype=np.float32)
        height_ax_j = jnp.array(height_axis)

        h_span = float(height_axis[-1] - height_axis[0])
        dh     = h_span / (H - 1)

        sigma_lower_j = jnp.float32(dh)
        sigma_upper_j = jnp.float32(h_span / 2.0)

        mu_lower_j = jnp.float32(height_axis[0])
        mu_upper_j = jnp.float32(height_axis[-1])

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
            mu_lower_j, mu_upper_j,
            n_params_out,
            threshold_factor, truncation_index,
        )

    def _warmup_kernel(
        self,
        height_ax_j   : jnp.ndarray,
        H             : int,
        sigma_lower_j : jnp.ndarray,
        sigma_upper_j : jnp.ndarray,
        mu_lower_j    : jnp.ndarray,
        mu_upper_j    : jnp.ndarray,
    ) -> None:

        free = "+".join(self.fit_settings.free_parameters)

        self.logger.section("[Kernel Compilation]")
        self.logger.subsection(f"Compiling JAX kernel (free: {free}) for K={self.k_max}")

        N_warm = self._n_devices * max(1, 4 // self._n_devices)
        K      = self.k_max

        dummy_s = jnp.ones((N_warm, K),  dtype=jnp.float32) * 5.0
        dummy_p = jnp.ones((N_warm, H),  dtype=jnp.float32)
        dummy_a = jnp.ones((N_warm, K),  dtype=jnp.float32) * 0.5
        dummy_m = jnp.zeros((N_warm, K), dtype=jnp.float32)

        self._kernel(dummy_a, dummy_m, dummy_s, height_ax_j, dummy_p, self.amp_mask, self.mu_mask, mu_lower_j, mu_upper_j, sigma_lower_j, sigma_upper_j, 2, self.adam_lr, self.adam_b1, self.adam_b2)

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
        active = pf.max(axis=1) > self.activity_threshold
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
        mu_lower_j    : jnp.ndarray,
        mu_upper_j    : jnp.ndarray,
        N_act         : int,
        batch_tag     : str,
    ) -> dict:

        gpu_results = {}
        B           = self.gpu_pixel_batch_size

        self.logger.section(f"[{batch_tag} | Phase 2 — GPU Fitting ({'+'.join(self.fit_settings.free_parameters)})]")

        for K in range(1, self.k_max + 1):
            amps_raw, mus, sigs_init = inits[K]
            amps_norm  = amps_raw / scale_all[:, None]
            final_amps = np.empty((N_act, K), dtype=np.float32)
            final_mus  = np.empty((N_act, K), dtype=np.float32)
            final_sigs = np.empty((N_act, K), dtype=np.float32)

            for i_start in range(0, N_act, B):
                i_end   = min(i_start + B, N_act)
                n_chunk = i_end - i_start
                out_a, out_m, out_s = self._kernel(
                    jnp.array(self._pad_rows(amps_norm    [i_start:i_end], B)),
                    jnp.array(self._pad_rows(mus          [i_start:i_end], B)),
                    jnp.array(self._pad_rows(sigs_init    [i_start:i_end], B)),
                    height_ax_j,
                    jnp.array(self._pad_rows(prof_norm_all[i_start:i_end], B)),
                    self.amp_mask,
                    self.mu_mask,
                    mu_lower_j,
                    mu_upper_j,
                    sigma_lower_j,
                    sigma_upper_j,
                    self.adam_steps,
                    self.adam_lr,
                    self.adam_b1,
                    self.adam_b2,
                )

                final_amps[i_start:i_end] = np.array(out_a[:n_chunk], dtype=np.float32)
                final_mus [i_start:i_end] = np.array(out_m[:n_chunk], dtype=np.float32)
                final_sigs[i_start:i_end] = np.array(out_s[:n_chunk], dtype=np.float32)
                del out_a, out_m, out_s

            gpu_results[K] = (final_amps, final_mus, final_sigs)
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
        mu_lower_j    : jnp.ndarray,
        mu_upper_j    : jnp.ndarray,
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

        gpu_results = self._fit_all_K(inits, prof_norm_all, scale_all, height_ax_j, sigma_lower_j, sigma_upper_j, mu_lower_j, mu_upper_j, N_act, batch_tag)

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
        mu_lower_j       : jnp.ndarray,
        mu_upper_j       : jnp.ndarray,
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

                        r_start = r
                        r_count = r_end - r
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
                            mu_lower_j, mu_upper_j,
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
            mu_lower_j, mu_upper_j,
            n_params_out,
            threshold_factor, truncation_index,
        ) = self._prepare_data(tomogram_path, height_range)

        self._warmup_kernel(height_ax_j, H, sigma_lower_j, sigma_upper_j, mu_lower_j, mu_upper_j)

        output         = np.zeros((n_params_out, Az, R),        dtype=np.float32)
        mse_maps       = np.full ((self.k_max, Az, R), np.nan,  dtype=np.float32)
        penalised_maps = np.full ((self.k_max, Az, R), np.nan,  dtype=np.float32)
        best_k_map     = np.zeros((Az, R),                      dtype=np.int16)

        try:
            total_attempted = self._run_fitting(
                tomogram_mmap, height_axis, height_ax_j,
                sigma_lower_j, sigma_upper_j,
                mu_lower_j, mu_upper_j,
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
