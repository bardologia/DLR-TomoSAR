from __future__ import annotations

import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses        import dataclass
from pathlib            import Path
from typing             import Dict, Optional, Tuple

import numpy as np

import jax
import jax.numpy as jnp

from configuration.param_extraction import FitMode
from tools.data.preprocessing       import ProfilePreprocessor
from tools.monitoring.logger        import Logger

from .initialiser import PeakInitialiser
from .selection   import BestKSelector, KernelBackendSelector


@dataclass
class PreparedBatch:
    r_end        : int
    n_total      : int
    n_active     : int
    active_idx   : np.ndarray
    prof_norm    : np.ndarray
    scale        : np.ndarray
    inits        : Optional[dict]
    load_seconds : float
    init_seconds : float


class SigmaFittingExtractor:

    MAX_PENDING_SCORINGS = 2

    def __init__(
        self,
        logger              : Logger,
        modes               : list,
        lambda_values       : list,
        k_max               : int                 = 5,
        threshold_factor    : float               = 0.25,
        truncation_index    : int                 = 170,
        prominence_frac     : float               = 0.05,
        sigma_init_divisor  : float               = 1.0,
        activity_threshold  : float               = 1e-3,
        range_batch_size    : int                 = 256,
        adam_steps          : int                 = 2000,
        adam_lr             : float               = 1e-2,
        adam_b1             : float               = 0.9,
        adam_b2             : float               = 0.999,
        gpu_pixel_batch_size: int                 = 8192,
        init_workers        : Optional[int]       = None,
        peak_initialiser    : Optional[PeakInitialiser] = None,
        kernel_backend      : Optional[tuple]     = None,
    ) -> None:

        self.logger               = logger
        self.modes                = list(modes)
        self.lambda_values        = [float(lambda_k) for lambda_k in lambda_values]
        self.k_max                = k_max
        self.threshold_factor     = threshold_factor
        self.truncation_index     = truncation_index
        self.prominence_frac      = prominence_frac
        self.sigma_init_divisor   = sigma_init_divisor
        self.activity_threshold   = activity_threshold
        self.range_batch_size     = range_batch_size
        self.adam_steps           = adam_steps
        self.adam_lr              = adam_lr
        self.adam_b1              = adam_b1
        self.adam_b2              = adam_b2
        self.gpu_pixel_batch_size = gpu_pixel_batch_size
        self._init_workers        = min(32, os.cpu_count() or 8) if init_workers is None else init_workers

        self.mode_masks = {}
        for mode in self.modes:
            fit_sigma, fit_amplitude, fit_mean = FitMode.free_flags(mode)
            self.mode_masks[mode] = (
                jnp.float32(1.0 if fit_amplitude else 0.0),
                jnp.float32(1.0 if fit_mean      else 0.0),
                jnp.float32(1.0 if fit_sigma     else 0.0),
            )

        if peak_initialiser is not None:
            self._peak_initialiser = peak_initialiser
            self._owns_initialiser = False
        else:
            self._peak_initialiser = PeakInitialiser(n_workers=self._init_workers)
            self._owns_initialiser = True

        self._best_k_selector = BestKSelector(k_max=k_max, logger=logger)

        if kernel_backend is not None:
            kernel, n_devices, backend, active_devices = kernel_backend
            self._owns_kernel                          = False
        else:
            kernel, n_devices, backend, active_devices = KernelBackendSelector().select()
            self._owns_kernel                          = True

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
        self.logger.subsection(f"lambda_values      : {self.lambda_values}")
        self.logger.subsection(f"fit modes          : {self.modes}")
        self.logger.subsection(f"sigma_init_divisor : {sigma_init_divisor}")
        self.logger.subsection(f"activity_threshold : {self.activity_threshold}")
        self.logger.subsection(f"init_workers       : {self._init_workers}")
        self.logger.subsection(f"n_devices          : {self._n_devices}")

    def _prepare_data(
        self,
        tomogram_path : Path,
        height_range  : Tuple[float, float],
    ) -> tuple:

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

        self.logger.section("[Kernel Compilation]")
        self.logger.subsection(f"Compiling JAX kernel (modes: {', '.join(self.modes)}) for K={self.k_max}")

        B = self.gpu_pixel_batch_size
        K = self.k_max

        dummy_s = jnp.ones((B, K),  dtype=jnp.float32) * 5.0
        dummy_p = jnp.ones((B, H),  dtype=jnp.float32)
        dummy_a = jnp.ones((B, K),  dtype=jnp.float32) * 0.5
        dummy_m = jnp.zeros((B, K), dtype=jnp.float32)

        amp_mask, mu_mask, sigma_mask = self.mode_masks[self.modes[0]]
        self._kernel(dummy_a, dummy_m, dummy_s, height_ax_j, dummy_p, amp_mask, mu_mask, sigma_mask, mu_lower_j, mu_upper_j, sigma_lower_j, sigma_upper_j, self.adam_steps, self.adam_lr, self.adam_b1, self.adam_b2)

        self.logger.subsection(f"Kernel compiled (K={K}, batch={B}, n_steps={self.adam_steps})")

    def _load_batch(
        self,
        tomogram_mmap : np.ndarray,
        r_start       : int,
        R             : int,
        Az            : int,
        H             : int,
        height_axis   : np.ndarray,
    ) -> PreparedBatch:

        t_load = time.monotonic()

        r_end = min(r_start + self.range_batch_size, R)
        r_c   = r_end - r_start
        raw   = np.abs(np.array(tomogram_mmap[:, :, r_start:r_end])).astype(np.float32)
        raw   = ProfilePreprocessor.apply(raw, self.threshold_factor, self.truncation_index)

        pf     = raw.transpose(2, 1, 0).reshape(r_c * Az, H).copy()
        active = pf.max(axis=1) > self.activity_threshold
        pmax   = pf.max(axis=1, keepdims=True)
        scale  = np.where(active[:, None], pmax, 1.0).astype(np.float32)

        del raw

        n_total    = pf.shape[0]
        active_idx = np.where(active)[0]
        n_active   = len(active_idx)

        prof_raw_all  = pf[active_idx]
        scale_all     = scale[active_idx, 0]
        prof_norm_all = prof_raw_all / scale_all[:, None]

        del pf, active, scale, pmax

        load_seconds = time.monotonic() - t_load
        t_init       = time.monotonic()

        inits = None
        if n_active > 0:
            amps_km, mus_km, sigs_km = self._peak_initialiser.run(prof_raw_all, height_axis, self.k_max, self.prominence_frac, self.sigma_init_divisor)
            inits = {K: (amps_km[:, :K].copy(), mus_km[:, :K].copy(), sigs_km[:, :K].copy()) for K in range(1, self.k_max + 1)}

        del prof_raw_all

        return PreparedBatch(
            r_end        = r_end,
            n_total      = n_total,
            n_active     = n_active,
            active_idx   = active_idx,
            prof_norm    = prof_norm_all.astype(np.float32, copy=False),
            scale        = scale_all,
            inits        = inits,
            load_seconds = load_seconds,
            init_seconds = time.monotonic() - t_init,
        )

    @staticmethod
    def _pad_rows(arr : np.ndarray, target : int) -> np.ndarray:
        pad = target - arr.shape[0]
        if pad == 0:
            return np.ascontiguousarray(arr, dtype=np.float32)
        return np.concatenate([arr.astype(np.float32, copy=False), np.zeros((pad, arr.shape[1]), dtype=np.float32)], axis=0)

    @staticmethod
    def _materialize_chunk(pending : tuple, final_amps : np.ndarray, final_mus : np.ndarray, final_sigs : np.ndarray) -> None:
        out_a, out_m, out_s, i_start, i_end = pending
        n_chunk = i_end - i_start

        final_amps[i_start:i_end] = np.array(out_a[:n_chunk], dtype=np.float32)
        final_mus [i_start:i_end] = np.array(out_m[:n_chunk], dtype=np.float32)
        final_sigs[i_start:i_end] = np.array(out_s[:n_chunk], dtype=np.float32)

    def _fit_all_K(
        self,
        prepared      : PreparedBatch,
        height_ax_j   : jnp.ndarray,
        sigma_lower_j : jnp.ndarray,
        sigma_upper_j : jnp.ndarray,
        mu_lower_j    : jnp.ndarray,
        mu_upper_j    : jnp.ndarray,
        mode          : str,
        batch_tag     : str,
    ) -> dict:

        gpu_results                   = {}
        B                             = self.gpu_pixel_batch_size
        N_act                         = prepared.n_active
        amp_mask, mu_mask, sigma_mask = self.mode_masks[mode]

        self.logger.section(f"[{batch_tag} | Phase 2 — GPU Fitting (mode {mode})]")

        for K in range(1, self.k_max + 1):
            amps_raw, mus, sigs_init = prepared.inits[K]
            amps_norm  = amps_raw / prepared.scale[:, None]
            final_amps = np.empty((N_act, K), dtype=np.float32)
            final_mus  = np.empty((N_act, K), dtype=np.float32)
            final_sigs = np.empty((N_act, K), dtype=np.float32)

            pending = None
            for i_start in range(0, N_act, B):
                i_end = min(i_start + B, N_act)
                out_a, out_m, out_s = self._kernel(
                    jnp.array(self._pad_rows(amps_norm         [i_start:i_end], B)),
                    jnp.array(self._pad_rows(mus               [i_start:i_end], B)),
                    jnp.array(self._pad_rows(sigs_init         [i_start:i_end], B)),
                    height_ax_j,
                    jnp.array(self._pad_rows(prepared.prof_norm[i_start:i_end], B)),
                    amp_mask,
                    mu_mask,
                    sigma_mask,
                    mu_lower_j,
                    mu_upper_j,
                    sigma_lower_j,
                    sigma_upper_j,
                    self.adam_steps,
                    self.adam_lr,
                    self.adam_b1,
                    self.adam_b2,
                )

                if pending is not None:
                    self._materialize_chunk(pending, final_amps, final_mus, final_sigs)
                pending = (out_a, out_m, out_s, i_start, i_end)

            if pending is not None:
                self._materialize_chunk(pending, final_amps, final_mus, final_sigs)

            gpu_results[K] = (final_amps, final_mus, final_sigs)
            self.logger.subsection(f"K={K} done")

        return gpu_results

    def _score_and_store(
        self,
        mode         : str,
        gpu_results  : dict,
        prepared     : PreparedBatch,
        height_axis  : np.ndarray,
        n_params_out : int,
        outputs      : Dict[tuple, dict],
        r_start      : int,
        Az           : int,
        batch_tag    : str,
    ) -> None:

        r_count = prepared.r_end - r_start
        mse_all = self._best_k_selector.score(gpu_results, prepared.prof_norm, height_axis, batch_tag=f"{batch_tag} | mode {mode}")

        for lambda_k in self.lambda_values:
            best_params, mse_act, pen_act, best_idx_act = self._best_k_selector.select(gpu_results, mse_all, prepared.scale, lambda_k, n_params_out, batch_tag=f"{batch_tag} | mode {mode}")

            output     = np.zeros((prepared.n_total, n_params_out),       dtype=np.float32)
            mse_out    = np.full ((prepared.n_total, self.k_max), np.nan, dtype=np.float32)
            pen_out    = np.full ((prepared.n_total, self.k_max), np.nan, dtype=np.float32)
            best_k_out = np.zeros(prepared.n_total,                       dtype=np.int16)

            output    [prepared.active_idx] = best_params
            mse_out   [prepared.active_idx] = mse_act
            pen_out   [prepared.active_idx] = pen_act
            best_k_out[prepared.active_idx] = best_idx_act + 1

            maps = outputs[(mode, lambda_k)]
            maps["params"][:, :, r_start:prepared.r_end] = output.reshape(r_count, Az, n_params_out).transpose(2, 1, 0)
            maps["mse"   ][:, :, r_start:prepared.r_end] = mse_out.reshape(r_count, Az, self.k_max).transpose(2, 1, 0)
            maps["pen"   ][:, :, r_start:prepared.r_end] = pen_out.reshape(r_count, Az, self.k_max).transpose(2, 1, 0)
            maps["best_k"][:,    r_start:prepared.r_end] = best_k_out.reshape(r_count, Az).T

    @staticmethod
    def _throttle_scorings(futures : list, max_pending : int) -> None:
        pending = [future for future in futures if not future.done()]
        while len(pending) >= max_pending:
            pending[0].result()
            pending = [future for future in futures if not future.done()]

    def _run_fitting(
        self,
        tomogram_mmap : np.ndarray,
        height_axis   : np.ndarray,
        height_ax_j   : jnp.ndarray,
        sigma_lower_j : jnp.ndarray,
        sigma_upper_j : jnp.ndarray,
        mu_lower_j    : jnp.ndarray,
        mu_upper_j    : jnp.ndarray,
        Az            : int,
        R             : int,
        H             : int,
        n_params_out  : int,
        outputs       : Dict[tuple, dict],
    ) -> int:

        n_batches = -(-R // self.range_batch_size)

        self.logger.section("[Range Bin Loading]")
        self.logger.subsection("Pipelined range batches: load + init prefetched, scoring in background, GPU fed continuously")
        self.logger.subsection(f"Range batches : {n_batches} x {self.range_batch_size} bins")

        total_attempted = 0

        with self.logger.track(transient=True) as progress:
            bar_task = progress.add_task("  [section]Processing range bins[/section]", total=R,)

            with ThreadPoolExecutor(max_workers=1) as load_pool, ThreadPoolExecutor(max_workers=1) as score_pool:
                r               = 0
                batch_index     = 0
                score_futures   = []
                prefetch_future = load_pool.submit(self._load_batch, tomogram_mmap, r, R, Az, H, height_axis)

                try:
                    while r < R:
                        prepared = prefetch_future.result()

                        r_start = r
                        batch_index += 1
                        batch_tag    = f"Batch {batch_index}/{n_batches}"

                        if prepared.r_end < R:
                            prefetch_future = load_pool.submit(self._load_batch, tomogram_mmap, prepared.r_end, R, Az, H, height_axis)

                        total_attempted += prepared.n_active

                        self.logger.section(f"[{batch_tag} | Phase 1 — Load + CPU Initialisation (prefetched)]")
                        self.logger.subsection(f"Active pixels : {prepared.n_active} / {prepared.n_total}")
                        self.logger.subsection(f"Load          : {prepared.load_seconds:.1f}s")
                        self.logger.subsection(f"Init          : {prepared.init_seconds:.1f}s ({self._init_workers} workers, shared across {len(self.modes)} modes)")

                        if prepared.n_active > 0:
                            for mode in self.modes:
                                t_fit       = time.monotonic()
                                gpu_results = self._fit_all_K(prepared, height_ax_j, sigma_lower_j, sigma_upper_j, mu_lower_j, mu_upper_j, mode, batch_tag)

                                self.logger.subsection(f"Mode {mode} fit : {time.monotonic() - t_fit:.1f}s")

                                self._throttle_scorings(score_futures, self.MAX_PENDING_SCORINGS)
                                score_futures.append(score_pool.submit(self._score_and_store, mode, gpu_results, prepared, height_axis, n_params_out, outputs, r_start, Az, batch_tag))

                        self.logger.subsection(f"{batch_tag} dispatched — range bins {r_start}-{prepared.r_end} of {R}")

                        progress.advance(bar_task, advance=prepared.r_end - r_start)
                        r = prepared.r_end

                    for future in score_futures:
                        future.result()

                except Exception:
                    prefetch_future.cancel()
                    for future in score_futures:
                        future.cancel()
                    raise

        gc.collect()

        self.logger.subsection(f"Total pixels   : {R * Az:,}")
        self.logger.subsection(f"Active pixels  : {total_attempted:,}")

        return total_attempted

    def run(
        self,
        tomogram_path : Path,
        height_range  : Tuple[float, float],
    ) -> Dict[tuple, Tuple[np.ndarray, Dict[str, np.ndarray]]]:

        (
            tomogram_mmap, H, Az, R,
            height_axis, height_ax_j,
            sigma_lower_j, sigma_upper_j,
            mu_lower_j, mu_upper_j,
            n_params_out,
        ) = self._prepare_data(tomogram_path, height_range)

        self._warmup_kernel(height_ax_j, H, sigma_lower_j, sigma_upper_j, mu_lower_j, mu_upper_j)

        outputs = {}
        for mode in self.modes:
            for lambda_k in self.lambda_values:
                outputs[(mode, lambda_k)] = {
                    "params" : np.zeros((n_params_out, Az, R),        dtype=np.float32),
                    "mse"    : np.full ((self.k_max, Az, R), np.nan,  dtype=np.float32),
                    "pen"    : np.full ((self.k_max, Az, R), np.nan,  dtype=np.float32),
                    "best_k" : np.zeros((Az, R),                      dtype=np.int16),
                }

        try:
            total_attempted = self._run_fitting(
                tomogram_mmap, height_axis, height_ax_j,
                sigma_lower_j, sigma_upper_j,
                mu_lower_j, mu_upper_j,
                Az, R, H, n_params_out, outputs,
            )
        finally:
            if self._owns_initialiser:
                self._peak_initialiser.close()
            if self._owns_kernel:
                jax.clear_caches()
            gc.collect()

        self.logger.section("[Results]")
        self.logger.subsection(f"Active pixels fitted : {total_attempted:,} / {R * Az:,}")
        self.logger.subsection(f"Permutations fitted  : {len(outputs)} ({len(self.modes)} modes x {len(self.lambda_values)} lambdas)")

        results = {}
        for key, maps in outputs.items():
            mode, lambda_k = key
            diagnostics    = {
                "mse_per_k"       : maps["mse"],
                "penalised_per_k" : maps["pen"],
                "best_k_map"      : maps["best_k"],
                "lambda_k"        : np.float32(lambda_k),
            }
            results[key] = (maps["params"], diagnostics)

        return results
