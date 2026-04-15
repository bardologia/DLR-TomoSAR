from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, SymLogNorm
import multiprocessing as mp
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tomo_pixel_profiles_ref import (
    GaussianModel,
    ProfileFitter,
    ProfileConfig,
    ProfileParallelConfig,
    ProfileFitResult,
    TomogramLoader,
    SliceReconstructor as ProfileReconstructor,
    ParameterStorage,
)


plt.rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":     "dejavuserif",
    "font.size":            11,
    "axes.titlesize":       12,
    "axes.labelsize":       11,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "legend.fontsize":      9,
    "axes.linewidth":       0.8,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "xtick.top":            True,
    "ytick.right":          True,
    "xtick.minor.visible":  True,
    "ytick.minor.visible":  True,
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
})


@dataclass
class ParallelConfig:
    enabled:    bool = True
    n_workers:  Optional[int] = None
    method:     str = "fork"
    chunksize:  Optional[int] = None

    @property
    def effective_workers(self) -> int:
        if not self.enabled:
            return 1
        if self.n_workers is not None:
            return max(1, int(self.n_workers))
        return max(1, (os.cpu_count() or 1) - 1)

    def capped_workers(self, max_tasks: int) -> int:
        return min(self.effective_workers, max_tasks)

    def auto_chunksize(self, n_tasks: int) -> int:
        if self.chunksize is not None:
            return max(1, self.chunksize)
        n_workers = self.capped_workers(n_tasks)
        return max(1, n_tasks // (n_workers * 4))


@dataclass
class TomoConfig:
    tomo_file:            str
    output_dir:           Path
    height_axis_range:    Tuple[float, float]
    n_gauss_range:        List[int]
    range_index:          int            = 500
    n_example_profiles:   int            = 6
    n_height_samples:     int            = 40
    save_params:         bool            = False
    sar_gif_fps:          int            = 10
    sar_gif_n_frames:     int            = 150
    sar_gif_cache_limit_mb: int          = 512
    base_colors:          list           = field(default_factory=lambda: [cm.tab10(i) for i in range(10)])
    parallel:             ParallelConfig = field(default_factory=ParallelConfig)

    @property
    def tomo_name(self) -> str:
        return Path(self.tomo_file).stem


@dataclass
class FitResult:
    params:       np.ndarray
    fit_success:  np.ndarray
    recon:        np.ndarray
    rmse:         float
    mae:          float


@dataclass
class SarMetrics:
    mae:            float = np.nan
    rmse:           float = np.nan
    cross_entropy:  float = np.nan


_worker_tomo_abs       = None
_worker_height_axis    = None
_worker_height_indices = None
_worker_n_gaussians    = None
_worker_fitter         = None
_worker_reconstructor  = None


def _init_sar_worker(tomo_abs, height_axis, height_indices, n_gaussians):
    global _worker_tomo_abs, _worker_height_axis, _worker_height_indices, _worker_n_gaussians
    global _worker_fitter, _worker_reconstructor
    _worker_tomo_abs       = tomo_abs
    _worker_height_axis    = height_axis
    _worker_height_indices = height_indices
    _worker_n_gaussians    = n_gaussians
    _worker_fitter         = ProfileFitter(ProfileConfig(
        tomo_file="", output_dir=Path("."),
        height_axis_range=(float(height_axis[0]), float(height_axis[-1])),
        n_gaussians=n_gaussians,
        parallel=ProfileParallelConfig(enabled=False),
    ))
    _worker_reconstructor  = ProfileReconstructor()


def _sar_slice_worker(range_idx):
    slide_abs  = _worker_tomo_abs[:, :, range_idx]
    fit_result = _worker_fitter.fit_all_profiles(slide_abs, _worker_height_axis, _worker_n_gaussians, verbose=False)
    recon      = _worker_reconstructor.reconstruct(slide_abs, _worker_height_axis, fit_result, verbose=False)
    return range_idx, recon, fit_result.params, fit_result.fit_success


class GaussianFitter:

    def __init__(self, config: TomoConfig):
        self.config        = config
        self.model         = GaussianModel()
        self.reconstructor = ProfileReconstructor()

    def _build_profile_config(self, n_gaussians: int) -> ProfileConfig:
        return ProfileConfig(
            tomo_file         = self.config.tomo_file,
            output_dir        = self.config.output_dir,
            height_axis_range = self.config.height_axis_range,
            n_gaussians       = n_gaussians,
            range_index       = self.config.range_index,
            base_colors       = self.config.base_colors,
            parallel          = ProfileParallelConfig(
                enabled   = self.config.parallel.enabled,
                n_workers = self.config.parallel.n_workers,
                method    = self.config.parallel.method,
                chunksize = self.config.parallel.chunksize,
            ),
        )

    def fit_single_slice(self, slide_abs: np.ndarray, height_axis: np.ndarray, n_gaussians: int, verbose: bool = True) -> FitResult:
        start_time                  = time()
        profile_fitter              = ProfileFitter(self._build_profile_config(n_gaussians))
        profile_result              = profile_fitter.fit_all_profiles(slide_abs, height_axis, n_gaussians, verbose=verbose)
        reconstruction              = self.reconstructor.reconstruct(slide_abs, height_axis, profile_result, verbose=verbose)
        residual                    = slide_abs - reconstruction
        converged_cols              = profile_result.fit_success
        if converged_cols.any():
            valid_residual          = residual[:, converged_cols]
            root_mean_squared_error = np.sqrt(np.mean(valid_residual ** 2))
            mean_absolute_error     = np.mean(np.abs(valid_residual))
        else:
            root_mean_squared_error = np.nan
            mean_absolute_error     = np.nan
        elapsed                     = time() - start_time
        converged_count             = profile_result.fit_success.sum()
        total_azimuth               = slide_abs.shape[1]

        if verbose:
            print(
                f"  Done in {elapsed:.1f} s — converged {converged_count}/{total_azimuth} "
                f"({100 * converged_count / total_azimuth:.1f} %)  "
                f"RMSE = {root_mean_squared_error:.6f}  MAE = {mean_absolute_error:.6f}"
            )

        return FitResult(
            params      = profile_result.params,
            fit_success = profile_result.fit_success,
            recon       = reconstruction,
            rmse        = root_mean_squared_error,
            mae         = mean_absolute_error,
        )

    def fit_all_orders(self, slide_abs: np.ndarray, height_axis: np.ndarray) -> Dict[int, FitResult]:
        results = {}
        for n_gaussians in self.config.n_gauss_range:
            print(f"\n  Fitting N_g = {n_gaussians}")
            result = self.fit_single_slice(slide_abs, height_axis, n_gaussians)
            results[n_gaussians] = result

            if self.config.save_params:
                profile_fit_result = ProfileFitResult(
                    params      = result.params,
                    fit_success = result.fit_success,
                    n_gaussians = n_gaussians,
                )
                ParameterStorage.save(self._build_profile_config(n_gaussians), profile_fit_result, height_axis)

        return results

    def fit_profiles_quiet(self, slide_abs: np.ndarray, height_axis: np.ndarray, n_gaussians: int) -> ProfileFitResult:
        profile_fitter = ProfileFitter(self._build_profile_config(n_gaussians))
        return profile_fitter.fit_all_profiles(slide_abs, height_axis, n_gaussians, verbose=False)

    def pick_example_pixels(self, slide_abs: np.ndarray, results: Dict[int, FitResult], fallback_ng: int = 2, min_variance_percentile: float = 30.0, max_variance_percentile: float = 80.0) -> np.ndarray:
        first_key    = next(iter(results))
        n_azimuth    = results[first_key].params.shape[0]
        all_ok       = np.ones(n_azimuth, dtype=bool)

        for n_gaussians in self.config.n_gauss_range:
            all_ok &= results[n_gaussians].fit_success

        ok_indices = np.where(all_ok)[0]
        if len(ok_indices) < self.config.n_example_profiles:
            ok_indices = np.where(results[fallback_ng].fit_success)[0]

        if len(ok_indices) == 0:
            return np.array([], dtype=int)

        variance_scores = np.array([np.var(slide_abs[:, pixel]) for pixel in ok_indices])
        finite_mask = np.isfinite(variance_scores)
        ok_indices = ok_indices[finite_mask]
        variance_scores = variance_scores[finite_mask]

        if len(ok_indices) == 0:
            return np.array([], dtype=int)

        lo = np.nanpercentile(variance_scores, min_variance_percentile)
        hi = np.nanpercentile(variance_scores, max_variance_percentile)
        keep_mask = (variance_scores >= lo) & (variance_scores <= hi)
        if np.count_nonzero(keep_mask) >= 1:
            ok_indices = ok_indices[keep_mask]
            variance_scores = variance_scores[keep_mask]

        if len(ok_indices) == 0:
            return np.array([], dtype=int)

        order_simple_to_complex = np.argsort(variance_scores)
        n_examples = min(self.config.n_example_profiles, len(order_simple_to_complex))
        sample_positions = np.linspace(0, len(order_simple_to_complex) - 1, n_examples, dtype=int)
        return ok_indices[order_simple_to_complex[sample_positions]]


class SarReconstructor:
    def __init__(self, config: TomoConfig):
        self.config = config

    @staticmethod
    def height_average_sar_image(tomo_abs: np.ndarray, n_height_samples: int = 40) -> Tuple[np.ndarray, np.ndarray]:
        n_height   = tomo_abs.shape[0]
        n_samples  = min(n_height_samples, n_height)
        h_indices  = np.linspace(0, n_height - 1, n_samples, dtype=int)
        sar_image  = np.mean(tomo_abs[h_indices, :, :], axis=0)
        return sar_image, h_indices

    @staticmethod
    def image_cross_entropy(reference: np.ndarray, estimate: np.ndarray, epsilon: float = 1e-12) -> float:
        ref_clipped   = np.clip(np.asarray(reference, dtype=np.float64), 0.0, None)
        est_clipped   = np.clip(np.asarray(estimate, dtype=np.float64), 0.0, None)
        ref_total     = ref_clipped.sum()
        est_total     = est_clipped.sum()

        if ref_total <= 0 or est_total <= 0:
            return np.nan

        probability   = ref_clipped.ravel() / ref_total
        distribution  = est_clipped.ravel() / est_total
        return -np.sum(probability * np.log(distribution + epsilon))

    @staticmethod
    def _print_progress(prefix: str, current: int, total: int, bar_length: int = 36):
        fraction   = current / total if total > 0 else 1.0
        filled     = int(bar_length * fraction)
        bar        = "█" * filled + "·" * (bar_length - filled)
        print(f"\r{prefix} [{bar}] {current}/{total} ({100.0 * fraction:5.1f}%)", end="", flush=True)
        if current >= total:
            print()

    def _reconstruct_tomo_serial(self, tomo_abs, height_axis, n_gaussians, n_range, progress_prefix):
        profile_config = ProfileConfig(
            tomo_file         = "", output_dir=Path("."),
            height_axis_range = (float(height_axis[0]), float(height_axis[-1])),
            n_gaussians       = n_gaussians,
            parallel          = ProfileParallelConfig(enabled=False),
        )
        
        profile_fitter  = ProfileFitter(profile_config)
        reconstructor   = ProfileReconstructor()
        n_height        = tomo_abs.shape[0]
        n_azimuth       = tomo_abs.shape[1]
        n_params        = 3 * n_gaussians
        recon_tomo      = np.zeros((n_height, n_azimuth, n_range), dtype=np.float64)
        all_params      = np.full((n_range, n_azimuth, n_params), np.nan)
        all_success     = np.zeros((n_range, n_azimuth), dtype=bool)

        for iteration, range_idx in enumerate(range(n_range), start=1):
            slide_abs              = tomo_abs[:, :, range_idx]
            fit_result             = profile_fitter.fit_all_profiles(slide_abs, height_axis, n_gaussians, verbose=False)
            recon_tomo[:, :, range_idx] = reconstructor.reconstruct(slide_abs, height_axis, fit_result, verbose=False)
            all_params[range_idx]  = fit_result.params
            all_success[range_idx] = fit_result.fit_success
            self._print_progress(progress_prefix, iteration, n_range)

        return recon_tomo, all_params, all_success

    def _reconstruct_tomo_parallel(self, tomo_abs, height_axis, height_indices, n_gaussians, n_range, n_workers_eff, progress_prefix):
        n_height    = tomo_abs.shape[0]
        n_azimuth   = tomo_abs.shape[1]
        n_params    = 3 * n_gaussians
        recon_tomo  = np.zeros((n_height, n_azimuth, n_range), dtype=np.float64)
        all_params  = np.full((n_range, n_azimuth, n_params), np.nan)
        all_success = np.zeros((n_range, n_azimuth), dtype=bool)
        chunksize   = self.config.parallel.auto_chunksize(n_range)
        context     = mp.get_context(self.config.parallel.method)

        if progress_prefix:
            print(f"    chunksize={chunksize}")

        with ProcessPoolExecutor(max_workers=n_workers_eff, mp_context=context, initializer=_init_sar_worker, initargs=(tomo_abs, height_axis, height_indices, n_gaussians),) as executor:
            mapped = executor.map(_sar_slice_worker, range(n_range), chunksize=chunksize)
            for iteration, (range_idx, recon, params, success) in enumerate(mapped, start=1):
                recon_tomo[:, :, range_idx] = recon
                all_params[range_idx]  = params
                all_success[range_idx] = success
                self._print_progress(progress_prefix, iteration, n_range)

        return recon_tomo, all_params, all_success

    def compute_pixel_error_maps(self, reference: np.ndarray, reconstructed: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        difference     = reference - reconstructed
        absolute_error = np.abs(difference)
        squared_error  = difference ** 2

        ref_total   = np.sum(np.clip(reference, 0.0, None))
        recon_total = np.sum(np.clip(reconstructed, 0.0, None))

        if ref_total > 0 and recon_total > 0:
            prob_map     = np.clip(reference, 0.0, None) / ref_total
            dist_map     = np.clip(reconstructed, 0.0, None) / recon_total
            ce_map       = -prob_map * np.log(dist_map + 1e-12)
        else:
            ce_map = np.full_like(reference, np.nan, dtype=np.float64)

        return absolute_error, squared_error, ce_map

    @staticmethod
    def compute_profile_nrmse_map(reference_tomo: np.ndarray, reconstructed_tomo: np.ndarray) -> np.ndarray:
        difference            = reference_tomo - reconstructed_tomo
        rmse_map              = np.sqrt(np.mean(difference ** 2, axis=0))
        dynamic_range         = np.max(reference_tomo, axis=0) - np.min(reference_tomo, axis=0)
        nrmse_map             = np.full_like(rmse_map, np.nan, dtype=np.float64)
        valid_mask            = dynamic_range > 0
        nrmse_map[valid_mask] = rmse_map[valid_mask] / dynamic_range[valid_mask]
        return nrmse_map

    @staticmethod
    def compute_heightwise_metrics(reference_tomo: np.ndarray, reconstructed_tomo: np.ndarray, epsilon: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        difference      = np.asarray(reference_tomo, dtype=np.float64) - np.asarray(reconstructed_tomo, dtype=np.float64)
        mae_per_height  = np.mean(np.abs(difference), axis=(1, 2))
        rmse_per_height = np.sqrt(np.mean(difference ** 2, axis=(1, 2)))

        ref_pos    = np.clip(np.asarray(reference_tomo, dtype=np.float64), 0.0, None)
        rec_pos    = np.clip(np.asarray(reconstructed_tomo, dtype=np.float64), 0.0, None)
        ref_totals = np.sum(ref_pos, axis=(1, 2))
        rec_totals = np.sum(rec_pos, axis=(1, 2))

        ce_per_height = np.full(reference_tomo.shape[0], np.nan, dtype=np.float64)
        valid_heights = (ref_totals > 0) & (rec_totals > 0)
        
        for h_idx in np.where(valid_heights)[0]:
            prob                 = ref_pos[h_idx] / ref_totals[h_idx]
            dist                 = rec_pos[h_idx] / rec_totals[h_idx]
            ce_per_height[h_idx] = -np.sum(prob * np.log(dist + epsilon))

        return mae_per_height, rmse_per_height, ce_per_height

    def evaluate_sar_reconstruction(self) -> Tuple[Dict[int, SarMetrics], np.ndarray, Dict[int, np.ndarray]]:
        print("\nEvaluating SAR-image reconstruction across Gaussian orders ...")

        with h5py.File(self.config.tomo_file, "r") as h5_file:
            tomo_abs = np.abs(h5_file["tomogram"][:])

        n_height, n_azimuth, n_range = tomo_abs.shape
        height_axis = np.linspace(self.config.height_axis_range[0], self.config.height_axis_range[1], n_height)

        sar_reference, height_indices = self.height_average_sar_image(tomo_abs, n_height_samples=self.config.n_height_samples)

        recon_sar_images    = {}
        nrmse_maps          = {}
        abs_error_maps      = {}
        squared_error_maps  = {}
        ce_contrib_maps     = {}
        heightwise_metrics  = {}
        recon_tomos_for_gif = {}
        all_params_for_gif  = {}
        all_success_for_gif = {}
        sar_metrics         = {ng: SarMetrics() for ng in self.config.n_gauss_range}

        for n_gaussians in self.config.n_gauss_range:
            print(f"  Reconstructing SAR image for N_g = {n_gaussians} ...")

            n_workers_eff   = self.config.parallel.capped_workers(n_range)
            progress_prefix = f"    N_g={n_gaussians}"
            print(f"    Using {n_workers_eff} worker(s) for range-slice fitting")

            if n_workers_eff == 1:
                recon_tomo, all_params, all_success = self._reconstruct_tomo_serial(tomo_abs, height_axis, n_gaussians, n_range, progress_prefix)
            else:
                recon_tomo, all_params, all_success = self._reconstruct_tomo_parallel(tomo_abs, height_axis, height_indices, n_gaussians, n_range, n_workers_eff, progress_prefix)

            sar_reconstructed = np.mean(recon_tomo[height_indices, :, :], axis=0)

            nrmse_map = self.compute_profile_nrmse_map(tomo_abs, recon_tomo)
            nrmse_maps[n_gaussians] = nrmse_map

            mae_per_height, rmse_per_height, ce_per_height = self.compute_heightwise_metrics(tomo_abs, recon_tomo)
            heightwise_metrics[n_gaussians] = {
                "mae": mae_per_height,
                "rmse": rmse_per_height,
                "cross_entropy": ce_per_height,
            }

            if self.config.sar_gif_fps > 0:
                recon_tomos_for_gif[n_gaussians] = recon_tomo
                all_params_for_gif[n_gaussians]  = all_params
                all_success_for_gif[n_gaussians] = all_success
            else:
                del recon_tomo, all_params, all_success

            absolute_error, squared_error, ce_map = self.compute_pixel_error_maps(sar_reference, sar_reconstructed)

            abs_error_maps[n_gaussians]     = absolute_error
            squared_error_maps[n_gaussians] = squared_error
            ce_contrib_maps[n_gaussians]    = ce_map

            sar_metrics[n_gaussians].mae           = np.mean(absolute_error)
            sar_metrics[n_gaussians].rmse          = np.sqrt(np.mean(squared_error))
            sar_metrics[n_gaussians].cross_entropy = self.image_cross_entropy(sar_reference, sar_reconstructed)
            recon_sar_images[n_gaussians]          = sar_reconstructed

            print(
                f"    MAE = {sar_metrics[n_gaussians].mae:.6f}  "
                f"RMSE = {sar_metrics[n_gaussians].rmse:.6f}  "
                f"Cross-entropy = {sar_metrics[n_gaussians].cross_entropy:.6f}  "
                f"Mean profile NRMSE = {np.nanmean(nrmse_map):.6f}"
            )

        if self.config.sar_gif_fps > 0:
            gif_gen = SarHeightSweepGifGenerator(self.config)
            gif_gen.generate_dual_gifs(height_axis, tomo_abs, recon_tomos_for_gif, all_params_for_gif, all_success_for_gif, heightwise_metrics)
            for recon_tomo in recon_tomos_for_gif.values():
                del recon_tomo
            del recon_tomos_for_gif, all_params_for_gif, all_success_for_gif

        height_sweep_dir = self.config.output_dir / "sar_height_sweep"
        height_sweep_dir.mkdir(parents=True, exist_ok=True)
        plotter_height = TomoPlotter(self.config)
        plotter_height.plot_sar_height_metrics(height_axis, heightwise_metrics, height_sweep_dir)

        self._save_sar_plots(sar_reference, recon_sar_images, sar_metrics, nrmse_maps, abs_error_maps, squared_error_maps, ce_contrib_maps, n_azimuth, n_range)
        return sar_metrics, sar_reference, recon_sar_images

    def _save_sar_plots(self, sar_reference, recon_sar_images, sar_metrics, nrmse_maps, abs_error_maps, squared_error_maps, ce_contrib_maps, n_azimuth, n_range):
        save_dir = self.config.output_dir / "sar_reconstruction"
        save_dir.mkdir(parents=True, exist_ok=True)
        plotter = TomoPlotter(self.config)
        plotter.plot_sar_overview(sar_reference, recon_sar_images, n_azimuth, n_range, save_dir)
        plotter.plot_sar_metrics(sar_metrics, save_dir)
        plotter.plot_sar_error_maps(nrmse_maps, abs_error_maps, squared_error_maps, ce_contrib_maps, n_azimuth, n_range, save_dir)


class SarHeightSweepGifGenerator:
    def __init__(self, config: TomoConfig):
        self.config = config

    @staticmethod
    def _prepare_height_eval_tensors(all_params: np.ndarray, all_success: np.ndarray, n_gaussians: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        params = np.asarray(all_params, dtype=np.float32).reshape(all_params.shape[0], all_params.shape[1], n_gaussians, 3)
        mu    = params[..., 0]
        sigma = params[..., 1]
        amp   = params[..., 2]
        sigma_safe = np.where(sigma > 1e-12, sigma, 1.0).astype(np.float32, copy=False)
        valid_mask = np.asarray(all_success.T, dtype=bool)
        return mu, sigma_safe, amp, valid_mask

    @staticmethod
    def _evaluate_at_height_from_tensors(h: float, mu: np.ndarray, sigma_safe: np.ndarray, amp: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        exponent = -((np.float32(h) - mu) ** 2) / (2.0 * sigma_safe ** 2)
        image = np.sum(amp * np.exp(exponent), axis=2, dtype=np.float32)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        image = image.T
        image[~valid_mask] = 0.0
        return image

    @classmethod
    def _evaluate_at_height(cls, h: float, all_params: np.ndarray, all_success: np.ndarray, n_gaussians: int) -> np.ndarray:
        mu, sigma_safe, amp, valid_mask = cls._prepare_height_eval_tensors(all_params, all_success, n_gaussians)
        return cls._evaluate_at_height_from_tensors(h, mu, sigma_safe, amp, valid_mask)

    @staticmethod
    def _image_cross_entropy(reference: np.ndarray, estimate: np.ndarray, epsilon: float = 1e-12) -> float:
        ref_clipped = np.clip(np.asarray(reference, dtype=np.float64), 0.0, None)
        est_clipped = np.clip(np.asarray(estimate, dtype=np.float64), 0.0, None)
        ref_total   = ref_clipped.sum()
        est_total   = est_clipped.sum()

        if ref_total <= 0 or est_total <= 0:
            return np.nan

        probability  = ref_clipped.ravel() / ref_total
        distribution = est_clipped.ravel() / est_total
        return -np.sum(probability * np.log(distribution + epsilon))

    @staticmethod
    def _pixel_metric_maps(reference: np.ndarray, estimate: np.ndarray, epsilon: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        difference = np.asarray(reference, dtype=np.float64) - np.asarray(estimate, dtype=np.float64)
        mae_map    = np.abs(difference)
        rmse_map   = mae_map

        ref_pos   = np.clip(np.asarray(reference, dtype=np.float64), 0.0, None)
        est_pos   = np.clip(np.asarray(estimate, dtype=np.float64), 0.0, None)
        ref_total = ref_pos.sum()
        est_total = est_pos.sum()
        if ref_total > 0 and est_total > 0:
            prob_map = ref_pos / ref_total
            dist_map = est_pos / est_total
            ce_map   = -prob_map * np.log(dist_map + epsilon)
        else:
            ce_map = np.full_like(mae_map, np.nan, dtype=np.float64)

        return difference, mae_map, rmse_map, ce_map

    def generate_dual_gifs(self, height_axis: np.ndarray, tomo_abs: np.ndarray, recon_tomos_by_ng: Dict[int, np.ndarray], all_params_by_ng: Dict[int, np.ndarray], all_success_by_ng: Dict[int, np.ndarray], heightwise_metrics: Dict[int, Dict[str, np.ndarray]]) -> None:
        print("  Generating dual SAR height-sweep GIFs...")
        n_height, n_azimuth, n_range = tomo_abs.shape
        fps      = max(1, self.config.sar_gif_fps)
        n_frames = min(self.config.sar_gif_n_frames, n_height)
        gauss_list = sorted(recon_tomos_by_ng.keys())

        frame_h_indices = np.linspace(0, n_height - 1, n_frames, dtype=int)
        frame_heights   = height_axis[frame_h_indices]

        self._generate_slices_gif(height_axis, tomo_abs, recon_tomos_by_ng, gauss_list, frame_h_indices, frame_heights, fps, n_azimuth, n_range)
        self._generate_metrics_gif(recon_tomos_by_ng, all_params_by_ng, all_success_by_ng, gauss_list, frame_h_indices, frame_heights, fps, n_azimuth, n_range)
        self._generate_curves_image(heightwise_metrics, height_axis, gauss_list)

    def _generate_slices_gif(self, height_axis: np.ndarray, tomo_abs: np.ndarray, recon_tomos_by_ng: Dict[int, np.ndarray], gauss_list: List[int], frame_h_indices: np.ndarray, frame_heights: np.ndarray, fps: int, n_azimuth: int, n_range: int) -> None:
        print("    Generating slices GIF (2 rows × N Gaussians)...")
        n_frames = len(frame_h_indices)
        n_cols = len(gauss_list)

        sample_idx   = frame_h_indices[:: max(1, len(frame_h_indices) // 10)]
        sample_block = tomo_abs[sample_idx, :, :]
        vmax = float(np.nanpercentile(sample_block, 99.5))
        vmax = max(vmax, 1e-12)
        del sample_block

        diff_samples = []
        for h_idx in sample_idx:
            for ng in gauss_list[:min(3, len(gauss_list))]:
                ref_frame  = tomo_abs[h_idx, :, :]
                rec_frame  = recon_tomos_by_ng[ng][h_idx, :, :]
                diff_frame = ref_frame - rec_frame
                finite = np.abs(diff_frame[np.isfinite(diff_frame)])
                if finite.size > 0:
                    diff_samples.append(finite)
        if diff_samples:
            diff_abs_concat = np.concatenate(diff_samples)
            diff_max = max(float(np.nanpercentile(diff_abs_concat, 99.5)), 1e-8)
        else:
            diff_max = 1.0

        save_dir = self.config.output_dir / "sar_height_sweep"
        save_dir.mkdir(parents=True, exist_ok=True)
        gif_path = save_dir / f"{self.config.tomo_name}_sar_height_sweep_slices.gif"

        figure, axes = plt.subplots(2, n_cols, figsize=(4.0 * n_cols, 8.0), constrained_layout=True, squeeze=False)

        h0 = frame_heights[0]
        h_idx0 = frame_h_indices[0]

        im_recons = []
        im_diffs = []
        for col, ng in enumerate(gauss_list):
            ref_frame = tomo_abs[h_idx0, :, :]
            rec_frame = recon_tomos_by_ng[ng][h_idx0, :, :]
            diff_frame = ref_frame - rec_frame

            im_r = axes[0, col].imshow(rec_frame, origin="lower", cmap="jet", aspect="auto", vmin=0, vmax=vmax, extent=[0, n_range, 0, n_azimuth])
            axes[0, col].set_title(rf"Reconstruction ($N_g = {ng}$)")
            axes[0, col].set_ylabel("Azimuth (pixels)" if col == 0 else "")

            im_d = axes[1, col].imshow(diff_frame, origin="lower", cmap="RdBu_r", aspect="auto", vmin=-diff_max, vmax=diff_max, extent=[0, n_range, 0, n_azimuth])
            axes[1, col].set_title(rf"Difference ($N_g = {ng}$)")
            axes[1, col].set_xlabel("Range (pixels)")
            axes[1, col].set_ylabel("Azimuth (pixels)" if col == 0 else "")

            im_recons.append(im_r)
            im_diffs.append(im_d)

        cbar_r = figure.colorbar(im_recons[0], ax=axes[0, :], shrink=0.9, pad=0.02)
        cbar_r.set_label("Amplitude (a.u.)")
        cbar_d = figure.colorbar(im_diffs[0], ax=axes[1, :], shrink=0.9, pad=0.02)
        cbar_d.set_label("Difference (a.u.)")

        sup_title = figure.suptitle(f"Height = {h0:.1f} m", fontsize=13, fontweight="bold")

        def _update(frame_idx):
            h_idx = frame_h_indices[frame_idx]
            h = frame_heights[frame_idx]
            for col, ng in enumerate(gauss_list):
                ref_frame = tomo_abs[h_idx, :, :]
                rec_frame = recon_tomos_by_ng[ng][h_idx, :, :]
                im_recons[col].set_data(rec_frame)
                im_diffs[col].set_data(ref_frame - rec_frame)
            sup_title.set_text(f"Height = {h:.1f} m")
            return [sup_title, *im_recons, *im_diffs]

        anim = animation.FuncAnimation(figure, _update, frames=n_frames, interval=1000 // fps, blit=False)
        anim.save(str(gif_path), writer="pillow", fps=fps, dpi=100)
        plt.close(figure)
        print(f"      Slices GIF saved → {gif_path} ({n_frames} frames)")

    def _generate_metrics_gif(self, recon_tomos_by_ng: Dict[int, np.ndarray], all_params_by_ng: Dict[int, np.ndarray], all_success_by_ng: Dict[int, np.ndarray], gauss_list: List[int], frame_h_indices: np.ndarray, frame_heights: np.ndarray, fps: int, n_azimuth: int, n_range: int, tomo_abs: np.ndarray = None) -> None:
        print("    Generating metrics GIF (3 rows × N Gaussians)...")
        n_frames = len(frame_h_indices)
        n_cols = len(gauss_list)

        metric_samples_list = [[], [], []]
        for h_idx_sample in frame_h_indices[::max(1, n_frames // 10)]:
            for ng in gauss_list[:min(3, len(gauss_list))]:
                rec_frame = recon_tomos_by_ng[ng][h_idx_sample, :, :]
                diff_frame, mae_map, rmse_map, ce_map = self._pixel_metric_maps(rec_frame, rec_frame)
                for metric_map, samples_list in zip([mae_map, rmse_map, ce_map], metric_samples_list):
                    valid = metric_map[np.isfinite(metric_map) & (metric_map > 0)]
                    if valid.size > 0:
                        samples_list.append(valid)

        def _positive_bounds(sample_list):
            if not sample_list:
                return 1e-12, 1.0
            merged = np.concatenate(sample_list)
            vmin = max(float(np.nanpercentile(merged, 2.0)), 1e-12)
            vmax = max(float(np.nanpercentile(merged, 99.5)), vmin * 10.0)
            return vmin, vmax

        mae_vmin, mae_vmax = _positive_bounds(metric_samples_list[0])
        rmse_vmin, rmse_vmax = _positive_bounds(metric_samples_list[1])
        ce_vmin, ce_vmax = _positive_bounds(metric_samples_list[2])

        save_dir = self.config.output_dir / "sar_height_sweep"
        save_dir.mkdir(parents=True, exist_ok=True)
        gif_path = save_dir / f"{self.config.tomo_name}_sar_height_sweep_metrics.gif"

        figure, axes = plt.subplots(3, n_cols, figsize=(4.0 * n_cols, 10.0), constrained_layout=True, squeeze=False)
        extent = [0, n_range, 0, n_azimuth]

        h0 = frame_heights[0]
        h_idx0 = frame_h_indices[0]

        im_list = []
        bounds_list = [(mae_vmin, mae_vmax), (rmse_vmin, rmse_vmax), (ce_vmin, ce_vmax)]
        for row, (vmin, vmax) in enumerate(bounds_list):
            for col, ng in enumerate(gauss_list):
                rec_frame = recon_tomos_by_ng[ng][h_idx0, :, :]
                _, mae_m, rmse_m, ce_m = self._pixel_metric_maps(rec_frame, rec_frame)
                metric_maps = [mae_m, rmse_m, ce_m]
                im_data = np.clip(metric_maps[row], vmin, None)
                im = axes[row, col].imshow(im_data, origin="lower", cmap="viridis", aspect="auto", norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent)
                axes[row, col].set_title(rf"$N_g={ng}$")
                axes[row, col].set_ylabel("Azimuth (pixels)" if col == 0 else "")
                if row == 2:
                    axes[row, col].set_xlabel("Range (pixels)")
                im_list.append(im)

        sup_title = figure.suptitle(f"Height = {h0:.1f} m", fontsize=13, fontweight="bold")

        def _update(frame_idx):
            h_idx = frame_h_indices[frame_idx]
            h = frame_heights[frame_idx]
            im_idx = 0
            for row, (vmin, vmax) in enumerate(bounds_list):
                for col, ng in enumerate(gauss_list):
                    rec_frame = recon_tomos_by_ng[ng][h_idx, :, :]
                    _, mae_m, rmse_m, ce_m = self._pixel_metric_maps(rec_frame, rec_frame)
                    metric_maps = [mae_m, rmse_m, ce_m]
                    im_data = np.clip(metric_maps[row], vmin, None)
                    im_list[im_idx].set_data(im_data)
                    im_idx += 1
            sup_title.set_text(f"Height = {h:.1f} m")
            return [sup_title, *im_list]

        anim = animation.FuncAnimation(figure, _update, frames=n_frames, interval=1000 // fps, blit=False)
        anim.save(str(gif_path), writer="pillow", fps=fps, dpi=100)
        plt.close(figure)
        print(f"      Metrics GIF saved → {gif_path} ({n_frames} frames)")

    def _generate_curves_image(self, heightwise_metrics: Dict[int, Dict[str, np.ndarray]], height_axis: np.ndarray, gauss_list: List[int]) -> None:
        print("    Generating curves static image...")
        n_cols = len(gauss_list)

        save_dir = self.config.output_dir / "sar_height_sweep"
        save_dir.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(3, n_cols, figsize=(4.0 * n_cols, 10.0), constrained_layout=True, squeeze=False)

        row_specs = [("Average MAE", "mae", "tab:blue"), ("Average RMSE", "rmse", "tab:orange"), ("Cross-entropy", "cross_entropy", "tab:green")]

        for row, (row_title, metric_key, color) in enumerate(row_specs):
            all_values = []
            for ng in gauss_list:
                values = np.asarray(heightwise_metrics[ng][metric_key], dtype=np.float64)
                finite_values = values[np.isfinite(values)]
                if finite_values.size > 0:
                    all_values.append(finite_values)

            y_limits = None
            if all_values:
                merged = np.concatenate(all_values)
                y_min = float(np.nanmin(merged))
                y_max = float(np.nanmax(merged))
                if np.isclose(y_min, y_max):
                    pad = max(1e-8, 0.05 * max(abs(y_min), 1.0))
                    y_min -= pad
                    y_max += pad
                y_limits = (y_min, y_max)

            for col, ng in enumerate(gauss_list):
                axis = axes[row, col]
                values = np.asarray(heightwise_metrics[ng][metric_key], dtype=np.float64)
                axis.plot(height_axis, values, color=color, lw=1.6)
                axis.grid(True, alpha=0.25, linewidth=0.5)
                finite_mask = np.isfinite(values)
                if np.any(finite_mask):
                    best_index = int(np.nanargmin(values))
                    axis.scatter([height_axis[best_index]], [values[best_index]], color="black", s=22, zorder=3)
                if y_limits is not None:
                    axis.set_ylim(*y_limits)
                if row == 0:
                    axis.set_title(rf"$N_g = {ng}$")
                if col == 0:
                    axis.set_ylabel(row_title)
                else:
                    axis.set_ylabel("")
                if row == 2:
                    axis.set_xlabel("Height (m)")

        figure.suptitle("Height-wise SAR reconstruction metrics (static)")
        gif_path = save_dir / f"{self.config.tomo_name}_sar_height_sweep_curves.png"
        figure.savefig(gif_path, dpi=100)
        plt.close(figure)
        print(f"      Curves image saved → {gif_path}")

    def generate(self, height_axis: np.ndarray, tomo_abs: np.ndarray, all_params: np.ndarray, all_success: np.ndarray, n_gaussians: int) -> None:
        print(f"  Generating SAR height-sweep GIF for N_g = {n_gaussians} ...")
        n_height, n_azimuth, n_range = tomo_abs.shape
        fps      = max(1, self.config.sar_gif_fps)
        n_frames = min(self.config.sar_gif_n_frames, n_height)

        frame_h_indices = np.linspace(0, n_height - 1, n_frames, dtype=int)
        frame_heights   = height_axis[frame_h_indices]
        mu, sigma_safe, amp, valid_mask = self._prepare_height_eval_tensors(all_params, all_success, n_gaussians)

        sample_idx   = frame_h_indices[:: max(1, len(frame_h_indices) // 10)]
        sample_block = tomo_abs[sample_idx, :, :]
        vmax         = float(np.nanpercentile(sample_block, 99.5))
        vmax         = max(vmax, 1e-12)
        del sample_block

        frame_cache_limit_bytes = max(0, int(self.config.sar_gif_cache_limit_mb)) * 1024 * 1024
        frame_cache_bytes       = n_frames * n_azimuth * n_range * np.dtype(np.float32).itemsize
        use_frame_cache         = frame_cache_bytes <= frame_cache_limit_bytes
        recon_frames            = None
        
        if use_frame_cache:
            recon_frames = np.empty((n_frames, n_azimuth, n_range), dtype=np.float32)
            print(f"    Caching reconstructed GIF frames in memory ({frame_cache_bytes / 1024**2:.1f} MiB)")
        else:
            print(f"    Skipping GIF frame cache ({frame_cache_bytes / 1024**2:.1f} MiB needed > {self.config.sar_gif_cache_limit_mb} MiB limit)")

        mae_curve  = np.zeros(n_frames, dtype=np.float64)
        rmse_curve = np.zeros(n_frames, dtype=np.float64)
        ce_curve   = np.full(n_frames, np.nan, dtype=np.float64)
        diff_abs_samples = []
        mae_map_samples  = []
        rmse_map_samples = []
        ce_map_samples   = []

        sample_frame_positions = set(np.linspace(0, n_frames - 1, min(12, n_frames), dtype=int).tolist())

        for i, (h_idx, h_val) in enumerate(zip(frame_h_indices, frame_heights)):
            ref_frame = tomo_abs[h_idx, :, :]
            rec_frame = self._evaluate_at_height_from_tensors(h_val, mu, sigma_safe, amp, valid_mask)
            if recon_frames is not None:
                recon_frames[i] = rec_frame
            
            diff_frame, mae_map, rmse_map, ce_map = self._pixel_metric_maps(ref_frame, rec_frame)

            mae_curve[i]  = float(np.mean(np.abs(diff_frame)))
            rmse_curve[i] = float(np.sqrt(np.mean(diff_frame ** 2)))
            ce_curve[i]   = self._image_cross_entropy(ref_frame, rec_frame)

            if i in sample_frame_positions:
                finite_abs = np.abs(diff_frame[np.isfinite(diff_frame)])
                if finite_abs.size > 0:
                    diff_abs_samples.append(finite_abs)
                
                mae_valid  = mae_map[np.isfinite(mae_map) & (mae_map > 0)]
                rmse_valid = rmse_map[np.isfinite(rmse_map) & (rmse_map > 0)]
                ce_valid   = ce_map[np.isfinite(ce_map) & (ce_map > 0)]
                
                if mae_valid.size > 0:
                    mae_map_samples.append(mae_valid)
                if rmse_valid.size > 0:
                    rmse_map_samples.append(rmse_valid)
                if ce_valid.size > 0:
                    ce_map_samples.append(ce_valid)

        if diff_abs_samples:
            diff_abs_concat = np.concatenate(diff_abs_samples)
            diff_max        = max(float(np.nanpercentile(diff_abs_concat, 99.5)), 1e-8)
        else:
            diff_max = 1.0

        def _positive_bounds(sample_list):
            if not sample_list:
                return 1e-12, 1.0
            merged = np.concatenate(sample_list)
            vmin = max(float(np.nanpercentile(merged, 2.0)), 1e-12)
            vmax = max(float(np.nanpercentile(merged, 99.5)), vmin * 10.0)
            return vmin, vmax

        mae_vmin, mae_vmax   = _positive_bounds(mae_map_samples)
        rmse_vmin, rmse_vmax = _positive_bounds(rmse_map_samples)
        ce_vmin, ce_vmax     = _positive_bounds(ce_map_samples)

        save_dir = self.config.output_dir / "sar_height_sweep"
        save_dir.mkdir(parents=True, exist_ok=True)
        gif_path = save_dir / (f"{self.config.tomo_name}_sar_height_sweep_Ng{n_gaussians}.gif")

        figure, axes = plt.subplots(3, 3, figsize=(16, 13), constrained_layout=True, gridspec_kw={"height_ratios": [1.85, 1.85, 1.15]})
        ax_ref, ax_recon, ax_diff          = axes[0, 0], axes[0, 1], axes[0, 2]
        ax_mae_map, ax_rmse_map, ax_ce_map = axes[1, 0], axes[1, 1], axes[1, 2]
        ax_mae, ax_rmse, ax_ce             = axes[2, 0], axes[2, 1], axes[2, 2]
        extent                             = [0, n_range, 0, n_azimuth]

        h0        = frame_heights[0]
        h_idx0    = frame_h_indices[0]
        ref_frame = tomo_abs[h_idx0, :, :]
        rec_frame = recon_frames[0] if recon_frames is not None else self._evaluate_at_height_from_tensors(h0, mu, sigma_safe, amp, valid_mask)

        im_ref = ax_ref.imshow(ref_frame, origin="lower", cmap="jet", aspect="auto", vmin=0, vmax=vmax, extent=extent)
        ax_ref.set_title("Reference tomogram slice")
        ax_ref.set_xlabel("Range (pixels)")
        ax_ref.set_ylabel("Azimuth (pixels)")

        im_recon = ax_recon.imshow(rec_frame, origin="lower", cmap="jet", aspect="auto", vmin=0, vmax=vmax, extent=extent)
        ax_recon.set_title(rf"Gaussian reconstruction ($N_g = {n_gaussians}$)")
        ax_recon.set_xlabel("Range (pixels)")
        ax_recon.set_ylabel("Azimuth (pixels)")

        diff_frame, mae_map, rmse_map, ce_map = self._pixel_metric_maps(ref_frame, rec_frame)
        im_diff = ax_diff.imshow(diff_frame, origin="lower", cmap="RdBu_r", aspect="auto", vmin=-diff_max, vmax=diff_max, extent=extent)
        ax_diff.set_title("Difference (Reference - Reconstruction)")
        ax_diff.set_xlabel("Range (pixels)")
        ax_diff.set_ylabel("Azimuth (pixels)")

        im_mae_map = ax_mae_map.imshow(np.clip(mae_map, mae_vmin, None), origin="lower", cmap="inferno", aspect="auto", norm=LogNorm(vmin=mae_vmin, vmax=mae_vmax), extent=extent)
        ax_mae_map.set_title("Per-pixel MAE")
        ax_mae_map.set_xlabel("Range (pixels)")
        ax_mae_map.set_ylabel("Azimuth (pixels)")

        im_rmse_map = ax_rmse_map.imshow(np.clip(rmse_map, rmse_vmin, None), origin="lower", cmap="magma", aspect="auto", norm=LogNorm(vmin=rmse_vmin, vmax=rmse_vmax), extent=extent)
        ax_rmse_map.set_title("Per-pixel RMSE")
        ax_rmse_map.set_xlabel("Range (pixels)")
        ax_rmse_map.set_ylabel("Azimuth (pixels)")

        im_ce_map = ax_ce_map.imshow(np.where(np.isfinite(ce_map), np.clip(ce_map, ce_vmin, None), np.nan), origin="lower", cmap="cividis", aspect="auto", norm=LogNorm(vmin=ce_vmin, vmax=ce_vmax), extent=extent)
        ax_ce_map.set_title("Per-pixel Cross-entropy")
        ax_ce_map.set_xlabel("Range (pixels)")
        ax_ce_map.set_ylabel("Azimuth (pixels)")

        colorbar_amp = figure.colorbar(im_ref, ax=[ax_ref, ax_recon], shrink=0.84, pad=0.02)
        colorbar_amp.set_label("Amplitude (a.u.)")

        colorbar_diff = figure.colorbar(im_diff, ax=[ax_diff], shrink=0.84, pad=0.02)
        colorbar_diff.set_label("Difference (a.u.)")

        colorbar_mae = figure.colorbar(im_mae_map, ax=[ax_mae_map], shrink=0.84, pad=0.02)
        colorbar_mae.set_label("MAE (log scale)")

        colorbar_rmse = figure.colorbar(im_rmse_map, ax=[ax_rmse_map], shrink=0.84, pad=0.02)
        colorbar_rmse.set_label("RMSE (log scale)")

        colorbar_ce = figure.colorbar(im_ce_map, ax=[ax_ce_map], shrink=0.84, pad=0.02)
        colorbar_ce.set_label("Cross-entropy contribution (log scale)")

        metric_specs = [
            (ax_mae,  "Average MAE",   mae_curve,  "tab:blue"),
            (ax_rmse, "Average RMSE",  rmse_curve, "tab:orange"),
            (ax_ce,   "Cross-entropy", ce_curve,   "tab:green"),
        ]

        vlines  = []
        markers = []
        for axis, title, values, color in metric_specs:
            axis.plot(frame_heights, values, color=color, lw=1.6)
            axis.set_title(title)
            axis.set_xlabel("Height (m)")
            axis.set_ylabel(title)
            axis.grid(True, alpha=0.25, linewidth=0.5)

            finite_mask = np.isfinite(values)
            if np.any(finite_mask):
                y_min = float(np.nanmin(values[finite_mask]))
                y_max = float(np.nanmax(values[finite_mask]))
                if np.isclose(y_min, y_max):
                    pad = max(1e-8, 0.05 * max(abs(y_min), 1.0))
                    y_min -= pad
                    y_max += pad
                axis.set_ylim(y_min, y_max)

            current_vline = axis.axvline(h0, color="black", ls="--", lw=1.0)
            current_marker, = axis.plot([h0], [values[0]], marker="o", ms=5, color="black", linestyle="None")
            vlines.append(current_vline)
            markers.append(current_marker)

        sup_title = figure.suptitle(f"Height = {h0:.1f} m", fontsize=13, fontweight="bold")

        def _update(frame_idx):
            h_idx = frame_h_indices[frame_idx]
            h     = frame_heights[frame_idx]
            ref = tomo_abs[h_idx, :, :]
            im_ref.set_data(ref)
            rec = recon_frames[frame_idx] if recon_frames is not None else self._evaluate_at_height_from_tensors(h, mu, sigma_safe, amp, valid_mask)
            im_recon.set_data(rec)
            diff_map, mae_map_current, rmse_map_current, ce_map_current = self._pixel_metric_maps(ref, rec)
            im_diff.set_data(diff_map)
            im_mae_map.set_data(np.clip(mae_map_current, mae_vmin, None))
            im_rmse_map.set_data(np.clip(rmse_map_current, rmse_vmin, None))
            im_ce_map.set_data(np.where(np.isfinite(ce_map_current), np.clip(ce_map_current, ce_vmin, None), np.nan))

            for metric_idx, values in enumerate((mae_curve, rmse_curve, ce_curve)):
                vlines[metric_idx].set_xdata([h, h])
                markers[metric_idx].set_data([h], [values[frame_idx]])

            sup_title.set_text(f"Height = {h:.1f} m")
            return [im_ref, im_recon, im_diff, im_mae_map, im_rmse_map, im_ce_map, sup_title, *vlines, *markers]

        anim = animation.FuncAnimation(
            figure, _update, frames=n_frames,
            interval=1000 // fps, blit=False,
        )
        anim.save(str(gif_path), writer="pillow", fps=fps, dpi=100)
        plt.close(figure)
        print(
            f"    GIF saved → {gif_path}  "
            f"({n_frames} frames, {fps} fps)"
        )


class TomoPlotter:
    def __init__(self, config: TomoConfig):
        self.config = config
        self.model  = GaussianModel()

    @staticmethod
    def _robust_positive_bounds(arrays: List[np.ndarray], low_pct: float = 2.0, high_pct: float = 99.5, floor: float = 1e-12) -> Tuple[float, float]:
        finite_positive = []
        for arr in arrays:
            values = np.asarray(arr, dtype=np.float64)
            mask = np.isfinite(values) & (values > 0)
            if np.any(mask):
                finite_positive.append(values[mask])

        if not finite_positive:
            return floor, 1.0

        merged = np.concatenate(finite_positive)
        vmin = max(float(np.nanpercentile(merged, low_pct)), floor)
        vmax = max(float(np.nanpercentile(merged, high_pct)), vmin * 10.0)
        return vmin, vmax

    def plot_slice_comparison(self, slide_abs: np.ndarray, height_axis: np.ndarray, results: Dict[int, FitResult]) -> plt.Figure:
        print("\nPlotting slice comparison")
        n_azimuth         = slide_abs.shape[1]
        n_gauss_count     = len(self.config.n_gauss_range)
        amplitude_min     = np.nanmin(slide_abs)
        amplitude_max     = np.nanmax(slide_abs)
        extent_slice      = [0, n_azimuth, height_axis[0], height_axis[-1]]

        figure, axes = plt.subplots(2, 1 + n_gauss_count, figsize=(4.2 * (1 + n_gauss_count), 8), constrained_layout=True)

        image_original = axes[0, 0].imshow(slide_abs, origin="lower", cmap="jet", aspect="auto", vmin=amplitude_min, vmax=amplitude_max, extent=extent_slice)
        axes[0, 0].set_ylabel("Height (m)")
        colorbar_original = plt.colorbar(image_original, ax=axes[0, 0], shrink=0.75)
        colorbar_original.set_label("Amplitude (a.u.)")
        axes[1, 0].set_visible(False)

        residual_max = max(np.nanmax(np.abs(slide_abs - results[ng].recon)) for ng in self.config.n_gauss_range)

        for column, n_gaussians in enumerate(self.config.n_gauss_range, start=1):
            reconstruction = results[n_gaussians].recon
            residual       = slide_abs - reconstruction

            image_recon = axes[0, column].imshow(reconstruction, origin="lower", cmap="jet", aspect="auto", vmin=amplitude_min, vmax=amplitude_max, extent=extent_slice)
            axes[0, column].set_xlabel("Azimuth (pixels)")
            axes[0, column].set_ylabel("Height (m)")
            colorbar_recon = plt.colorbar(image_recon, ax=axes[0, column], shrink=0.75)
            colorbar_recon.set_label("Amplitude (a.u.)")

            image_residual = axes[1, column].imshow(residual, origin="lower", cmap="RdBu_r", aspect="auto", vmin=-residual_max, vmax=residual_max, extent=extent_slice)
            axes[1, column].set_xlabel("Azimuth (pixels)")
            axes[1, column].set_ylabel("Height (m)")
            colorbar_residual = plt.colorbar(image_residual, ax=axes[1, column], shrink=0.75)
            colorbar_residual.set_label("Residual (a.u.)")

        save_dir = self.config.output_dir / "slice_comparison"
        save_dir.mkdir(parents=True, exist_ok=True)
        ng_min = min(self.config.n_gauss_range)
        ng_max = max(self.config.n_gauss_range)
        figure.savefig(save_dir / f"{self.config.tomo_name}_slice_orig_vs_recon_rn{self.config.range_index}_Ng{ng_min}-{ng_max}.png")
        return figure

    def plot_peak_heights(self, results: Dict[int, FitResult]) -> plt.Figure:
        print("Plotting peak-height maps")
        n_gauss_count = len(self.config.n_gauss_range)
        n_azimuth     = results[next(iter(results))].params.shape[0]
        azimuth_axis  = np.arange(n_azimuth)

        mu_all = []
        for n_gaussians in self.config.n_gauss_range:
            valid_mask = results[n_gaussians].fit_success
            params_current = results[n_gaussians].params
            if np.any(valid_mask):
                mu_values = params_current[valid_mask, 0::3].ravel()
                mu_values = mu_values[np.isfinite(mu_values)]
                if mu_values.size:
                    mu_all.append(mu_values)

        if mu_all:
            mu_all = np.concatenate(mu_all)
            y_min = float(np.nanmin(mu_all))
            y_max = float(np.nanmax(mu_all))
            if np.isclose(y_min, y_max):
                pad = max(1e-6, 0.05 * max(abs(y_min), 1.0))
                y_min -= pad
                y_max += pad
        else:
            y_min, y_max = 0.0, 1.0

        figure, axes = plt.subplots(n_gauss_count, 1, figsize=(14, 3.5 * n_gauss_count), constrained_layout=True, sharex=True)
        if n_gauss_count == 1:
            axes = [axes]

        for row, n_gaussians in enumerate(self.config.n_gauss_range):
            axis           = axes[row]
            valid_mask     = results[n_gaussians].fit_success
            params_current = results[n_gaussians].params

            for peak_index in range(n_gaussians):
                mu_values = params_current[:, 3 * peak_index]
                axis.scatter(
                    azimuth_axis[valid_mask], mu_values[valid_mask],
                    s=4, alpha=0.55, marker="o", edgecolors="none",
                    color=self.config.base_colors[peak_index],
                    label=rf"$\mu_{{{peak_index + 1}}}$",
                )

            axis.set_ylabel("Height (m)")
            axis.set_ylim(y_min, y_max)
            axis.legend(markerscale=3, ncol=n_gaussians, loc="upper right", framealpha=0.85, edgecolor="gray")
            axis.grid(True, alpha=0.25, linewidth=0.5)

        axes[-1].set_xlabel("Azimuth (pixels)")
        save_dir = self.config.output_dir / "peak_heights"
        save_dir.mkdir(parents=True, exist_ok=True)
        ng_min = min(self.config.n_gauss_range)
        ng_max = max(self.config.n_gauss_range)
        figure.savefig(save_dir / f"{self.config.tomo_name}_scatterer_heights_rn{self.config.range_index}_Ng{ng_min}-{ng_max}.png")
        return figure

    def plot_example_fits(self, slide_abs: np.ndarray, height_axis: np.ndarray, results: Dict[int, FitResult], example_pixels: np.ndarray) -> plt.Figure:
        print("Plotting example fits")
        n_gauss_count   = len(self.config.n_gauss_range)
        n_examples      = len(example_pixels)
        height_fine     = np.linspace(height_axis[0], height_axis[-1], 500)

        if n_examples == 0:
            print("  No valid example pixels found — skipping example fits plot.")
            return plt.figure()

        figure, axes = plt.subplots(n_gauss_count, n_examples, figsize=(3.5 * n_examples, 3.2 * n_gauss_count), constrained_layout=True, sharex=True, squeeze=False)

        global_y_min = np.inf
        global_y_max = -np.inf
        for pixel in example_pixels:
            profile      = slide_abs[:, pixel]
            global_y_min = min(global_y_min, np.nanmin(profile))
            global_y_max = max(global_y_max, np.nanmax(profile))

        for row, n_gaussians in enumerate(self.config.n_gauss_range):
            params_current  = results[n_gaussians].params
            success_current = results[n_gaussians].fit_success

            for column, pixel in enumerate(example_pixels):
                axis    = axes[row, column]
                profile = slide_abs[:, pixel]
                axis.plot(height_axis, profile, "k-", lw=0.8, label="Observed")

                if success_current[pixel]:
                    fitted_curve = self.model.multi_gaussian(height_fine, *params_current[pixel])
                    axis.plot(height_fine, fitted_curve, color="tab:red", ls="-", lw=1.4, label="Fitted")
                    for peak_index in range(n_gaussians):
                        single_peak = self.model.single_gaussian(height_fine, *params_current[pixel, 3 * peak_index:3 * peak_index + 3])
                        axis.fill_between(height_fine, single_peak, alpha=0.18, color=self.config.base_colors[peak_index], label=rf"$g_{{{peak_index + 1}}}$")
                else:
                    axis.text(0.5, 0.5, "No convergence", transform=axis.transAxes, ha="center", va="center", fontsize=9, fontstyle="italic", color="red")

                axis.set_ylim(global_y_min, global_y_max)
                if column == 0:
                    axis.set_ylabel(rf"$N_g = {n_gaussians}$" + "\nAmplitude (a.u.)")
                if row == n_gauss_count - 1:
                    axis.set_xlabel("Height (m)")
                if row == 0:
                    axis.set_title(rf"Range = {self.config.range_index}, Azimuth = {pixel}")
                axis.grid(True, alpha=0.25, linewidth=0.5)
                if row == 0 and column == n_examples - 1:
                    axis.legend(fontsize=7, loc="upper right", framealpha=0.85, edgecolor="gray")

        save_dir = self.config.output_dir / "exemple_fits"
        save_dir.mkdir(parents=True, exist_ok=True)
        ng_min = min(self.config.n_gauss_range)
        ng_max = max(self.config.n_gauss_range)
        figure.savefig(save_dir / f"{self.config.tomo_name}_profile_fit_examples_rn{self.config.range_index}_Ng{ng_min}-{ng_max}.png")
        return figure

    def plot_fit_quality(self, slide_abs: np.ndarray, height_axis: np.ndarray, results: Dict[int, FitResult]) -> plt.Figure:
        print("Plotting per-pixel fit quality across Gaussian orders ...")
        n_height, n_azimuth = slide_abs.shape
        n_gauss_count       = len(self.config.n_gauss_range)
        azimuth_axis        = np.arange(n_azimuth)

        figure, axes = plt.subplots(
            n_gauss_count, 2, figsize=(16, 3.0 * n_gauss_count),
            constrained_layout=True,
            gridspec_kw={"width_ratios": [1, 3]},
        )
        if n_gauss_count == 1:
            axes = axes[np.newaxis, :]

        nrmse_per_order = {}
        for n_gaussians in self.config.n_gauss_range:
            params_current  = results[n_gaussians].params
            success_current = results[n_gaussians].fit_success
            nrmse_values    = np.full(n_azimuth, np.nan)

            for azimuth in range(n_azimuth):
                if success_current[azimuth]:
                    observed       = slide_abs[:, azimuth]
                    reconstructed  = self.model.multi_gaussian(height_axis, *params_current[azimuth])
                    value_range    = observed.max() - observed.min()
                    if value_range > 0:
                        nrmse_values[azimuth] = np.sqrt(np.mean((observed - reconstructed) ** 2)) / value_range

            nrmse_per_order[n_gaussians] = nrmse_values

        global_nrmse_max = np.nanmax([np.nanmax(values) for values in nrmse_per_order.values()]) * 1.05

        for row, n_gaussians in enumerate(self.config.n_gauss_range):
            success_current = results[n_gaussians].fit_success
            nrmse_values    = nrmse_per_order[n_gaussians]
            valid_mask      = success_current

            axis_convergence = axes[row, 0]
            convergence_colors = np.where(success_current, 0.2, 0.9)
            axis_convergence.barh(azimuth_axis, np.ones(n_azimuth), height=1.0, color=plt.cm.RdYlGn_r(convergence_colors), edgecolor="none")
            axis_convergence.set_ylim(0, n_azimuth)
            axis_convergence.set_xticks([])
            axis_convergence.set_ylabel("Azimuth (pixels)")
            axis_convergence.invert_xaxis()

            axis_error = axes[row, 1]
            axis_error.bar(azimuth_axis[valid_mask], nrmse_values[valid_mask], width=1.0, color="tab:blue", alpha=0.7, edgecolor="none")
            median_nrmse = np.nanmedian(nrmse_values[valid_mask]) if valid_mask.any() else 0.0
            axis_error.axhline(median_nrmse, color="tab:red", ls="--", lw=1.2, label=rf"Median NRMSE$\,={median_nrmse:.4f}$")
            axis_error.set_xlim(0, n_azimuth)
            axis_error.set_ylim(0, global_nrmse_max)
            axis_error.set_ylabel("NRMSE")
            axis_error.legend(fontsize=8, framealpha=0.85, edgecolor="gray")
            axis_error.grid(True, alpha=0.25, linewidth=0.5, axis="y")
            if row == n_gauss_count - 1:
                axis_error.set_xlabel("Azimuth (pixels)")

        save_dir = self.config.output_dir / "residual_metrics"
        save_dir.mkdir(parents=True, exist_ok=True)
        ng_min = min(self.config.n_gauss_range)
        ng_max = max(self.config.n_gauss_range)
        figure.savefig(save_dir / f"{self.config.tomo_name}_fit_quality_map_rn{self.config.range_index}_Ng{ng_min}-{ng_max}.png")
        return figure

    def plot_residual_metrics(self, results: Dict[int, FitResult]) -> plt.Figure:
        print("Plotting residual metrics")
        gauss_list = list(self.config.n_gauss_range)

        rmse_values = [results[ng].rmse for ng in gauss_list]
        mae_values  = [results[ng].mae for ng in gauss_list]

        rmse_baseline    = rmse_values[0]
        mae_baseline     = mae_values[0]
        rmse_improvement = [(rmse_baseline - value) / rmse_baseline * 100 for value in rmse_values]
        mae_improvement  = [(mae_baseline - value) / mae_baseline * 100 for value in mae_values]

        figure, (axis_error, axis_improvement) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        bar_positions = np.arange(len(gauss_list))
        bar_width     = 0.35

        axis_error.bar(bar_positions - bar_width / 2, rmse_values, bar_width, label="RMSE", color="tab:blue", edgecolor="black", linewidth=0.5)
        axis_error.bar(bar_positions + bar_width / 2, mae_values, bar_width, label="MAE", color="tab:orange", edgecolor="black", linewidth=0.5)
        axis_error.set_xticks(bar_positions)
        axis_error.set_xticklabels([rf"$N_g = {ng}$" for ng in gauss_list])
        axis_error.set_xlabel(r"Number of Gaussian components ($N_g$)")
        axis_error.set_ylabel("Error (a.u.)")
        axis_error.legend(framealpha=0.85, edgecolor="gray")
        axis_error.grid(True, alpha=0.25, linewidth=0.5, axis="y")

        axis_improvement.plot(gauss_list, rmse_improvement, "o-", color="tab:blue", lw=1.8, ms=6, label="RMSE improvement")
        axis_improvement.plot(gauss_list, mae_improvement, "s-", color="tab:orange", lw=1.8, ms=6, label="MAE improvement")
        axis_improvement.axhline(0, color="gray", ls="--", lw=0.8)
        axis_improvement.set_xticks(gauss_list)
        axis_improvement.set_xticklabels([rf"$N_g = {ng}$" for ng in gauss_list])
        axis_improvement.set_xlabel(r"Number of Gaussian components ($N_g$)")
        axis_improvement.set_ylabel(rf"Improvement over $N_g=1$ (%)")
        axis_improvement.legend(framealpha=0.85, edgecolor="gray")
        axis_improvement.grid(True, alpha=0.25, linewidth=0.5, axis="y")

        save_dir = self.config.output_dir / "residual_metrics"
        save_dir.mkdir(parents=True, exist_ok=True)
        ng_min = min(self.config.n_gauss_range)
        ng_max = max(self.config.n_gauss_range)
        figure.savefig(save_dir / f"{self.config.tomo_name}_reconstruction_error_rn{self.config.range_index}_Ng{ng_min}-{ng_max}.png")
        return figure

    def plot_sar_overview(self, sar_reference: np.ndarray, recon_sar_images: Dict[int, np.ndarray], n_azimuth: int, n_range: int, save_dir: Path) -> plt.Figure:
        gauss_list   = list(self.config.n_gauss_range)
        n_cols       = len(gauss_list) + 1
        figure, axes = plt.subplots(2, n_cols, figsize=(3.8 * n_cols, 8.4), constrained_layout=True, squeeze=False)

        all_sar_images     = [sar_reference] + [recon_sar_images[ng] for ng in gauss_list]
        sar_vmin, sar_vmax = self._robust_positive_bounds(all_sar_images, low_pct=1.0, high_pct=99.7)
        sar_norm           = LogNorm(vmin=sar_vmin, vmax=sar_vmax)
        extent             = [0, n_range, 0, n_azimuth]

        image_ref = axes[0, 0].imshow(np.clip(sar_reference, sar_vmin, None), origin="lower", cmap="jet", aspect="auto", norm=sar_norm, extent=extent)
        axes[0, 0].set_title("Reference SAR image (log)")
        axes[0, 0].set_xlabel("Range (pixels)")
        axes[0, 0].set_ylabel("Azimuth (pixels)")

        differences = [sar_reference - recon_sar_images[ng] for ng in gauss_list]
        diff_abs = np.concatenate([np.abs(d[np.isfinite(d)]) for d in differences if np.any(np.isfinite(d))])
        if diff_abs.size == 0:
            diff_max       = 1.0
            diff_linthresh = 1e-6
        else:
            diff_max       = max(float(np.nanpercentile(diff_abs, 99.5)), 1e-8)
            diff_linthresh = max(float(np.nanpercentile(diff_abs, 60.0)) * 0.15, 1e-8)
        diff_norm = SymLogNorm(linthresh=diff_linthresh, vmin=-diff_max, vmax=diff_max, base=10)

        axes[1, 0].axis("off")
    
        last_diff_image = None
        for column, n_gaussians in enumerate(gauss_list, start=1):
            axes[0, column].imshow(np.clip(recon_sar_images[n_gaussians], sar_vmin, None), origin="lower", cmap="jet", aspect="auto", norm=sar_norm, extent=extent)
            axes[0, column].set_title(rf"Reconstructed SAR (log), $N_g={n_gaussians}$")
            axes[0, column].set_xlabel("Range (pixels)")
            axes[0, column].set_ylabel("Azimuth (pixels)")

            diff_map = sar_reference - recon_sar_images[n_gaussians]
            last_diff_image = axes[1, column].imshow(diff_map, origin="lower", cmap="RdBu_r", aspect="auto", norm=diff_norm, extent=extent)
            axes[1, column].set_title(rf"Difference, $N_g={n_gaussians}$")
            axes[1, column].set_xlabel("Range (pixels)")
            axes[1, column].set_ylabel("Azimuth (pixels)")

        colorbar_amp = figure.colorbar(image_ref, ax=axes[0, :], shrink=0.82, pad=0.02)
        colorbar_amp.set_label("Amplitude (a.u., log scale)")

        colorbar_diff = figure.colorbar(last_diff_image, ax=axes[1, 1:], shrink=0.82, pad=0.02)
        colorbar_diff.set_label("Reference - Reconstruction (a.u., symlog)")

        ng_min = min(self.config.n_gauss_range)
        ng_max = max(self.config.n_gauss_range)
        figure.savefig(save_dir / f"{self.config.tomo_name}_sar_reconstruction_overview_Ng{ng_min}-{ng_max}.png")
        return figure

    def plot_sar_metrics(self, sar_metrics: Dict[int, SarMetrics], save_dir: Path) -> plt.Figure:
        gauss_list    = list(self.config.n_gauss_range)
        mae_values    = [sar_metrics[ng].mae for ng in gauss_list]
        rmse_values   = [sar_metrics[ng].rmse for ng in gauss_list]
        ce_values     = [sar_metrics[ng].cross_entropy for ng in gauss_list]

        figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

        metric_specs = [
            ("MAE",            mae_values,  "tab:blue"),
            ("RMSE",           rmse_values, "tab:orange"),
            ("Cross-entropy",  ce_values,   "tab:green"),
        ]

        for axis, (label, values, color) in zip(axes, metric_specs):
            axis.plot(gauss_list, values, marker="o", ms=6, lw=1.8, color=color)
            axis.set_xticks(gauss_list)
            axis.set_xlabel(r"Number of Gaussian components ($N_g$)")
            axis.set_ylabel(label)
            axis.set_title(label)
            axis.grid(True, alpha=0.25, linewidth=0.5)

            best_index = int(np.nanargmin(values))
            axis.scatter([gauss_list[best_index]], [values[best_index]], color="black", s=26, zorder=3)

        ng_min = min(self.config.n_gauss_range)
        ng_max = max(self.config.n_gauss_range)
        figure.savefig(save_dir / f"{self.config.tomo_name}_sar_reconstruction_metrics_Ng{ng_min}-{ng_max}.png")
        return figure

    def plot_sar_height_metrics(self, height_axis: np.ndarray, heightwise_metrics: Dict[int, Dict[str, np.ndarray]], save_dir: Path) -> plt.Figure:
        gauss_list = list(self.config.n_gauss_range)
        n_cols = len(gauss_list)

        figure, axes = plt.subplots(3, n_cols, figsize=(3.8 * n_cols, 10.5), constrained_layout=True, squeeze=False)

        row_specs = [
            ("Average MAE",   "mae",           "tab:blue"),
            ("Average RMSE",  "rmse",          "tab:orange"),
            ("Cross-entropy", "cross_entropy", "tab:green"),
        ]

        for row, (row_title, metric_key, color) in enumerate(row_specs):
            all_values = []
            for n_gaussians in gauss_list:
                values = np.asarray(heightwise_metrics[n_gaussians][metric_key], dtype=np.float64)
                finite_values = values[np.isfinite(values)]
                if finite_values.size > 0:
                    all_values.append(finite_values)

            y_limits = None
            if all_values:
                merged = np.concatenate(all_values)
                y_min = float(np.nanmin(merged))
                y_max = float(np.nanmax(merged))
                if np.isclose(y_min, y_max):
                    pad = max(1e-8, 0.05 * max(abs(y_min), 1.0))
                    y_min -= pad
                    y_max += pad
                y_limits = (y_min, y_max)

            for column, n_gaussians in enumerate(gauss_list):
                axis = axes[row, column]
                values = np.asarray(heightwise_metrics[n_gaussians][metric_key], dtype=np.float64)
                axis.plot(height_axis, values, color=color, lw=1.6)
                axis.grid(True, alpha=0.25, linewidth=0.5)

                finite_mask = np.isfinite(values)
                if np.any(finite_mask):
                    best_index = int(np.nanargmin(values))
                    axis.scatter([height_axis[best_index]], [values[best_index]], color="black", s=22, zorder=3)

                if y_limits is not None:
                    axis.set_ylim(*y_limits)

                if row == 0:
                    axis.set_title(rf"$N_g = {n_gaussians}$")
                if column == 0:
                    axis.set_ylabel(row_title)
                else:
                    axis.set_ylabel("")
                if row == len(row_specs) - 1:
                    axis.set_xlabel("Height (m)")
                else:
                    axis.set_xlabel("")

        figure.suptitle("Height-wise SAR reconstruction metrics across Gaussian orders")

        ng_min = min(gauss_list)
        ng_max = max(gauss_list)
        figure.savefig(save_dir / f"{self.config.tomo_name}_sar_height_metrics_Ng{ng_min}-{ng_max}.png")
        return figure

    def plot_sar_error_maps(self, nrmse_maps: dict, abs_error_maps: dict, squared_error_maps: dict, ce_contrib_maps: dict, n_azimuth: int, n_range: int, save_dir: Path) -> plt.Figure:
        gauss_list = list(self.config.n_gauss_range)
        extent     = [0, n_range, 0, n_azimuth]

        figure, axes = plt.subplots(4, len(gauss_list), figsize=(3.8 * len(gauss_list), 13.2), constrained_layout=True, squeeze=False)
        figure.subplots_adjust(left=0.12)

        nrmse_vmax = max(float(np.nanpercentile(nrmse_maps[ng][np.isfinite(nrmse_maps[ng])], 99.5)) if np.any(np.isfinite(nrmse_maps[ng])) else 1.0 for ng in gauss_list)
        nrmse_vmax = max(nrmse_vmax, 1e-8)
        abs_vmin, abs_vmax = self._robust_positive_bounds([abs_error_maps[ng]     for ng in gauss_list], low_pct=3.0, high_pct=99.5)
        sq_vmin, sq_vmax   = self._robust_positive_bounds([squared_error_maps[ng] for ng in gauss_list], low_pct=3.0, high_pct=99.5)
        ce_vmin, ce_vmax   = self._robust_positive_bounds([ce_contrib_maps[ng]    for ng in gauss_list], low_pct=3.0, high_pct=99.5)

        row_specs = [
            ("Profile NRMSE map",               nrmse_maps,         "viridis",  None,                                  "NRMSE",                               0.0, nrmse_vmax),
            ("Absolute error map",              abs_error_maps,     "inferno",  LogNorm(vmin=abs_vmin, vmax=abs_vmax), "Absolute error (log scale)", abs_vmin),
            ("Squared error map",               squared_error_maps, "magma",    LogNorm(vmin=sq_vmin,  vmax=sq_vmax),  "Squared error (log scale)", sq_vmin),
            ("Cross-entropy contribution map",  ce_contrib_maps,    "cividis",  LogNorm(vmin=ce_vmin,  vmax=ce_vmax),  "Cross-entropy contribution (log scale)", ce_vmin),
        ]

        row_images = []
        for row, row_spec in enumerate(row_specs):
            if row == 0:
                row_title, data_maps, colormap, row_norm, colorbar_label, vmin_row, vmax_row = row_spec
                row_floor = None
            else:
                row_title, data_maps, colormap, row_norm, colorbar_label, row_floor = row_spec
                vmin_row = None
                vmax_row = None

            last_image = None
            for column, n_gaussians in enumerate(gauss_list):
                axis = axes[row, column]
                plot_data = np.asarray(data_maps[n_gaussians], dtype=np.float64)
                if row == 0:
                    last_image = axis.imshow(plot_data, origin="lower", cmap=colormap, aspect="auto", vmin=vmin_row, vmax=vmax_row, extent=extent)
                else:
                    plot_data = np.where(np.isfinite(plot_data), np.clip(plot_data, row_floor, None), np.nan)
                    last_image = axis.imshow(plot_data, origin="lower", cmap=colormap, aspect="auto", norm=row_norm, extent=extent)
                if row == 0:
                    axis.set_title(rf"$N_g = {n_gaussians}$")
                if column == 0:
                    axis.set_ylabel("Azimuth (pixels)", labelpad=2)
                    axis.text(-0.30, 0.5, row_title, transform=axis.transAxes, rotation=90, ha="center", va="center", fontsize=10)
                else:
                    axis.set_ylabel("")
                if row == len(row_specs) - 1:
                    axis.set_xlabel("Range (pixels)")
                else:
                    axis.set_xlabel("")
            row_images.append((last_image, colorbar_label))

        for row, (image, colorbar_label) in enumerate(row_images):
            colorbar = figure.colorbar(image, ax=axes[row, :], shrink=0.86, pad=0.015)
            colorbar.set_label(colorbar_label)

        ng_min = min(self.config.n_gauss_range)
        ng_max = max(self.config.n_gauss_range)
        figure.savefig(save_dir / f"{self.config.tomo_name}_sar_reconstruction_pixel_metrics_Ng{ng_min}-{ng_max}.png")
        return figure


if __name__ == "__main__":

    config = TomoConfig(
        tomo_file          = "/ste/rnd/User/vice_vi/Pruebas/TOMO/TOMO-SR/tomograms/tomo_17sartom0102_Lhv_MSF_w10_20.hd5",
        output_dir         = Path("/ste/rnd/User/vice_vi/Pruebas/TOMO/TOMO-SR/"),
        height_axis_range  = (-20, 80),
        n_gauss_range      = list(range(1, 6)),
        range_index        = 500,
        n_example_profiles = 6,
        save_params       = False,
        parallel           = ParallelConfig(enabled=True, n_workers=64, method="fork"),
    )

    fitter  = GaussianFitter(config)
    plotter = TomoPlotter(config)

    slide_abs, height_axis = TomogramLoader.load_slice(config.tomo_file, config.range_index, config.height_axis_range)

    results        = fitter.fit_all_orders(slide_abs, height_axis)
    example_pixels = fitter.pick_example_pixels(slide_abs, results)

    plotter.plot_slice_comparison(slide_abs, height_axis, results)
    plotter.plot_peak_heights(results)
    plotter.plot_example_fits(slide_abs, height_axis, results, example_pixels)
    plotter.plot_fit_quality(slide_abs, height_axis, results)
    plotter.plot_residual_metrics(results)

    reconstructor = SarReconstructor(config)
    reconstructor.evaluate_sar_reconstruction()

    plt.show()
