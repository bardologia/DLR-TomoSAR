from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import multiprocessing as mp
import os
from scipy.optimize import curve_fit


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
class ProfileParallelConfig:
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
class ProfileConfig:
    tomo_file:          str
    output_dir:         Path
    height_axis_range:  Tuple[float, float]
    n_gaussians:        int            = 3
    range_index:        int            = 500
    n_display_profiles: int            = 6
    max_fit_iterations: int            = 5000
    save_results:       bool           = True
    gif_fps:            int            = 20
    base_colors:        list           = field(default_factory=lambda: [cm.tab10(i) for i in range(10)])
    parallel:           ProfileParallelConfig = field(default_factory=ProfileParallelConfig)

    @property
    def tomo_name(self) -> str:
        return Path(self.tomo_file).stem

    @property
    def run_directory(self) -> Path:
        return self.output_dir / self.tomo_name / f"rn{self.range_index}"


@dataclass
class ProfileFitResult:
    params:       np.ndarray
    fit_success:  np.ndarray
    n_gaussians:  int


class GaussianModel:
    @staticmethod
    def multi_gaussian(height_axis: np.ndarray, *parameters) -> np.ndarray:
        result      = np.zeros_like(height_axis, dtype=np.float64)
        n_gaussians = len(parameters) // 3

        for index in range(n_gaussians):
            mu        = parameters[3 * index]
            sigma     = parameters[3 * index + 1]
            amplitude = parameters[3 * index + 2]
            result   += amplitude * np.exp(-((height_axis - mu) ** 2) / (2 * sigma ** 2))

        return result

    @staticmethod
    def single_gaussian(height_axis: np.ndarray, mu: float, sigma: float, amplitude: float) -> np.ndarray:
        return amplitude * np.exp(-((height_axis - mu) ** 2) / (2 * sigma ** 2))

    @staticmethod
    def estimate_initial_params(height_axis: np.ndarray, profile: np.ndarray, n_gaussians: int) -> list:
        smoothed    = np.convolve(profile, np.ones(5) / 5, mode="same")
        sigma_guess = (height_axis[-1] - height_axis[0]) / (4.0 * n_gaussians)
        working     = smoothed.copy()
        initial     = []

        for _ in range(n_gaussians):
            peak_index     = np.argmax(working)
            peak_mu        = height_axis[peak_index]
            peak_amplitude = working[peak_index]
            initial.extend([peak_mu, sigma_guess, max(peak_amplitude, 1e-10)])
            suppression_mask         = np.abs(height_axis - peak_mu) < 2 * sigma_guess
            working[suppression_mask] = 0.0

        return initial


_worker_height_axis = None
_worker_n_gaussians = None
_worker_max_fit_iterations = None
_worker_slide_abs = None
_worker_fitter = None


def _init_profile_worker(height_axis: np.ndarray, n_gaussians: int, max_fit_iterations: int, slide_abs: np.ndarray):
    global _worker_height_axis, _worker_n_gaussians, _worker_max_fit_iterations
    global _worker_slide_abs, _worker_fitter
    _worker_height_axis = np.asarray(height_axis, dtype=np.float64)
    _worker_n_gaussians = int(n_gaussians)
    _worker_max_fit_iterations = int(max_fit_iterations)
    _worker_slide_abs = slide_abs
    _worker_fitter = ProfileFitter(ProfileConfig(
        tomo_file="",
        output_dir=Path("."),
        height_axis_range=(float(height_axis[0]), float(height_axis[-1])),
        n_gaussians=n_gaussians,
        max_fit_iterations=max_fit_iterations,
        parallel=ProfileParallelConfig(enabled=False),
    ))


def _fit_profile_worker(azimuth: int) -> Tuple[int, np.ndarray, bool]:
    profile = _worker_slide_abs[:, azimuth]
    optimized, converged = _worker_fitter.fit_single_profile(_worker_height_axis, profile, _worker_n_gaussians)
    return azimuth, optimized, converged


class ProfileFitter:
    def __init__(self, config: ProfileConfig):
        self.config = config
        self.model  = GaussianModel()

    def fit_single_profile(self, height_axis: np.ndarray, profile: np.ndarray, n_gaussians: Optional[int] = None) -> Tuple[np.ndarray, bool]:
        if n_gaussians is None:
            n_gaussians = self.config.n_gaussians

        n_params         = 3 * n_gaussians
        absolute_profile = np.abs(profile).astype(np.float64)
        initial_params   = self.model.estimate_initial_params(height_axis, absolute_profile, n_gaussians)

        height_low   = height_axis[0]
        height_high  = height_axis[-1]
        lower_bounds = []
        upper_bounds = []
        for _ in range(n_gaussians):
            lower_bounds.extend([height_low, 1e-6, 0.0])
            upper_bounds.extend([height_high, (height_high - height_low), np.inf])

        try:
            optimized, _ = curve_fit(
                self.model.multi_gaussian, height_axis, absolute_profile, p0=initial_params,
                bounds=(lower_bounds, upper_bounds), maxfev=self.config.max_fit_iterations,
            )
            return optimized, True
        except (RuntimeError, ValueError):
            return np.full(n_params, np.nan), False

    def _fit_all_profiles_serial(self, slide_abs: np.ndarray, height_axis: np.ndarray, n_gaussians: int, verbose: bool) -> ProfileFitResult:
        n_height, n_azimuth = slide_abs.shape
        n_params             = 3 * n_gaussians
        params               = np.full((n_azimuth, n_params), np.nan)
        fit_success          = np.zeros(n_azimuth, dtype=bool)

        for azimuth in range(n_azimuth):
            optimized, converged  = self.fit_single_profile(height_axis, slide_abs[:, azimuth], n_gaussians)
            params[azimuth]       = optimized
            fit_success[azimuth]  = converged

            if verbose and ((azimuth + 1) % 200 == 0 or azimuth == n_azimuth - 1):
                converged_so_far = fit_success[:azimuth + 1].sum()
                print(f"  [{azimuth + 1}/{n_azimuth}] fitted  (success: {converged_so_far}/{azimuth + 1})")

        return ProfileFitResult(params=params, fit_success=fit_success, n_gaussians=n_gaussians)

    def _fit_all_profiles_parallel(self, slide_abs: np.ndarray, height_axis: np.ndarray, n_gaussians: int, verbose: bool) -> ProfileFitResult:
        n_height, n_azimuth = slide_abs.shape
        n_params             = 3 * n_gaussians
        params               = np.full((n_azimuth, n_params), np.nan)
        fit_success          = np.zeros(n_azimuth, dtype=bool)
        n_workers_eff        = self.config.parallel.capped_workers(n_azimuth)
        chunksize            = self.config.parallel.auto_chunksize(n_azimuth)
        context              = mp.get_context(self.config.parallel.method)

        if verbose:
            print(f"  Using {n_workers_eff} worker(s), chunksize={chunksize} for per-pixel fitting")

        with ProcessPoolExecutor(max_workers=n_workers_eff, mp_context=context, initializer=_init_profile_worker, initargs=(height_axis, n_gaussians, self.config.max_fit_iterations, slide_abs)) as executor:
            mapped = executor.map(_fit_profile_worker, range(n_azimuth), chunksize=chunksize)
            for iteration, (azimuth, optimized, converged) in enumerate(mapped, start=1):
                params[azimuth] = optimized
                fit_success[azimuth] = converged

                if verbose and (iteration % 200 == 0 or iteration == n_azimuth):
                    converged_so_far = fit_success[:iteration].sum()
                    print(f"  [{iteration}/{n_azimuth}] fitted  (success: {converged_so_far}/{iteration})")

        return ProfileFitResult(params=params, fit_success=fit_success, n_gaussians=n_gaussians)

    def fit_all_profiles(self, slide_abs: np.ndarray, height_axis: np.ndarray, n_gaussians: Optional[int] = None, verbose: bool = True) -> ProfileFitResult:
        if n_gaussians is None:
            n_gaussians = self.config.n_gaussians

        _, n_azimuth = slide_abs.shape

        if verbose:
            print(f"Fitting {n_gaussians}-Gaussian to each pixel")

        if self.config.parallel.capped_workers(n_azimuth) > 1:
            fit_result = self._fit_all_profiles_parallel(slide_abs, height_axis, n_gaussians, verbose)
        else:
            fit_result = self._fit_all_profiles_serial(slide_abs, height_axis, n_gaussians, verbose)

        if verbose:
            total_converged = fit_result.fit_success.sum()
            print(f"Fitting complete: {total_converged}/{n_azimuth} profiles converged ({100 * total_converged / n_azimuth:.1f}%)")

        return fit_result


class TomogramLoader:
    @staticmethod
    def load_slice(tomo_file: str, range_index: int, height_axis_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        print(f"Loading tomogram slice range_index={range_index}:")
        with h5py.File(tomo_file, "r") as h5_file:
            raw_slice = h5_file["tomogram"][:, :, range_index]

        slide_abs                = np.abs(raw_slice)
        n_height, n_azimuth      = slide_abs.shape
        print(f"  Slice shape: height={n_height}, azimuth={n_azimuth}")
        height_axis = np.linspace(height_axis_range[0], height_axis_range[1], n_height)

        return slide_abs, height_axis

    @staticmethod
    def load_from_config(config: ProfileConfig) -> Tuple[np.ndarray, np.ndarray]:
        return TomogramLoader.load_slice(config.tomo_file, config.range_index, config.height_axis_range)


class SliceReconstructor:
    def __init__(self):
        self.model = GaussianModel()

    def reconstruct(self, slide_abs: np.ndarray, height_axis: np.ndarray, fit_result: ProfileFitResult, verbose: bool = True) -> np.ndarray:
        if verbose:
            print(f"Reconstructing slice from fitted {fit_result.n_gaussians}-Gaussians ...")
        reconstruction = np.zeros_like(slide_abs)
        for azimuth in range(slide_abs.shape[1]):
            if fit_result.fit_success[azimuth]:
                reconstruction[:, azimuth] = self.model.multi_gaussian(height_axis, *fit_result.params[azimuth])
        
        return reconstruction


class ParameterStorage:
    @staticmethod
    def save(config: ProfileConfig, fit_result: ProfileFitResult, height_axis: np.ndarray):
        config.run_directory.mkdir(parents=True, exist_ok=True)
        output_path = config.run_directory / f"pixel_profiles_Ng{fit_result.n_gaussians}.hd5"
        print(f"Saving fitted parameters → {output_path}")
        param_names = ", ".join(f"mu{k + 1}, sigma{k + 1}, A{k + 1}" for k in range(fit_result.n_gaussians))

        with h5py.File(output_path, "w") as h5_file:
            h5_file.create_dataset("params",       data=fit_result.params)
            h5_file.create_dataset("fit_success",   data=fit_result.fit_success)
            h5_file.create_dataset("height_axis",   data=height_axis)
            h5_file.attrs["rn_idx"]       = config.range_index
            h5_file.attrs["n_gaussians"]  = fit_result.n_gaussians
            h5_file.attrs["source_file"]  = config.tomo_file
            h5_file.attrs["param_order"]  = param_names


class ProfilePlotter:
    def __init__(self, config: ProfileConfig):
        self.config = config
        self.model  = GaussianModel()

    def _select_display_pixels(self, fit_result: ProfileFitResult) -> np.ndarray:
        converged_indices = np.where(fit_result.fit_success)[0]
        sample_indices    = np.linspace(0, len(converged_indices) - 1, self.config.n_display_profiles, dtype=int)
        return converged_indices[sample_indices]

    def plot_slice_with_profiles(self, slide_abs: np.ndarray, height_axis: np.ndarray, fit_result: ProfileFitResult) -> plt.Figure:
        print("Plotting tomogram slice with sampled vertical profiles ...")
        n_azimuth     = slide_abs.shape[1]
        selected_pixels = self._select_display_pixels(fit_result)

        profile_palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#f006aa"]
        profile_colors  = [profile_palette[i % len(profile_palette)] for i in range(self.config.n_display_profiles)]

        height_fine = np.linspace(height_axis[0], height_axis[-1], 500)

        figure = plt.figure(figsize=(18, 7), constrained_layout=True)
        grid   = figure.add_gridspec(1, 2, width_ratios=[1.4, 1])

        axis_slice = figure.add_subplot(grid[0])
        image      = axis_slice.imshow(slide_abs, origin="lower", cmap="jet", aspect="auto", extent=[0, n_azimuth, height_axis[0], height_axis[-1]])
        for index, pixel in enumerate(selected_pixels):
            axis_slice.axvline(pixel, color=profile_colors[index], lw=1.0, ls="--", alpha=0.85)
        axis_slice.set_xlabel("Azimuth (pixels)")
        axis_slice.set_ylabel("Height (m)")
        colorbar = plt.colorbar(image, ax=axis_slice, shrink=0.8)
        colorbar.set_label("Amplitude (a.u.)")

        axis_profiles = figure.add_subplot(grid[1], sharey=axis_slice)
        for index, pixel in enumerate(selected_pixels):
            profile = slide_abs[:, pixel]
            axis_profiles.plot(profile, height_axis, color=profile_colors[index], lw=0.8, alpha=0.6)
            if fit_result.fit_success[pixel]:
                fitted = self.model.multi_gaussian(height_fine, *fit_result.params[pixel])
                axis_profiles.plot(fitted, height_fine, color=profile_colors[index], lw=1.6, label=rf"az$\,={pixel}$")
        axis_profiles.set_xlabel("Amplitude (a.u.)")
        axis_profiles.legend(fontsize=8, loc="upper right", framealpha=0.85, edgecolor="gray")
        axis_profiles.grid(True, alpha=0.25, linewidth=0.5)
        axis_profiles.tick_params(labelleft=False)

        save_dir = self.config.run_directory / "pixel_profiles"
        save_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_dir / "slice_with_profiles.png")
        return figure

    def plot_parameter_distributions(self, fit_result: ProfileFitResult) -> plt.Figure:
        print("Plotting fitted parameter distributions ...")
        valid_params = fit_result.params[fit_result.fit_success]

        n_bins = 60
        category_edges = []
        category_ylims = []
        for param_index in range(3):
            values_cat = valid_params[:, param_index::3].ravel()
            values_cat = values_cat[np.isfinite(values_cat)]

            if values_cat.size == 0:
                edges = np.linspace(0.0, 1.0, n_bins + 1)
                y_max = 1.0
            else:
                value_min = float(np.nanmin(values_cat))
                value_max = float(np.nanmax(values_cat))
                if np.isclose(value_min, value_max):
                    pad = max(1e-6, 0.05 * max(abs(value_min), 1.0))
                    value_min -= pad
                    value_max += pad
                edges = np.linspace(value_min, value_max, n_bins + 1)

                y_max = 0.0
                for gaussian_index in range(fit_result.n_gaussians):
                    values_k = valid_params[:, 3 * gaussian_index + param_index]
                    values_k = values_k[np.isfinite(values_k)]
                    if values_k.size:
                        counts, _ = np.histogram(values_k, bins=edges)
                        y_max = max(y_max, float(np.max(counts)))
                y_max = max(1.0, 1.05 * y_max)

            category_edges.append(edges)
            category_ylims.append((0.0, y_max))

        figure, axes = plt.subplots(fit_result.n_gaussians, 3, figsize=(15, 3.0 * fit_result.n_gaussians), constrained_layout=True)
        if fit_result.n_gaussians == 1:
            axes = axes[np.newaxis, :]

        param_info = [
            (r"$\mu$",     "Height (m)"),
            (r"$\sigma$",  "Width (m)"),
            (r"$A$",       "Amplitude (a.u.)"),
        ]

        for gaussian_index in range(fit_result.n_gaussians):
            for param_index, (symbol, xlabel) in enumerate(param_info):
                axis = axes[gaussian_index, param_index]
                data = valid_params[:, 3 * gaussian_index + param_index]
                data = data[np.isfinite(data)]

                axis.hist(data, bins=category_edges[param_index], color=self.config.base_colors[gaussian_index], alpha=0.7, edgecolor="black", linewidth=0.4)
                median_value = np.median(data) if data.size else np.nan
                if np.isfinite(median_value):
                    axis.axvline(median_value, color="black", ls="--", lw=1.0, label=rf"Median$\,={median_value:.2f}$")
                axis.set_xlabel(xlabel)
                axis.set_ylabel("Count")
                axis.set_xlim(category_edges[param_index][0], category_edges[param_index][-1])
                axis.set_ylim(*category_ylims[param_index])
                if np.isfinite(median_value):
                    axis.legend(fontsize=7, framealpha=0.85, edgecolor="gray")
                axis.grid(True, alpha=0.25, linewidth=0.5, axis="y")

        save_dir = self.config.run_directory / "pixel_profiles"
        save_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_dir / "param_distributions.png")
        return figure

    def plot_parameter_maps(self, fit_result: ProfileFitResult) -> plt.Figure:
        print("Plotting parameter maps ...")
        n_azimuth    = fit_result.params.shape[0]
        azimuth_axis = np.arange(n_azimuth)

        valid_params = fit_result.params[fit_result.fit_success]
        y_limits = []
        for param_index in range(3):
            values_cat = valid_params[:, param_index::3].ravel()
            values_cat = values_cat[np.isfinite(values_cat)]

            if values_cat.size == 0:
                y_limits.append((0.0, 1.0))
                continue

            value_min = float(np.nanmin(values_cat))
            value_max = float(np.nanmax(values_cat))
            if np.isclose(value_min, value_max):
                pad = max(1e-6, 0.05 * max(abs(value_min), 1.0))
                value_min -= pad
                value_max += pad
            y_limits.append((value_min, value_max))

        figure, axes = plt.subplots(fit_result.n_gaussians, 3, figsize=(16, 3.2 * fit_result.n_gaussians), constrained_layout=True)
        if fit_result.n_gaussians == 1:
            axes = axes[np.newaxis, :]

        for gaussian_index in range(fit_result.n_gaussians):
            labels_and_units = [
                (rf"$\mu_{{{gaussian_index + 1}}}$",     "(m)"),
                (rf"$\sigma_{{{gaussian_index + 1}}}$",  "(m)"),
                (rf"$A_{{{gaussian_index + 1}}}$",       "(a.u.)"),
            ]
            for param_index, (label, unit) in enumerate(labels_and_units):
                axis = axes[gaussian_index, param_index]
                data = fit_result.params[:, 3 * gaussian_index + param_index].copy()
                data[~fit_result.fit_success] = np.nan
                axis.plot(azimuth_axis, data, ".", markersize=1.5, alpha=0.65, color=self.config.base_colors[gaussian_index], rasterized=True)
                if gaussian_index == fit_result.n_gaussians - 1:
                    axis.set_xlabel("Azimuth (pixels)")
                else:
                    axis.set_xlabel("")
                axis.set_ylabel(f"{label} {unit}")
                axis.set_ylim(*y_limits[param_index])
                axis.grid(True, alpha=0.25, linewidth=0.5)

        save_dir = self.config.run_directory / "params_maps"
        save_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_dir / "gaussian_params_mu_sigma_A.png")
        return figure


class SweepGifGenerator:
    def __init__(self, config: ProfileConfig):
        self.config = config
        self.model  = GaussianModel()

    def generate(self, slide_abs: np.ndarray, height_axis: np.ndarray, fit_result: ProfileFitResult) -> None:
        print("Generating animated GIF (sweeping through azimuth pixels) ...")
        n_azimuth      = slide_abs.shape[1]
        step_size      = max(1, n_azimuth // 300)
        sweep_pixels   = np.arange(0, n_azimuth, step_size)
        height_fine    = np.linspace(height_axis[0], height_axis[-1], 500)
        global_amp_max = np.nanmax(slide_abs) * 1.1

        save_dir = self.config.run_directory / "profile_sweep"
        save_dir.mkdir(parents=True, exist_ok=True)
        gif_path = save_dir / "azimuth_sweep_profiles.gif"

        figure, (axis_profile, axis_slice) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.4]})
        plt.subplots_adjust(wspace=0.25)

        line_observed,  = axis_profile.plot([], [], "k-", lw=0.9, label="Observed")
        line_fitted,    = axis_profile.plot([], [], color="tab:red", ls="-", lw=1.8, label="Fitted")
        fill_patches    = {index: None for index in range(fit_result.n_gaussians)}

        axis_profile.set_xlim(height_axis[0], height_axis[-1])
        axis_profile.set_ylim(0, global_amp_max)
        axis_profile.set_xlabel("Height (m)")
        axis_profile.set_ylabel("Amplitude (a.u.)")
        axis_profile.grid(True, alpha=0.25, linewidth=0.5)
        axis_profile.legend(loc="upper right", framealpha=0.85, edgecolor="gray")

        axis_slice.imshow(slide_abs, origin="lower", cmap="jet", aspect="auto", extent=[0, n_azimuth, height_axis[0], height_axis[-1]])
        vertical_marker = axis_slice.axvline(0, color="white", lw=1.2, ls="--")
        axis_slice.set_xlabel("Azimuth (pixels)")
        axis_slice.set_ylabel("Height (m)")

        def _update_frame(frame_index):
            pixel   = sweep_pixels[frame_index]
            profile = slide_abs[:, pixel]
            line_observed.set_data(height_axis, profile)

            for gaussian_index in range(fit_result.n_gaussians):
                if fill_patches[gaussian_index] is not None:
                    fill_patches[gaussian_index].remove()

            if fit_result.fit_success[pixel]:
                fitted_curve = self.model.multi_gaussian(height_fine, *fit_result.params[pixel])
                line_fitted.set_data(height_fine, fitted_curve)
                for gaussian_index in range(fit_result.n_gaussians):
                    single_peak = self.model.single_gaussian(height_fine, *fit_result.params[pixel, 3 * gaussian_index:3 * gaussian_index + 3])
                    fill_patches[gaussian_index] = axis_profile.fill_between(height_fine, single_peak, alpha=0.18, color=self.config.base_colors[gaussian_index], label="_")
            else:
                line_fitted.set_data([], [])
                for gaussian_index in range(fit_result.n_gaussians):
                    fill_patches[gaussian_index] = axis_profile.fill_between([], [], alpha=0)

            vertical_marker.set_xdata([pixel, pixel])
            return [line_observed, line_fitted, vertical_marker] + [fill_patches[index] for index in range(fit_result.n_gaussians)]

        animated = animation.FuncAnimation(figure, _update_frame, frames=len(sweep_pixels), interval=1000 // self.config.gif_fps, blit=False)
        animated.save(str(gif_path), writer="pillow", fps=self.config.gif_fps, dpi=100)
        plt.close(figure)
        print(f"GIF saved → {gif_path}  ({len(sweep_pixels)} frames, {self.config.gif_fps} fps)")


if __name__ == "__main__":

    config = ProfileConfig(
        tomo_file          = "/ste/rnd/User/vice_vi/Pruebas/TOMO/TOMO-SR/tomograms/tomo_17sartom0102_Lhv_MSF_w10_20.hd5",
        output_dir         = Path("/ste/rnd/User/vice_vi/Pruebas/TOMO/TOMO-SR/"),
        height_axis_range  = (-20, 80),
        n_gaussians        = 3,
        range_index        = 500,
        n_display_profiles = 4,
        save_results       = True,
    )

    loader                       = TomogramLoader()
    fitter                       = ProfileFitter(config)
    reconstructor                = SliceReconstructor()
    plotter                      = ProfilePlotter(config)
    gif_generator                = SweepGifGenerator(config)

    slide_abs, height_axis       = loader.load_from_config(config)
    fit_result                   = fitter.fit_all_profiles(slide_abs, height_axis)
    reconstruction               = reconstructor.reconstruct(slide_abs, height_axis, fit_result)

    if config.save_results:
        ParameterStorage.save(config, fit_result, height_axis)

    plotter.plot_slice_with_profiles(slide_abs, height_axis, fit_result)
    plotter.plot_parameter_distributions(fit_result)
    plotter.plot_parameter_maps(fit_result)
    gif_generator.generate(slide_abs, height_axis, fit_result)

    plt.show()