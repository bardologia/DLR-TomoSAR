from __future__ import annotations

import gc
import multiprocessing as mp
import threading
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit, minimize

from configuration.param_extraction_config import FitMode, FitSettings
from pipelines.param_extraction_pipeline.gaussian_model import GaussianModel
from tools.logger import Logger


class FittingMethods:
    @staticmethod
    def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        residual_sum = float(np.sum((y_true - y_pred) ** 2))
        total_sum    = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if total_sum <= 1e-20:
            return float("nan")
        return 1.0 - (residual_sum / total_sum)

    @staticmethod
    def _build_bounds(
        number_of_gaussians : int,
        height_low          : float,
        height_high         : float,
        lower_bounds_config,
        upper_bounds_config,
        sigma_lower_min     : float = 1e-6,
    ) -> tuple:
        if lower_bounds_config is not None:
            lower_bounds = list(lower_bounds_config)
        else:
            lower_bounds = []
            for _ in range(number_of_gaussians):
                lower_bounds.extend([0.0, height_low, sigma_lower_min])

        if upper_bounds_config is not None:
            upper_bounds = list(upper_bounds_config)
        else:
            upper_bounds = []
            for _ in range(number_of_gaussians):
                upper_bounds.extend([np.inf, height_high, height_high - height_low])

        return lower_bounds, upper_bounds

    @staticmethod
    def fit_curve(
        range_start         : int,
        range_end           : int,
        tomogram_file_path  : str,
        height_range_start  : float,
        height_range_end    : float,
        number_of_gaussians : int,
        max_fit_iterations  : int,
        fit_config          : FitMode.Adaptive,
        progress_queue      = None,
    ) -> tuple[list[np.ndarray], int, int, float, int]:
        threshold_factor   = fit_config.threshold_factor
        truncation_index   = fit_config.truncation_index
        configured_initial = fit_config.initial_guess

        model                = GaussianModel()
        number_of_parameters = 3 * number_of_gaussians
        fitted_slices        = []
        failed_fits          = 0
        attempted_fits       = 0
        quality_sum          = 0.0
        quality_count        = 0

        tomogram_mmap = np.load(tomogram_file_path, mmap_mode="r", allow_pickle=False)
        height_samples, azimuth_size, _ = tomogram_mmap.shape
        height_axis                     = np.linspace(height_range_start, height_range_end, height_samples)

        lower_bounds, upper_bounds = FittingMethods._build_bounds(
            number_of_gaussians = number_of_gaussians,
            height_low          = float(height_axis[0]),
            height_high         = float(height_axis[-1]),
            lower_bounds_config = fit_config.lower_bounds,
            upper_bounds_config = fit_config.upper_bounds,
        )
        bounds_pair = (lower_bounds, upper_bounds)

        for range_index in range(range_start, range_end):
            tomogram_slice              = tomogram_mmap[:, :, range_index]
            fitted_parameters_for_slice = np.zeros((number_of_parameters, azimuth_size))
            previous_valid_parameters   = np.array(configured_initial) if configured_initial is not None else np.zeros(number_of_parameters)

            for azimuth_index in range(azimuth_size):
                absolute_profile  = np.abs(tomogram_slice[:, azimuth_index]).astype(np.float64)
                maximum_amplitude = np.max(absolute_profile) * threshold_factor
                profile           = np.where(absolute_profile > maximum_amplitude, absolute_profile, 0.0)
                profile[truncation_index:] = 0.0

                initial_parameters = (
                    list(configured_initial)
                    if configured_initial is not None
                    else model.estimate_initial_parameters(height_axis, profile, number_of_gaussians)
                )

                if np.max(profile) < 1e-7:
                    fitted_parameters_for_slice[:, azimuth_index] = initial_parameters
                else:
                    attempted_fits += 1
                    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
                        warnings.filterwarnings("ignore", category=OptimizeWarning)
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        try:
                            fitted_parameters, _ = curve_fit(
                                f      = model.multi_gaussian,
                                xdata  = height_axis,
                                ydata  = profile,
                                p0     = initial_parameters,
                                bounds = bounds_pair,
                                maxfev = max_fit_iterations,
                            )
                            predicted_profile = model.multi_gaussian(height_axis, *fitted_parameters)
                            r2_score          = FittingMethods._compute_r2(profile, predicted_profile)
                            if np.isfinite(r2_score):
                                quality_sum   += float(r2_score)
                                quality_count += 1
                            fitted_parameters_for_slice[:, azimuth_index] = fitted_parameters
                            previous_valid_parameters = fitted_parameters
                        except (RuntimeError, ValueError):
                            failed_fits += 1
                            fitted_parameters_for_slice[:, azimuth_index] = previous_valid_parameters

            fitted_slices.append(fitted_parameters_for_slice)
            del tomogram_slice
            if progress_queue is not None:
                progress_queue.put(1)

        return fitted_slices, failed_fits, attempted_fits, quality_sum, quality_count

    @staticmethod
    def fit_mle(
        range_start         : int,
        range_end           : int,
        tomogram_file_path  : str,
        height_range_start  : float,
        height_range_end    : float,
        number_of_gaussians : int,
        max_fit_iterations  : int,
        fit_config          : FitMode.MLE,
        progress_queue      = None,
    ) -> tuple[list[np.ndarray], int, int, float, int]:
        model                = GaussianModel()
        number_of_parameters = 3 * number_of_gaussians
        fitted_slices        = []
        configured_initial   = fit_config.initial_guess
        epsilon              = fit_config.epsilon
        ftol                 = fit_config.ftol
        gtol                 = fit_config.gtol
        failed_fits          = 0
        attempted_fits       = 0
        quality_sum          = 0.0
        quality_count        = 0

        tomogram_mmap = np.load(tomogram_file_path, mmap_mode="r", allow_pickle=False)
        height_samples, azimuth_size, _ = tomogram_mmap.shape
        height_axis                     = np.linspace(height_range_start, height_range_end, height_samples)

        height_low  = float(height_axis[0])
        height_high = float(height_axis[-1])

        lower_bounds, upper_bounds = FittingMethods._build_bounds(
            number_of_gaussians = number_of_gaussians,
            height_low          = height_low,
            height_high         = height_high,
            lower_bounds_config = fit_config.lower_bounds,
            upper_bounds_config = fit_config.upper_bounds,
        )
        lbfgsb_bounds = list(zip(lower_bounds, upper_bounds))

        def _neg_log_likelihood(params: np.ndarray, y: np.ndarray) -> float:
            mu  = model.multi_gaussian(height_axis, *params)
            nll = np.sum(mu - y * np.log(mu + epsilon))
            return float(nll)

        for range_index in range(range_start, range_end):
            tomogram_slice              = tomogram_mmap[:, :, range_index]
            fitted_parameters_for_slice = np.zeros((number_of_parameters, azimuth_size))
            previous_valid_parameters   = (
                np.array(configured_initial) if configured_initial is not None
                else np.zeros(number_of_parameters)
            )

            for azimuth_index in range(azimuth_size):
                absolute_profile = np.abs(tomogram_slice[:, azimuth_index]).astype(np.float64)
                initial_parameters = (
                    list(configured_initial)
                    if configured_initial is not None
                    else model.estimate_initial_parameters(height_axis, absolute_profile, number_of_gaussians)
                )

                if np.max(absolute_profile) < 1e-7:
                    fitted_parameters_for_slice[:, azimuth_index] = initial_parameters
                else:
                    attempted_fits += 1
                    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        try:
                            result = minimize(
                                fun     = _neg_log_likelihood,
                                x0      = np.array(initial_parameters, dtype=np.float64),
                                args    = (absolute_profile,),
                                method  = "L-BFGS-B",
                                bounds  = lbfgsb_bounds,
                                options = {
                                    "maxiter" : max_fit_iterations,
                                    "ftol"    : ftol,
                                    "gtol"    : gtol,
                                },
                            )
                            fitted_parameters = result.x
                            predicted_profile = model.multi_gaussian(height_axis, *fitted_parameters)
                            r2_score          = FittingMethods._compute_r2(absolute_profile, predicted_profile)
                            if np.isfinite(r2_score):
                                quality_sum   += float(r2_score)
                                quality_count += 1
                            fitted_parameters_for_slice[:, azimuth_index] = fitted_parameters
                            previous_valid_parameters = fitted_parameters
                        
                        except (ValueError, np.linalg.LinAlgError):
                            failed_fits += 1
                            fitted_parameters_for_slice[:, azimuth_index] = previous_valid_parameters

            fitted_slices.append(fitted_parameters_for_slice)
            del tomogram_slice
            if progress_queue is not None:
                progress_queue.put(1)

        return fitted_slices, failed_fits, attempted_fits, quality_sum, quality_count


FittingMethods.REGISTRY = {
    FitMode.Adaptive : FittingMethods.fit_curve,
    FitMode.MLE      : FittingMethods.fit_mle,
}


class ParameterExtractor:
    def __init__(
        self,
        parameter_extraction : FitSettings,
        parameter_workers    : int,
        logger               : Logger,
    ) -> None:
        self.parameter_extraction = parameter_extraction
        self.parameter_workers    = parameter_workers
        self.logger               = logger

        self.logger.section("[Parameter Extractor Initialized]")
        self.logger.subsection(f"Workers : {self.parameter_workers}")
        self.logger.subsection(f"Method  : {self.parameter_extraction.fitting_method}")

    def _read_tomogram_shape(self, tomogram_path: Path) -> Tuple[int, int, int]:
        tomogram_mmap = np.load(str(tomogram_path), mmap_mode="r", allow_pickle=False)
        shape = tomogram_mmap.shape
        return int(shape[0]), int(shape[1]), int(shape[2])

    def _parallel_extraction(
        self,
        total_range_bins    : int,
        tomogram_file_path  : str,
        number_of_workers   : int,
        height_range_start  : float,
        height_range_end    : float,
        number_of_gaussians : int,
        fit_config          : object,
        max_fit_iterations  : int,
    ) -> tuple[list[np.ndarray], int, int, float]:
       
        target_function = FittingMethods.REGISTRY.get(type(fit_config))
        chunk_size      = total_range_bins // max(1, number_of_workers)
        manager         = mp.Manager()
        progress_queue  = manager.Queue()
        tasks           = []

        for worker_index in range(number_of_workers):
            start_index = worker_index * chunk_size
            end_index   = total_range_bins if worker_index == (number_of_workers - 1) else (worker_index + 1) * chunk_size
            tasks.append((
                start_index, end_index, tomogram_file_path, height_range_start, height_range_end,
                number_of_gaussians, max_fit_iterations, fit_config, progress_queue,
            ))

        progress_bar = self.logger.track(transient=True)
        progress     = progress_bar.__enter__()
        bar_task     = progress.add_task("  [section]Fitting range bins[/section]", total=total_range_bins)
        stop_listener = threading.Event()

        def _listen() -> None:
            while not (stop_listener.is_set() and progress_queue.empty()):
                try:
                    progress_queue.get(timeout=0.2)
                except Exception:
                    continue
                progress.advance(bar_task)

        listener_thread = threading.Thread(target=_listen, daemon=True)
        listener_thread.start()

        try:
            with mp.Pool(processes=number_of_workers) as pool:
                results = pool.starmap(target_function, tasks)
        finally:
            stop_listener.set()
            listener_thread.join(timeout=2.0)
            progress_bar.__exit__(None, None, None)

        all_slices      = []
        total_failed    = 0
        total_attempted = 0
        total_q_sum     = 0.0
        total_q_count   = 0

        for fitted_slices, failed_fits, attempted_fits, quality_sum, quality_count in results:
            all_slices.extend(fitted_slices)
            total_failed    += failed_fits
            total_attempted += attempted_fits
            total_q_sum     += quality_sum
            total_q_count   += quality_count

        average_quality = (total_q_sum / total_q_count) if total_q_count > 0 else float("nan")
        return all_slices, total_failed, total_attempted, average_quality

    def run(self, tomogram_path: Path, height_range: Tuple[float, float]) -> np.ndarray:
        self.logger.section(f"[Extraction Start] Source: {tomogram_path.name}")

        _, _, range_size = self._read_tomogram_shape(tomogram_path)

        fitted_results, failed_fits, attempted_fits, average_quality = self._parallel_extraction(
            total_range_bins    = range_size,
            tomogram_file_path  = str(tomogram_path),
            number_of_workers   = self.parameter_workers,
            height_range_start  = height_range[0],
            height_range_end    = height_range[1],
            number_of_gaussians = self.parameter_extraction.number_of_gaussians,
            fit_config          = self.parameter_extraction.fit_config,
            max_fit_iterations  = self.parameter_extraction.max_fit_iterations,
        )

        parameters_array = np.stack(fitted_results, axis=-1)
        del fitted_results
        gc.collect()

        failed_ratio = (100.0 * failed_fits / attempted_fits) if attempted_fits > 0 else 0.0
        self.logger.subsection(f"Failed fits: {failed_fits}/{attempted_fits} ({failed_ratio:.2f}%)")
        if np.isfinite(average_quality):
            self.logger.subsection(f"Average fit quality (R²): {average_quality:.4f}")
        else:
            self.logger.subsection("Average fit quality (R²): N/A")
        self.logger.subsection("[Extraction Complete]")
        return parameters_array
