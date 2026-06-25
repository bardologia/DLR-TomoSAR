from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib  import Path
from typing   import Tuple

import numpy as np

from configuration.param_extraction   import ExtractionConfig, FitSettings
from pipelines.processing.param_extraction.metrics import FittingMetricsCalculator
from pipelines.processing.param_extraction.plots   import FittingResultPlotter
from tools.data.io                                 import FileIO
from tools.monitoring.logger                       import Logger


class DatasetQueueResolver:
    def __init__(self, base_path: Path, dataset_filter: list) -> None:
        self.base_path      = base_path
        self.dataset_filter = dataset_filter

    def resolve(self) -> list[Path]:
        if not isinstance(self.dataset_filter, (list, tuple)):
            raise TypeError(f"dataset_filter must be a list of dataset names, got {type(self.dataset_filter).__name__}: {self.dataset_filter!r}")

        if not self.base_path.is_dir():
            raise NotADirectoryError(f"dataset_base_path does not exist: {self.base_path}")

        dataset_dirs = sorted(
            [d for d in self.base_path.iterdir() if d.is_dir()]
            if not self.dataset_filter
            else [self.base_path / str(name) for name in self.dataset_filter]
        )

        invalid = [d for d in dataset_dirs if not (d / "data").is_dir()]
        if invalid:
            names = ", ".join(d.name for d in invalid)
            raise NotADirectoryError(f"Queue entries without a data/ directory under {self.base_path}: {names}")

        return dataset_dirs


class ExtractionMetadataManager:
    def __init__(self, config: ExtractionConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger

    def save_run_metadata(self, npy_path: Path, diagnostics_path: Path, tomogram_path: Path, height_range: tuple) -> Path:
        meta_path = self.config.output_directory / "param_extraction_meta.json"
        ext       = self.config.fit_settings

        payload = {
            "timestamp"           : datetime.now().isoformat(timespec="seconds"),
            "processed_data_path" : str(self.config.processed_data_path),
            "source_tomogram"     : str(tomogram_path),
            "height_range"        : list(height_range),
            "output_directory"    : str(self.config.output_directory),
            "output_prefix"       : self.config.output_prefix,
            "output_suffix"       : self.config.output_suffix_value,
            "parameters_npy"      : npy_path.name,
            "diagnostics_npz"     : diagnostics_path.name,
            "k_max"               : ext.fit_config.k_max,
            "lambda_k"            : ext.fit_config.lambda_k,
            "sigma_init_divisor"  : ext.fit_config.sigma_init_divisor,
            "activity_threshold"  : ext.fit_config.activity_threshold,
        }

        FileIO.save_json(payload, meta_path)

        self.logger.subsection(f"-> Metadata written: {meta_path}")
        return meta_path


class ParameterIO:
    def __init__(self, logger : Logger) -> None:
        self.logger = logger

    def save_params(self, parameters_array : np.ndarray, npy_path : Path) -> Path:
        npy_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.subsection(f"Saving parameter stack of shape {parameters_array.shape} to disk")
        np.save(str(npy_path), np.ascontiguousarray(parameters_array), allow_pickle=False)

        return npy_path

    def load_params(self, npy_path : Path) -> np.ndarray:
        self.logger.subsection("Loading saved parameters for metrics and plots")
        return np.load(str(npy_path)).astype(np.float32, copy=False)

    def save_diagnostics(self, diagnostics : dict, npz_path : Path) -> Path:
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.subsection(f"Saving fit diagnostics ({', '.join(diagnostics.keys())}) to disk")
        np.savez(str(npz_path), **diagnostics)

        return npz_path

    def load_diagnostics(self, npz_path : Path) -> dict:
        self.logger.subsection("Loading fit diagnostics for metrics and plots")
        with np.load(str(npz_path)) as data:
            return {key: data[key] for key in data.files}

    def load_metadata(self, meta_path : Path) -> dict:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)


class ParameterExtractor:
    def __init__(
        self,
        parameter_extraction : FitSettings,
        logger               : Logger,
        range_batch_size     : int                 = 256,
        adam_steps           : int                 = 800,
        adam_lr              : float               = 1e-2,
        adam_b1              : float               = 0.9,
        adam_b2              : float               = 0.999,
        gpu_pixel_batch_size : int                 = 8192,
        init_workers         : int | None          = None,
    ) -> None:

        from pipelines.processing.param_extraction.sigma import SigmaFittingExtractor

        self.parameter_extraction = parameter_extraction
        self.logger               = logger
        self.range_batch_size     = range_batch_size
        self.gpu_pixel_batch_size = gpu_pixel_batch_size
        self.adam_steps           = adam_steps
        self.adam_lr              = adam_lr
        self.adam_b1              = adam_b1
        self.adam_b2              = adam_b2
        self.init_workers         = init_workers

        fit_cfg = parameter_extraction.fit_config

        k_max              = fit_cfg.k_max
        lambda_k           = fit_cfg.lambda_k
        prominence_frac    = fit_cfg.prominence_frac
        sigma_init_divisor = fit_cfg.sigma_init_divisor

        self._gpu_extractor = SigmaFittingExtractor(
            fit_settings         = parameter_extraction,
            logger               = logger,
            range_batch_size     = range_batch_size,
            adam_steps           = adam_steps,
            adam_lr              = adam_lr,
            adam_b1              = adam_b1,
            adam_b2              = adam_b2,
            k_max                = k_max,
            lambda_k             = lambda_k,
            prominence_frac      = prominence_frac,
            sigma_init_divisor   = sigma_init_divisor,
            gpu_pixel_batch_size = gpu_pixel_batch_size,
            init_workers         = init_workers,
        )

        self.logger.section("[Parameter Extractor Initialized]")
        self.logger.subsection("Backend : JAX GPU (Sigma Only)")
        self.logger.subsection(f"Method  : {self.parameter_extraction.fitting_method}")

    @staticmethod
    def _sort_gaussians(parameters_array: np.ndarray, n_gaussians: int, activity_threshold: float) -> np.ndarray:
        n_params, Az, R = parameters_array.shape
        reshaped = parameters_array.reshape(n_gaussians, 3, Az, R)

        amps = reshaped[:, 0, :, :]
        mus  = reshaped[:, 1, :, :]

        sort_keys    = np.where(amps > activity_threshold, mus, np.inf)
        order        = np.argsort(sort_keys, axis=0)
        out_reshaped = np.take_along_axis(reshaped, order[:, np.newaxis, :, :], axis=0)

        return out_reshaped.reshape(n_params, Az, R)

    def run(self, tomogram_path: Path, height_range: Tuple[float, float]) -> Tuple[np.ndarray, dict]:
        self.logger.section(f"[Extraction Start] Source: {tomogram_path.name}")

        parameters_array, diagnostics = self._gpu_extractor.run(tomogram_path, height_range)
        parameters_array              = self._sort_gaussians(parameters_array, self.parameter_extraction.fit_config.k_max, self.parameter_extraction.fit_config.activity_threshold)

        self.logger.subsection("[Extraction Complete]")
        return parameters_array, diagnostics


class ParamExtractionPipeline:
    def __init__(self, config: ExtractionConfig, logger: Logger | None = None) -> None:
        self.config           = config
        self.tomogram_path    = config.discover_tomogram_path()
        self.height_range     = config.discover_height_range()
        self.output_directory = config.output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)

        log_dir = self.output_directory / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logger or Logger(log_dir = str(log_dir), name = "param_extraction", level = "INFO")

        self.parameter_extractor   = ParameterExtractor(
            parameter_extraction = config.fit_settings,
            logger               = self.logger,
            range_batch_size     = config.range_batch_size,
            adam_steps           = config.adam_steps,
            adam_lr              = config.adam_lr,
            adam_b1              = config.adam_b1,
            adam_b2              = config.adam_b2,
            gpu_pixel_batch_size = config.gpu_pixel_batch_size,
            init_workers         = config.parameter_workers,
        )
        self.metadata_manager = ExtractionMetadataManager(config, logger=self.logger)
        self.parameter_io     = ParameterIO(logger=self.logger)

        n_K                = config.fit_settings.fit_config.k_max
        threshold_factor   = float(config.fit_settings.fit_config.threshold_factor)
        truncation_index   = int(  config.fit_settings.fit_config.truncation_index)
        activity_threshold = float(config.fit_settings.fit_config.activity_threshold)

        self.metrics_calculator = FittingMetricsCalculator(
            n_gaussians      = n_K,
            logger           = self.logger,
            threshold_factor = threshold_factor,
            truncation_index = truncation_index,
            amp_threshold    = activity_threshold,
        )
        self.result_plotter     = FittingResultPlotter(
            output_directory = self.output_directory,
            n_gaussians      = n_K,
            logger           = self.logger,
            threshold_factor = threshold_factor,
            truncation_index = truncation_index,
            amp_threshold    = activity_threshold,
        )

        self.logger.section("[Param Extraction Pipeline Initialized]")
        self.logger.subsection(f"Source tomogram   : {self.tomogram_path}")
        self.logger.subsection(f"Height range      : {self.height_range}")
        self.logger.subsection(f"Output directory  : {self.output_directory}")
        self.logger.subsection(f"k_max             : {config.fit_settings.fit_config.k_max}")
        self.logger.subsection(f"Fit method        : {config.fit_settings.fitting_method}")

    def _stage_extract(self) -> Tuple[np.ndarray, dict]:
        self.logger.subsection("Extracting multi-Gaussian parameters")
        return self.parameter_extractor.run(tomogram_path = self.tomogram_path, height_range = self.height_range)

    def _stage_save(self, parameters_array: np.ndarray) -> Path:
        return self.parameter_io.save_params(parameters_array, self.config.parameters_npy_path)

    def _stage_metrics(self, parameters_array: np.ndarray, metadata: dict, diagnostics: dict) -> dict:
        return self.metrics_calculator.run(parameters_array, metadata, self.tomogram_path, diagnostics)

    def _stage_summary(self, metrics_dict: dict) -> Path:
        payload = {
            "global_summary" : metrics_dict.get("global_summary", {}),
            "per_k_summary"  : metrics_dict.get("per_k_summary",  {}),
            "snr_summary"    : metrics_dict.get("snr_summary",    {}),
        }

        summary_path = self.output_directory / "fit_metrics_summary.json"
        FileIO.save_json(payload, summary_path)

        self.logger.subsection(f"-> Metrics summary written: {summary_path}")
        return summary_path

    def _stage_plots(self, parameters_array: np.ndarray, metadata: dict, metrics_dict: dict) -> dict[str, Path]:
        return self.result_plotter.run(parameters_array, metrics_dict, metadata, self.tomogram_path)

    def run(self) -> dict[str, Path]:
        self.logger.section("[Param Extraction Pipeline Execution]")

        parameters_array, diagnostics = self._stage_extract()
        npy_path  = self._stage_save(parameters_array)
        diag_path = self.parameter_io.save_diagnostics(diagnostics, self.config.diagnostics_npz_path)
        del parameters_array, diagnostics
        gc.collect()

        meta_path = self.metadata_manager.save_run_metadata(npy_path, diag_path, self.tomogram_path, self.height_range)

        parameters_array = self.parameter_io.load_params(npy_path)
        metadata         = self.parameter_io.load_metadata(meta_path)
        diagnostics      = self.parameter_io.load_diagnostics(diag_path)

        metrics_dict = self._stage_metrics(parameters_array, metadata, diagnostics)
        summary_path = self._stage_summary(metrics_dict)
        plot_paths   = self._stage_plots(parameters_array, metadata, metrics_dict)
        del parameters_array, diagnostics
        gc.collect()

        self.logger.section("[Param Extraction Completed]")

        return {
            "parameters_npy"   : npy_path,
            "diagnostics_npz"  : diag_path,
            "metadata"         : meta_path,
            "metrics_summary"  : summary_path,
            "output_directory" : self.output_directory,
            "source_tomogram"  : self.tomogram_path,
            "plots"            : plot_paths,
        }
