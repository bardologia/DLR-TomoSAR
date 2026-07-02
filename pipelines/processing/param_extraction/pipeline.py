from __future__ import annotations

import gc
from pathlib import Path
from typing  import Tuple

import numpy as np

from configuration.param_extraction import ExtractionConfig, FitSettings
from pipelines.processing.param_extraction.io import ExtractionMetadataManager, ParameterIO
from tools.monitoring.logger import Logger


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
        peak_initialiser     : object | None        = None,
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
            peak_initialiser     = peak_initialiser,
        )

        self.logger.section("[Parameter Extractor Initialized]")
        self.logger.subsection(f"Backend : JAX GPU (free: {'+'.join(self.parameter_extraction.free_parameters)})")
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
    def __init__(self, config: ExtractionConfig, logger: Logger | None = None, peak_initialiser: object | None = None) -> None:
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
            peak_initialiser     = peak_initialiser,
        )
        self.metadata_manager = ExtractionMetadataManager(config, logger=self.logger)
        self.parameter_io     = ParameterIO(logger=self.logger)

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

    def run(self) -> dict[str, Path]:
        self.logger.section("[Param Extraction Pipeline Execution]")

        parameters_array, diagnostics = self._stage_extract()
        npy_path  = self._stage_save(parameters_array)
        diag_path = self.parameter_io.save_diagnostics(diagnostics, self.config.diagnostics_npz_path)
        del parameters_array, diagnostics
        gc.collect()

        meta_path = self.metadata_manager.save_run_metadata(npy_path, diag_path, self.tomogram_path, self.height_range)

        self.logger.section("[Param Extraction Completed]")

        return {
            "parameters_npy"   : npy_path,
            "diagnostics_npz"  : diag_path,
            "metadata"         : meta_path,
            "output_directory" : self.output_directory,
            "source_tomogram"  : self.tomogram_path,
        }
