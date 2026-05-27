from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

from configuration.param_extraction_config import ExtractionConfig
from pipelines.param_pipeline.fitting      import ParameterExtractor
from pipelines.param_pipeline.metadata     import ExtractionMetadataManager
from pipelines.param_pipeline.metrics      import FittingMetricsCalculator
from pipelines.param_pipeline.plots        import FittingResultPlotter
from tools.logger                          import Logger


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
            parameter_workers    = config.parameter_workers,
            logger               = self.logger,
            use_gpu              = config.use_gpu,
            gpu_batch_size       = config.gpu_batch_size,
            adam_steps           = config.adam_steps,
            adam_lr              = config.adam_lr,
            adam_b1              = config.adam_b1,
            adam_b2              = config.adam_b2,
            gpu_device_ids       = config.gpu_device_ids,
            r2_sample_cap        = config.r2_sample_cap,
            gpu_pixel_batch_size = config.gpu_pixel_batch_size,
        )
        self.metadata_manager   = ExtractionMetadataManager(config, logger=self.logger)

        n_K = getattr(config.fit_settings.fit_config, "k_max", config.fit_settings.number_of_gaussians)

        self.metrics_calculator = FittingMetricsCalculator(
            n_gaussians = n_K,
            logger      = self.logger,
        )
        self.result_plotter     = FittingResultPlotter(
            output_directory = self.output_directory,
            n_gaussians      = n_K,
            logger           = self.logger,
        )

        self.logger.section("[Param Extraction Pipeline Initialized]")
        self.logger.subsection(f"Source tomogram   : {self.tomogram_path}")
        self.logger.subsection(f"Height range      : {self.height_range}")
        self.logger.subsection(f"Output directory  : {self.output_directory}")
        self.logger.subsection(f"N gaussians       : {config.fit_settings.number_of_gaussians}")
        self.logger.subsection(f"Fit method        : {config.fit_settings.fitting_method}")

    def _stage_extract(self) -> np.ndarray:
        self.logger.subsection("Extracting multi-Gaussian parameters")
        return self.parameter_extractor.run(tomogram_path = self.tomogram_path, height_range = self.height_range)

    def _stage_save(self, parameters_array: np.ndarray) -> Path:
        npy_path = self.config.parameters_npy_path
        npy_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.subsection(f"Saving parameter stack of shape {parameters_array.shape} to disk")
        np.save(str(npy_path), np.ascontiguousarray(parameters_array), allow_pickle=False)
        
        del parameters_array
        gc.collect()
        
        return npy_path

    def _stage_metrics(self, npy_path: Path, meta_path: Path) -> dict:
        self.logger.subsection("Loading saved parameters for metrics")
        parameters_array = np.load(str(npy_path)).astype(np.float32, copy=False)
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        result = self.metrics_calculator.run(parameters_array, metadata, self.tomogram_path)
        del parameters_array
        gc.collect()
        return result

    def _stage_plots(self, npy_path: Path, meta_path: Path, metrics_dict: dict) -> dict[str, Path]:
        self.logger.subsection("Loading saved parameters for plots")
        parameters_array = np.load(str(npy_path)).astype(np.float32, copy=False)
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        result = self.result_plotter.run(parameters_array, metrics_dict, metadata, self.tomogram_path)
        del parameters_array
        gc.collect()
        return result

    def run(self) -> dict[str, Path]:
        self.logger.section("[Param Extraction Pipeline Execution]")

        parameters_array = self._stage_extract()
        npy_path         = self._stage_save(parameters_array)
        meta_path        = self.metadata_manager.save_run_metadata(npy_path, self.tomogram_path, self.height_range)

        metrics_dict = self._stage_metrics(npy_path, meta_path)
        plot_paths   = self._stage_plots(npy_path, meta_path, metrics_dict)

        self.logger.section("[Param Extraction Completed]")

        return {
            "parameters_npy"   : npy_path,
            "metadata"         : meta_path,
            "output_directory" : self.output_directory,
            "source_tomogram"  : self.tomogram_path,
            "plots"            : plot_paths,
        }
