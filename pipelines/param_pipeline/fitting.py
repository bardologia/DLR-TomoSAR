from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Tuple
import numpy as np

from configuration.param_extraction_config  import FitSettings, FitMode
from tools.logger                           import Logger
from pipelines.param_pipeline.sigma_fitting import SigmaFittingExtractor


class ParameterExtractor:
    def __init__(
        self,
        parameter_extraction : FitSettings,
        parameter_workers    : int,
        logger               : Logger,
        use_gpu              : bool                = True,
        gpu_batch_size       : int                 = 256,
        adam_steps           : int                 = 800,
        adam_lr              : float               = 1e-2,
        adam_b1              : float               = 0.9,
        adam_b2              : float               = 0.999,
        adam_warmup_steps    : int                 = 100,
        gpu_device_ids       : list | None         = None,
        r2_sample_cap        : int                 = 4096,
        gpu_pixel_batch_size : int                 = 8192,
        init_workers         : int | None          = None,
    ) -> None:
       
        self.parameter_extraction = parameter_extraction
        self.parameter_workers    = parameter_workers
        self.logger               = logger
        self.use_gpu              = use_gpu
        self.gpu_batch_size       = gpu_batch_size
        self.gpu_pixel_batch_size = gpu_pixel_batch_size
        self.adam_steps           = adam_steps
        self.adam_lr              = adam_lr
        self.adam_b1              = adam_b1
        self.adam_b2              = adam_b2
        self.adam_warmup_steps    = adam_warmup_steps
        self.gpu_device_ids       = gpu_device_ids
        self.init_workers         = init_workers

        fit_cfg = parameter_extraction.fit_config
        
        k_max           = fit_cfg.k_max
        lambda_k        = fit_cfg.lambda_k
        prominence_frac = fit_cfg.prominence_frac

        self._gpu_extractor = SigmaFittingExtractor(
            fit_settings         = parameter_extraction,
            logger               = logger,
            range_batch_size     = gpu_batch_size,
            adam_steps           = adam_steps,
            adam_lr              = adam_lr,
            adam_b1              = adam_b1,
            adam_b2              = adam_b2,
            k_max                = k_max,
            lambda_k             = lambda_k,
            prominence_frac      = prominence_frac,
            gpu_pixel_batch_size = gpu_pixel_batch_size,
            r2_sample_cap        = r2_sample_cap,
            gpu_device_ids       = gpu_device_ids,
        )

        self.logger.section("[Parameter Extractor Initialized]")
        self.logger.subsection(f"Backend : JAX GPU (Sigma Only)")
        self.logger.subsection(f"Method  : {self.parameter_extraction.fitting_method}")

    @staticmethod
    def _sort_gaussians_by_mu(parameters_array: np.ndarray, n_gaussians: int) -> np.ndarray:
        n_params, Az, R = parameters_array.shape
        out             = parameters_array.copy()
        mu_rows         = np.array([1 + 3 * g for g in range(n_gaussians)])
        
        for ai in range(Az):
            for ri in range(R):
                mus   = parameters_array[mu_rows, ai, ri]
                order = np.argsort(mus)
                
                for new_pos, old_pos in enumerate(order):
                    out[new_pos * 3 + 0, ai, ri] = parameters_array[old_pos * 3 + 0, ai, ri]
                    out[new_pos * 3 + 1, ai, ri] = parameters_array[old_pos * 3 + 1, ai, ri]
                    out[new_pos * 3 + 2, ai, ri] = parameters_array[old_pos * 3 + 2, ai, ri]
        
        return out

    def run(self, tomogram_path: Path, height_range: Tuple[float, float]) -> np.ndarray:
        self.logger.section(f"[Extraction Start] Source: {tomogram_path.name}")

        parameters_array = self._gpu_extractor.run(tomogram_path, height_range)
        parameters_array = self._sort_gaussians_by_mu(parameters_array, self.parameter_extraction.fit_config.k_max)
        
        self.logger.subsection("[Extraction Complete]")
        return parameters_array
