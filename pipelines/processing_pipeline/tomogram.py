from __future__ import annotations

import gc
import multiprocessing as mp
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np

from configuration.processing_config import ProcessingConfiguration, TomogramConfiguration
from tools.logger                    import Logger


def _run_pyrat(
    pyrat_root_path       : str,
    crop_tuple            : Tuple[int, int, int, int],
    suffix                : str,
    fusar_project_path    : str,
    stack_identifier      : str,
    base_directory        : str,
    polarisation          : str,
    track_selection       : str,
    height_range          : Tuple[float, float],
    filter_method         : str,
    filter_arguments      : dict,
    beamforming_method    : str,
    beamforming_arguments : list,
    output_directory      : str,
    apply_resampling      : bool,
    apply_presumming      : bool,
    pyrat_threads         : int,
) -> int:
    
    if pyrat_root_path not in sys.path:
        sys.path.insert(0, pyrat_root_path)

    from pyrat import pyrat_init, tomo
    pyrat_init(debug=True, nthreads=pyrat_threads, silent=True)

    tomo.fusartomo(
        FuSARproject = fusar_project_path,
        id           = stack_identifier,
        basedir      = base_directory,
        polarisation = polarisation,
        select       = track_selection,
        presum       = apply_presumming,
        crop         = crop_tuple,
        range        = list(height_range),
        filter       = filter_method,
        filargs      = filter_arguments,
        method       = beamforming_method,
        args         = beamforming_arguments,
        suffix       = suffix,
        dir          = output_directory,
        resampling   = apply_resampling,
    )

    gc.collect()
    return 0


class TomogramProcessor:
    def __init__(self, config: ProcessingConfiguration, logger: Logger) -> None:
        self.config = config
        self.logger = logger

        self.logger.section("[TomogramProcessor Initialization]")
        self.logger.subsection(f"Max Azimuth Width : {self.config.input_configs.max_crop_azimuth_width}")
        self.logger.subsection(f"Parallel Workers  : {self.config.parallel.tomogram_workers}")

    def _create_temp(self) -> Path:
        parent = self.config.paths.temporary_directory
        parent.mkdir(parents=True, exist_ok=True)
        temporary_directory = Path(tempfile.mkdtemp(prefix="tomo_", dir=str(parent)))
        return temporary_directory

    def _divide_crop(self, tomogram_config: TomogramConfiguration) -> list[Tuple[int, int, int, int]]:
        crop      = self.config.crop
        max_width = tomogram_config.max_crop_azimuth_width

        azimuth_start = crop.azimuth_start
        azimuth_end   = crop.azimuth_end
        total_width   = azimuth_end - azimuth_start

        if total_width <= max_width:
            self.logger.subsection(f"Crop width ({total_width}) fits within limit ({max_width}). Single section.")
            return [crop.as_tuple()]

        subsections     = []
        current_azimuth = azimuth_start
        
        while current_azimuth < azimuth_end:
            next_azimuth = min(current_azimuth + max_width, azimuth_end)
            subsections.append((current_azimuth, next_azimuth, crop.range_start, crop.range_end))
            current_azimuth = next_azimuth

        self.logger.subsection(f"Crop subdivided into {len(subsections)} sections.")
        
        return subsections

    def _dispatch_workers(
        self,
        subsections         : list[Tuple[int, int, int, int]],
        stack_identifier    : str,
        tomogram_config     : TomogramConfiguration,
        temporary_directory : Path,
    ) -> None:
        parallel_config = self.config.parallel
        tasks           = []

        for subsection_index, subsection_crop in enumerate(subsections):
            tasks.append((
                str(self.config.paths.pyrat_directory),
                subsection_crop,
                f"{subsection_index:04d}",
                tomogram_config.fusar_project_path,
                stack_identifier,
                tomogram_config.base_directory,
                tomogram_config.polarisation,
                tomogram_config.track_selection,
                tomogram_config.height_range,
                tomogram_config.filter_method,
                tomogram_config.filter_arguments,
                tomogram_config.beamforming_method,
                tomogram_config.beamforming_arguments,
                str(temporary_directory),
                tomogram_config.apply_resampling,
                tomogram_config.apply_presumming,
                parallel_config.pyrat_threads,
            ))

        self.logger.subsection(f"Dispatching {len(tasks)} PyRat jobs across {parallel_config.tomogram_workers} workers...")

        with ProcessPoolExecutor(max_workers=parallel_config.tomogram_workers, mp_context=mp.get_context("spawn")) as executor:
            futures = [executor.submit(_run_pyrat, *task) for task in tasks]
            try:
                for future in as_completed(futures):
                    future.result()
            except Exception:
                for f in futures:
                    f.cancel()
                raise

    def _concatenate(self, temporary_directory: Path) -> Tuple[np.ndarray, np.ndarray]:
        partial_files_directory = temporary_directory / "TOMO" / "TOMO-SR"
        partial_file_paths      = sorted(partial_files_directory.iterdir())
   
        self.logger.subsection(f"[Concatenation] Merging {len(partial_file_paths)} subsection artifacts")

        dem_chunks      : list[np.ndarray] = []
        tomogram_chunks : list[np.ndarray] = []

        for partial_file_path in partial_file_paths:
            with h5py.File(str(partial_file_path), "r") as hdf5_file:
                dem_chunks.append(hdf5_file["DEM"][:])
                tomogram_chunks.append(hdf5_file["tomogram"][:])

        combined_dem      = np.concatenate(dem_chunks,      axis=0)
        combined_tomogram = np.concatenate(tomogram_chunks, axis=1)

        self.logger.subsection(f"-> Combined DEM shape      : {combined_dem.shape}")
        self.logger.subsection(f"-> Combined Tomogram shape : {combined_tomogram.shape}")
      
        return combined_dem, combined_tomogram

    def _save(self, tomogram_path: Path, dem_path: Path, tomogram_array: np.ndarray, dem_array: np.ndarray) -> None:
        tomogram_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(tomogram_path), tomogram_array, allow_pickle=False)
        np.save(str(dem_path),      dem_array,      allow_pickle=False)

    def _cleanup_temp(self, temporary_directory: Path) -> None:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory, ignore_errors=True)

    def run(self, tomogram_path: Path, dem_path: Path, stack_identifier: str, tomogram_config: TomogramConfiguration) -> Tuple[Path, Path]:
        self.logger.section(f"[Generating Tomogram]")
        self.logger.subsection(f"Target: {tomogram_path.name}")
        
        temporary_directory = self._create_temp()

        try:
            subsections = self._divide_crop(tomogram_config)
            self._dispatch_workers(subsections, stack_identifier, tomogram_config, temporary_directory)
            combined_dem, combined_tomogram = self._concatenate(temporary_directory)
            self._save(tomogram_path, dem_path, combined_tomogram, combined_dem)
            
            self.logger.subsection(f"Tomogram saved : {tomogram_path}")
            self.logger.subsection(f"DEM saved      : {dem_path}")
        finally:
            self._cleanup_temp(temporary_directory)
            self.logger.subsection("Temporary directory cleaned up. \n")
            gc.collect()

        return tomogram_path, dem_path
