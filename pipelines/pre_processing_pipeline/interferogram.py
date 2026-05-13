from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

from configuration.preprocessing_config import PreProcessingConfiguration
from tools.logger                                          import Logger


class InterferogramBuilder:
    def __init__(self, config: PreProcessingConfiguration, logger: Logger) -> None:
        self.config = config
        self.logger = logger

        self.logger.section("[InterferogramBuilder Initialization]")
        self.logger.subsection(f"Dataset Type  : {self.config.dataset_type}")
        self.logger.subsection(f"PyRat Threads : {self.config.parallel.pyrat_threads}")

        pyrat_root = str(self.config.paths.pyrat_directory)
        if pyrat_root not in sys.path:
            sys.path.insert(0, pyrat_root)

    def _build_from_uavsar(self, crop_tuple: Tuple[int, int, int, int]) -> np.ndarray:
        from pyrat import getdata, load

        loading_parameters = {
            "dir"     : self.config.input_configs.fusar_project_path,
            "band"    : "L",
            "polar"   : self.config.input_configs.polarisation,
            "tracks"  : self.config.input_configs.track_selection,
            "crop"    : list(crop_tuple),
            "product" : "SLC",
            "sym"     : False,
        }

        self.logger.subsection("Loading UAVSAR layer...")
        loaded_layer = load.uavsar(**loading_parameters)
        data_array   = np.stack(getdata(loaded_layer), axis=0)

        self.logger.subsection(f"UAVSAR stack built. Final shape: {data_array.shape}")
        return data_array

    def _build_from_fsar(self, crop_tuple: Tuple[int, int, int, int]) -> np.ndarray:
        from pyrat import pyrat_init, tomo

        self.logger.subsection("Initializing PyRat for FSAR...")
        pyrat_init(debug=False, nthreads=self.config.parallel.pyrat_threads)
        tomogram_config = self.config.input_configs

        tomography_object = tomo.FuSARtomo(
            FuSARproject = tomogram_config.fusar_project_path,
            select       = tomogram_config.track_selection,
            id           = self.config.reduced_stack_identifier,
            basedir      = tomogram_config.base_directory,
            polarisation = tomogram_config.polarisation,
            crop         = list(crop_tuple),
        )

        self.logger.subsection(f"[FSAR] Master: {tomography_object.master}")
        self.logger.subsection(f"[FSAR] Slaves ({len(tomography_object.slaves)}):")
        for slave_index, slave in enumerate(tomography_object.slaves, start=1):
            self.logger.subsection(f"  - [{slave_index}] {slave}")

        data_array = self._compute_interferograms(tomography_object)
        self.logger.subsection(f"FSAR stack built. Final shape: {data_array.shape}")
        return data_array

    def _compute_interferograms(self, tomography_object) -> np.ndarray:
        import pyrat as pyrat_module
        from pyrat import getdata

        raw_options  = tomography_object.project.get("options", "")
        options_list = [opt.split("=") for opt in raw_options.split(",") if opt]
        options      = {opt[0].lower().strip(): True if len(opt) == 1 else opt[1].strip() for opt in options_list}
        suffix       = options.get("suffix", "")

        self.logger.subsection("[FSAR] Loading primary SLC...")
        primary_slc = getdata(
            pyrat_module.load.fsar(
                tomography_object.master,
                product       = "RGI-SLC",
                polarisations = tomography_object.polarisation,
                bands         = tomography_object.band,
                crop          = tomography_object.crop,
                sym           = True,
            )
        )

        master_layer = primary_slc.astype(np.complex64)

        stack_layers : list[np.ndarray] = [master_layer]
        num_slaves                       = len(tomography_object.slaves)

        for secondary_index, secondary in enumerate(tomography_object.slaves):
            self.logger.subsection(f"[FSAR] Loading secondary SLC {secondary_index + 1}/{num_slaves}: {secondary}")

            secondary_slc = getdata(
                pyrat_module.load.fsar(
                    secondary,
                    product       = "INF-SLC",
                    polarisations = tomography_object.polarisation,
                    bands         = tomography_object.band,
                    crop          = tomography_object.crop,
                    suffix        = suffix,
                    sym           = True,
                )
            )

            dem_phase = getdata(
                pyrat_module.load.fsar_phadem(
                    secondary,
                    bands  = tomography_object.band,
                    crop   = tomography_object.crop,
                    suffix = suffix,
                )
            )

            secondary_amplitude   = np.clip(np.abs(secondary_slc), 0.0, 1.25)
            deramped_secondary    = secondary_slc * np.exp(1.0j * dem_phase)
            phasor                = primary_slc * np.conj(deramped_secondary)
            phasor               /= (np.abs(phasor) + 1e-30)
            complex_interferogram = (secondary_amplitude * phasor).astype(np.complex64)

            stack_layers.append(complex_interferogram)
            del secondary_slc, dem_phase, deramped_secondary, phasor, secondary_amplitude, complex_interferogram

        return np.stack(stack_layers, axis=0)

    def build(self, crop_tuple: Tuple[int, int, int, int]) -> np.ndarray:
        self.logger.section(f"[Building Stack] Crop parameters: {crop_tuple}")

        if self.config.dataset_type == "UAVSAR":
            stack = self._build_from_uavsar(crop_tuple)
        else:
            stack = self._build_from_fsar(crop_tuple)

        gc.collect()
        return stack

    def run(self, crop_tuple: Tuple[int, int, int, int], output_path: Path) -> Tuple[int, ...]:
        stack = self.build(crop_tuple)
        stack = np.ascontiguousarray(stack)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(output_path), stack, allow_pickle=False)
        self.logger.subsection(f"Inputs saved: {output_path} (shape: {stack.shape})")

        shape = tuple(stack.shape)
        del stack
        gc.collect()
        return shape
