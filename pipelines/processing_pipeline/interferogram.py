from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

from configuration.processing_config import ProcessingConfiguration
from tools.logger                    import Logger
from tools.track_baselines           import BaselineExtractor, TrackBaselines, TrackProfiles


class InterferogramBuilder:
    def __init__(self, config: ProcessingConfiguration, logger: Logger) -> None:
        self.config          = config
        self.logger          = logger
        self.track_baselines = None
        self.track_profiles  = None

        self.logger.section("[InterferogramBuilder Initialization]")
        self.logger.subsection(f"Dataset Type  : {self.config.dataset_type}")
        self.logger.subsection(f"PyRat Threads : {self.config.parallel.interferogram_threads()}")

        pyrat_root = str(self.config.paths.pyrat_directory)
        if pyrat_root not in sys.path:
            sys.path.insert(0, pyrat_root)

    def _build_from_fsar(self, crop_tuple: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from pyrat import pyrat_init, tomo

        self.logger.subsection("Initializing PyRat for FSAR...")
        pyrat_init(debug=False, nthreads=self.config.parallel.interferogram_threads())
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

        self.track_baselines, self.track_profiles = self._extract_baselines([tomography_object.master, *tomography_object.slaves], crop_tuple)

        primary, secondaries, interferograms = self._compute_interferograms(tomography_object)
        self.logger.subsection(f"FSAR stack built — primary: {primary.shape}, secondaries: {secondaries.shape}, interferograms: {interferograms.shape}")
        return primary, secondaries, interferograms

    def _extract_baselines(self, pass_directories: list, crop_tuple: Tuple[int, int, int, int]) -> Tuple[TrackBaselines | None, TrackProfiles | None]:
        try:
            azimuth_window  = (crop_tuple[0], crop_tuple[1])
            extractor       = BaselineExtractor.from_pass_directories([str(p) for p in pass_directories], azimuth_window=azimuth_window)
            table, profiles = extractor.extract_with_profiles()

            self.logger.kv_table(table.describe(), title="Track Baselines")
            return table, profiles

        except Exception as error:
            self.logger.subsection(f"[FSAR] Baseline extraction skipped: {error}")
            return None, None

    def _compute_interferograms(self, tomography_object) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        import pyrat as pyrat_module
        from pyrat import getdata

        raw_options  = tomography_object.project.get("options", "")
        options_list = [opt.split("=") for opt in raw_options.split(",") if opt]
        options      = {opt[0].lower().strip(): True if len(opt) == 1 else opt[1].strip() for opt in options_list}
        suffix       = options.get("suffix", "")

        self.logger.subsection("[FSAR] Loading primary SLC")
        primary_slc  = getdata(
            pyrat_module.load.fsar(
                tomography_object.master,
                product       = "RGI-SLC",
                polarisations = tomography_object.polarisation,
                bands         = tomography_object.band,
                crop          = tomography_object.crop,
                sym           = True,
            )
        )
        primary = primary_slc.astype(np.complex64)

        n_secondaries = len(tomography_object.slaves)

        secondaries    = np.empty((n_secondaries,) + primary.shape, dtype=np.complex64)
        interferograms = np.empty((n_secondaries,) + primary.shape, dtype=np.complex64)

        for secondary_index, secondary in enumerate(tomography_object.slaves):
            self.logger.subsection(f"[FSAR] Loading secondary SLC {secondary_index + 1}/{n_secondaries}: {secondary}")

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

            secondaries[secondary_index] = secondary_slc

            secondary_amplitude   = np.clip(np.abs(secondary_slc), 0.0, self.config.input_configs.max_amplitude_clip)
            deramped_secondary    = secondary_slc * np.exp(1.0j * dem_phase)
            phasor                = primary_slc * np.conj(deramped_secondary)
            phasor               /= (np.abs(phasor) + 1e-30)

            interferograms[secondary_index] = secondary_amplitude * phasor

            del secondary_slc, dem_phase, deramped_secondary, phasor, secondary_amplitude

        return primary, secondaries, interferograms

    def build(self, crop_tuple: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.logger.section(f"[Building Stack] Crop parameters: {crop_tuple}")

        if self.config.dataset_type != "FSAR":
            raise NotImplementedError(f"Dataset type '{self.config.dataset_type}' is not supported. Only 'FSAR' is implemented.")

        primary, secondaries, interferograms = self._build_from_fsar(crop_tuple)

        gc.collect()
        
        return primary, secondaries, interferograms

    def run(
        self,
        crop_tuple           : Tuple[int, int, int, int],
        primary_path         : Path,
        secondaries_path     : Path,
        interferograms_path  : Path,
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        
        primary, secondaries, interferograms = self.build(crop_tuple)

        for path in (primary_path, secondaries_path, interferograms_path):
            path.parent.mkdir(parents=True, exist_ok=True)

        np.save(str(primary_path),        primary,        allow_pickle=False)
        np.save(str(secondaries_path),    secondaries,    allow_pickle=False)
        np.save(str(interferograms_path), interferograms, allow_pickle=False)

        self.logger.subsection(f"Primary saved        : {primary_path}        (shape: {primary.shape})")
        self.logger.subsection(f"Secondaries saved    : {secondaries_path}    (shape: {secondaries.shape})")
        self.logger.subsection(f"Interferograms saved : {interferograms_path} (shape: {interferograms.shape})")

        shapes = tuple(primary.shape), tuple(secondaries.shape), tuple(interferograms.shape)
        del primary, secondaries, interferograms
        gc.collect()
       
        return shapes
