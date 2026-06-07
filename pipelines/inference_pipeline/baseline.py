from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from tools.logger import Logger


class PreprocessConfigReader:
    def __init__(self, preprocessing_dir: Path) -> None:
        self.preprocessing_dir = Path(preprocessing_dir)

    def read(self) -> dict:
        meta_dir   = self.preprocessing_dir / "meta"
        candidates = sorted(meta_dir.glob("config_state_*.json"))

        if not candidates:
            raise FileNotFoundError(f"No config_state_*.json under {meta_dir}; cannot read the pre-processing Capon configuration")

        state    = json.loads(candidates[0].read_text(encoding="utf-8"))
        tomo_cfg = state["tomogram_config"]

        return {
            "window"                : tuple(int(v) for v in tomo_cfg["filter_arguments"]["win"]),
            "filter_method"         : str(tomo_cfg["filter_method"]),
            "beamforming_method"    : str(tomo_cfg["beamforming_method"]),
            "beamforming_arguments" : list(tomo_cfg["beamforming_arguments"]),
            "source"                : str(candidates[0]),
        }


class GeometryLoader:
    TRAINER_CONFIG = Path("docs") / "trainer_config.json"

    def __init__(self, run_directory: Path, logger: Logger) -> None:
        self.run_directory = Path(run_directory)
        self.logger        = logger

    def load_kz(self) -> np.ndarray:
        payload  = json.loads((self.run_directory / self.TRAINER_CONFIG).read_text(encoding="utf-8"))
        geometry = payload["geometry"]

        if geometry["kz_values"]:
            kz = np.asarray([float(v) for v in geometry["kz_values"]], dtype=np.float64)
        else:
            scale = 4.0 * math.pi / (float(geometry["wavelength"]) * float(geometry["slant_range"]))
            kz    = scale * np.asarray([float(b) for b in geometry["baselines"]], dtype=np.float64)

        self.logger.subsection(f"Geometry origin : {geometry['baselines_origin']}")
        self.logger.subsection(f"kz [rad/m]      : {', '.join(f'{v:.4f}' for v in kz)}")

        return kz


class CovarianceEstimator:
    def __init__(self, window: tuple) -> None:
        self.window = (int(window[0]), int(window[1]))

    def estimate(self, stack: np.ndarray) -> np.ndarray:
        n_tracks, height, width = stack.shape
        covariance              = np.empty((n_tracks, n_tracks, height, width), dtype=np.complex64)

        for i in range(n_tracks):
            for j in range(i, n_tracks):
                product = stack[i] * np.conj(stack[j])
                real    = uniform_filter(product.real.astype(np.float32), size=self.window, mode="nearest")
                imag    = uniform_filter(product.imag.astype(np.float32), size=self.window, mode="nearest")

                covariance[i, j] = real + 1j * imag
                if i != j:
                    covariance[j, i] = real - 1j * imag

        return covariance


class CaponSpectrum:
    def __init__(self, kz: np.ndarray, x_axis: np.ndarray, loading: float, phase_sign: float, chunk_rows: int = 64) -> None:
        self.kz         = np.asarray(kz, dtype=np.float64)
        self.x_axis     = np.asarray(x_axis, dtype=np.float64)
        self.loading    = float(loading)
        self.phase_sign = float(phase_sign)
        self.chunk_rows = int(chunk_rows)

        phase         = self.phase_sign * self.kz.reshape(-1, 1) * self.x_axis.reshape(1, -1)
        self.steering = np.exp(1j * phase).astype(np.complex64)

    def compute(self, covariance: np.ndarray) -> np.ndarray:
        n_tracks, _, height, width = covariance.shape
        n_elevation                = self.x_axis.size
        spectrum                   = np.empty((n_elevation, height, width), dtype=np.float32)
        eye                        = np.eye(n_tracks, dtype=np.complex64)

        for row_start in range(0, height, self.chunk_rows):
            row_end = min(row_start + self.chunk_rows, height)

            cov   = covariance[:, :, row_start:row_end, :].transpose(2, 3, 0, 1).astype(np.complex64)
            trace = np.einsum("hwii->hw", cov).real / n_tracks
            cov   = cov + (self.loading * np.clip(trace, 1e-12, None))[..., None, None] * eye

            inverse = np.linalg.inv(cov)
            denom   = np.einsum("ik,hwij,jk->hwk", self.steering.conj(), inverse, self.steering).real

            spectrum[:, row_start:row_end, :] = (1.0 / np.clip(denom, 1e-12, None)).transpose(2, 0, 1)

        return spectrum


class ClassicalBaseline:
    def __init__(
        self,
        run_directory     : Path,
        logger            : Logger,
        *,
        preprocessing_dir : Path | None  = None,
        window            : tuple | None = None,
        loading           : float        = 1e-2,
        phase_sign        : float        = 1.0,
        chunk_rows        : int          = 64,
    ) -> None:
        self.logger            = logger
        self.geometry          = GeometryLoader(run_directory, logger)
        self.preprocessing_dir = preprocessing_dir
        self.window            = window
        self.loading           = loading
        self.phase_sign        = phase_sign
        self.chunk_rows        = chunk_rows

    def _resolve_window(self) -> tuple:
        if self.window is not None:
            self.logger.subsection(f"Window origin   : explicit override {tuple(self.window)}")
            return tuple(self.window)

        if self.preprocessing_dir is None:
            raise ValueError("Either an explicit capon_window or the preprocessing_dir is required to resolve the covariance window")

        preprocess = PreprocessConfigReader(self.preprocessing_dir).read()

        self.logger.subsection(f"Window origin   : pre-processing config {preprocess['source']}")
        if preprocess["beamforming_method"].lower() != "capon":
            self.logger.subsection(f"Note            : full tomogram used '{preprocess['beamforming_method']}'; the baseline is always Capon")
        if preprocess["beamforming_arguments"]:
            self.logger.subsection(f"Note            : pre-processing beamforming arguments {preprocess['beamforming_arguments']} are PyRAT-internal and not replicated")

        return preprocess["window"]

    def _build_stack(self, complex_inputs: np.ndarray, n_secondaries: int) -> np.ndarray:
        n_tracks = 1 + n_secondaries
        stack    = np.empty((n_tracks,) + complex_inputs.shape[1:], dtype=np.complex64)

        stack[0]  = np.abs(complex_inputs[0]).astype(np.float32)
        stack[1:] = complex_inputs[1 + n_secondaries:1 + 2 * n_secondaries]

        return stack

    @staticmethod
    def normalize_per_pixel(spectrum: np.ndarray) -> np.ndarray:
        peak = spectrum.max(axis=0, keepdims=True)
        safe = np.where(peak > 1e-12, peak, 1.0)

        return (spectrum / safe).astype(np.float32)

    def compute(self, complex_inputs: np.ndarray, n_secondaries: int, x_axis: np.ndarray, secondary_labels=None) -> np.ndarray:
        self.logger.section("[Classical Capon Baseline]")

        window = self._resolve_window()

        self.logger.kv_table({
            "Passes"      : f"primary + {n_secondaries} secondaries" + (f"  ({', '.join(secondary_labels)})" if secondary_labels else ""),
            "Window"      : str(window),
            "Loading"     : self.loading,
            "Phase sign"  : self.phase_sign,
            "Elevations"  : int(np.asarray(x_axis).size),
        })

        kz = self.geometry.load_kz()
        if kz.size != 1 + n_secondaries:
            raise ValueError(f"Geometry provides {kz.size} kz values but the stack has {1 + n_secondaries} tracks; the training geometry must match the selected passes")

        stack      = self._build_stack(complex_inputs, n_secondaries)
        covariance = CovarianceEstimator(window).estimate(stack)
        spectrum   = CaponSpectrum(kz, x_axis, self.loading, self.phase_sign, self.chunk_rows).compute(covariance)

        reduced = self.normalize_per_pixel(spectrum)

        self.logger.subsection(f"Reduced tomogram : {reduced.shape}  ({reduced.nbytes / 1e9:.2f} GB)")

        return reduced
