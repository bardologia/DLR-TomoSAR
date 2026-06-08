from __future__ import annotations

import gc
from pathlib import Path
from typing  import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy             as np

from configuration.processing_config import ProcessingConfiguration
from pipelines.shared.io             import FileIO
from pipelines.shared.plotting       import PlotBase
from tools.logger                    import Logger


class StackPlotter(PlotBase):
    PHASE_TICKS  = [-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi]
    PHASE_LABELS = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]

    def __init__(self, config: ProcessingConfiguration, logger: Logger, fig_dpi: int = 150, save_dpi: int = 300) -> None:
        self.config      = config
        self.logger      = logger
        self.fig_dpi     = fig_dpi
        self.save_dpi    = save_dpi
        self.images_directory = Path(config.paths.run_directory) / "images"

    def _setup_output_dirs(self) -> Dict[str, Path]:
        dirs = {
            "slc"            : self.images_directory / "slc",
            "interferograms" : self.images_directory / "interferograms",
            "dem"            : self.images_directory / "dem",
        }
        FileIO.ensure_dirs(*dirs.values())
        return dirs

    @staticmethod
    def _amplitude_db(data: np.ndarray) -> np.ndarray:
        amplitude = np.abs(data).astype(np.float32)
        return 20.0 * np.log10(np.maximum(amplitude, 1e-12))

    def _plot_amplitude(self, amplitude_db: np.ndarray, title: str, out_path: Path) -> Path:
        Az, R      = amplitude_db.shape
        vmin, vmax = self._shared_clim(amplitude_db)

        fig, ax = plt.subplots(figsize=(8, 6))
        im      = ax.imshow(amplitude_db, cmap="gray", vmin=vmin, vmax=vmax, extent=[0, R, Az, 0], aspect="auto", interpolation="nearest")
        ax.set_xlabel("range [px]")
        ax.set_ylabel("azimuth [px]")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("amplitude [dB]")
        fig.tight_layout()

        return self._save(fig, out_path)

    def _plot_linear_amplitude(self, amplitude: np.ndarray, title: str, cbar_label: str, out_path: Path) -> Path:
        Az, R      = amplitude.shape
        vmin, vmax = self._shared_clim(amplitude)

        fig, ax = plt.subplots(figsize=(8, 6))
        im      = ax.imshow(amplitude, cmap="gray", vmin=vmin, vmax=vmax, extent=[0, R, Az, 0], aspect="auto", interpolation="nearest")
        ax.set_xlabel("range [px]")
        ax.set_ylabel("azimuth [px]")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label(cbar_label)
        fig.tight_layout()

        return self._save(fig, out_path)

    def _plot_phase(self, phase: np.ndarray, title: str, out_path: Path) -> Path:
        Az, R = phase.shape

        fig, ax = plt.subplots(figsize=(8, 6))
        im      = ax.imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi, extent=[0, R, Az, 0], aspect="auto", interpolation="nearest")
        ax.set_xlabel("range [px]")
        ax.set_ylabel("azimuth [px]")
        ax.set_title(title)

        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=self.PHASE_TICKS)
        cb.set_label("interferometric phase [rad]")
        cb.ax.set_yticklabels(self.PHASE_LABELS)
        fig.tight_layout()

        return self._save(fig, out_path)

    def _plot_interferogram(self, interferogram: np.ndarray, title: str, out_dir: Path, stem: str) -> Dict[str, Path]:
        clip      = float(self.config.tomogram_config.max_amplitude_clip)
        amplitude = np.abs(interferogram).astype(np.float32)
        phase     = np.angle(interferogram).astype(np.float32)

        return {
            "amplitude" : self._plot_linear_amplitude(amplitude, f"{title} — secondary SLC amplitude (clipped at {clip:g})", f"secondary SLC amplitude (clipped at {clip:g})", out_dir / f"{stem}_amplitude.png"),
            "phase"     : self._plot_phase(phase,                f"{title} — flattened phase",                              out_dir / f"{stem}_phase.png"),
        }

    def _plot_dem(self, dem: np.ndarray, title: str, out_path: Path) -> Path:
        Az, R      = dem.shape
        vmin, vmax = self._shared_clim(dem)
        cmap_obj   = self._cmap_with_bad("terrain")

        fig, ax = plt.subplots(figsize=(8, 6))
        im      = ax.imshow(dem, cmap=cmap_obj, vmin=vmin, vmax=vmax, extent=[0, R, Az, 0], aspect="auto", interpolation="nearest")
        ax.set_xlabel("range [px]")
        ax.set_ylabel("azimuth [px]")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("height [m]")
        fig.tight_layout()

        return self._save(fig, out_path)

    def run(
        self,
        primary_path        : Path,
        secondaries_path    : Path,
        interferograms_path : Path,
        dem_path            : Path,
        pass_labels         : Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        self.logger.section("[Stack Overview Plots]")
        self._apply_style()

        dirs  = self._setup_output_dirs()
        saved : Dict[str, Path] = {}

        primary       = np.load(str(primary_path), mmap_mode="r")
        primary_label = str(pass_labels[0]) if pass_labels else "primary"

        self.logger.subsection(f"Plotting primary SLC {tuple(primary.shape)} — {primary_label}")
        saved["primary"] = self._plot_amplitude(self._amplitude_db(np.asarray(primary)), f"Primary SLC amplitude — {primary_label}", dirs["slc"] / "primary.png")

        del primary
        gc.collect()

        secondaries   = np.load(str(secondaries_path), mmap_mode="r")
        n_secondaries = secondaries.shape[0]

        for index in range(n_secondaries):
            label = str(pass_labels[index + 1]) if pass_labels else f"pass_{index + 1:02d}"

            self.logger.subsection(f"Plotting secondary SLC {index + 1}/{n_secondaries} — {label}")
            saved[f"secondary_{index:02d}"] = self._plot_amplitude(self._amplitude_db(np.asarray(secondaries[index])), f"Secondary SLC amplitude — {label}", dirs["slc"] / f"secondary_{index + 1:02d}_{label}.png")

            gc.collect()

        del secondaries
        gc.collect()

        interferograms   = np.load(str(interferograms_path), mmap_mode="r")
        n_interferograms = interferograms.shape[0]

        for index in range(n_interferograms):
            label = str(pass_labels[index + 1]) if pass_labels else f"pass_{index + 1:02d}"

            self.logger.subsection(f"Plotting interferogram {index + 1}/{n_interferograms} — {label}")

            outputs = self._plot_interferogram(np.asarray(interferograms[index]), f"Interferogram — {primary_label} / {label}", dirs["interferograms"], f"interferogram_{index + 1:02d}_{label}")

            for kind, path in outputs.items():
                saved[f"interferogram_{index:02d}_{kind}"] = path

            gc.collect()

        del interferograms
        gc.collect()

        dem = np.asarray(np.load(str(dem_path), mmap_mode="r"), dtype=np.float32)

        self.logger.subsection(f"Plotting full DEM {tuple(dem.shape)}")
        saved["dem_full"] = self._plot_dem(dem, "DEM full", dirs["dem"] / "dem_full.png")

        del dem
        gc.collect()

        self.logger.subsection(f"Saved {len(saved)} figures → {self.images_directory}")
        return saved
