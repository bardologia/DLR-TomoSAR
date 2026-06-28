from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

import numpy as np

from pipelines.comparison.metric_table           import MetricTableRenderer
from pipelines.comparison.spatial_stats          import SpatialDispersion
from pipelines.processing.param_extraction.metrics import SnrEstimator
from tools.data.io                               import FileIO
from tools.reporting.markdown                    import MarkdownDoc
from tools.reporting.plotting                    import PlotBase
from tools.monitoring.logger                     import Logger


@dataclass
class WindowTrial:
    name           : str
    run_dir        : Path
    tomogram_path  : Path
    window         : tuple
    window_label   : str
    window_area    : int
    metrics        : dict = field(default_factory=dict)


class WindowTrialCollector:
    def __init__(self, runs_dir: Path, run_tags: list[str], logger: Logger) -> None:
        self.runs_dir = runs_dir
        self.run_tags = run_tags
        self.logger   = logger

    def _discover_tags(self) -> list[str]:
        if self.run_tags:
            return list(self.run_tags)

        found = [
            entry.name
            for entry in sorted(self.runs_dir.iterdir())
            if entry.is_dir() and (entry / "data" / "tomogram_full.npy").exists() and (entry / "meta" / "config_state.json").exists()
        ]

        return found

    def _build_trial(self, tag: str) -> WindowTrial:
        run_dir = self.runs_dir / tag
        state   = FileIO.load_json(run_dir / "meta" / "config_state.json")

        window = tuple(state["tomogram_config"]["filter_arguments"]["win"])
        label  = "x".join(str(int(value)) for value in window)
        area   = int(np.prod([int(value) for value in window]))

        return WindowTrial(
            name          = tag,
            run_dir       = run_dir,
            tomogram_path = run_dir / "data" / "tomogram_full.npy",
            window        = window,
            window_label  = label,
            window_area   = area,
        )

    def collect(self) -> list[WindowTrial]:
        self.logger.section("Collecting preprocessing trials")

        trials = []
        for tag in self._discover_tags():
            trial = self._build_trial(tag)
            self.logger.info(f"{trial.name:<48} window {trial.window_label}")
            trials.append(trial)

        if not trials:
            self.logger.error(f"No preprocessing trials found under {self.runs_dir}")

        return sorted(trials, key=lambda item: item.window_area)


class WindowMetrics:

    SPURIOUS_FRACTION = 0.3

    def __init__(self, block_size: int, range_chunk: int, logger: Logger) -> None:
        self.block_size  = block_size
        self.range_chunk = range_chunk
        self.logger      = logger

    def _contrast(self, tomogram: np.ndarray) -> np.ndarray:
        estimator = SnrEstimator(logger=self.logger, range_chunk=self.range_chunk)
        return estimator.run(tomogram)

    def _shape_maps(self, tomogram: np.ndarray) -> tuple:
        elevations, azimuths, ranges = tomogram.shape

        peak_map     = np.zeros((azimuths, ranges), dtype=np.float32)
        spurious_map = np.zeros((azimuths, ranges), dtype=np.float32)

        for start in range(0, ranges, self.range_chunk):
            stop = min(start + self.range_chunk, ranges)
            amp  = np.abs(tomogram[:, :, start:stop]).astype(np.float32)

            peak = amp.max(axis=0)
            peak_map[:, start:stop] = peak

            middle  = amp[1:-1]
            is_peak = (middle > amp[:-2]) & (middle >= amp[2:]) & (middle > self.SPURIOUS_FRACTION * peak[None, :, :])

            spurious_map[:, start:stop] = is_peak.sum(axis=0)

            del amp

        return peak_map, spurious_map

    def _summarise(self, contrast: np.ndarray, peak_map: np.ndarray, spurious_map: np.ndarray) -> dict:
        valid = np.isfinite(contrast)

        return {
            "valid_fraction"        : float(valid.mean()),
            "contrast_db_median"    : float(np.nanmedian(contrast)),
            "contrast_db_p10"       : float(np.nanpercentile(contrast, 10)),
            "contrast_db_p90"       : float(np.nanpercentile(contrast, 90)),
            "spurious_peaks_median" : float(np.median(spurious_map[valid])),
            "speckle_cv_median"     : SpatialDispersion.block_cv(peak_map, self.block_size),
            "autocorr_length_az"    : SpatialDispersion.autocorr_length(peak_map, axis=0),
        }

    def compute(self, trial: WindowTrial) -> dict:
        self.logger.subsection(f"Measuring {trial.name}")

        tomogram = np.load(trial.tomogram_path, mmap_mode="r")

        contrast            = self._contrast(tomogram)
        peak_map, spurious  = self._shape_maps(tomogram)

        return self._summarise(contrast, peak_map, spurious)


class WindowComparisonPlots(PlotBase):

    PANELS = [
        ("contrast_db_median",    "Peak-to-floor contrast (dB)", "Profile sharpness vs window"),
        ("speckle_cv_median",     "Local coefficient of variation", "Residual speckle vs window"),
        ("spurious_peaks_median", "Spurious peaks per profile",   "Competing peaks vs window"),
        ("autocorr_length_az",    "Azimuth correlation length (px)", "Spatial blurring vs window"),
    ]

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir

    def _bar(self, labels: list[str], values: list[float], y_label: str, title: str, path: Path):
        import matplotlib.pyplot as plt

        self._apply_style()

        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.bar(range(len(labels)), values, color="#3b6ea5", width=0.6)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Window (height x width)")
        ax.set_ylabel(y_label)
        ax.set_title(title)

        fig.tight_layout()
        return self._save(fig, path)

    def render(self, trials: list[WindowTrial]) -> list[Path]:
        labels  = [trial.window_label for trial in trials]
        written = []

        for key, y_label, title in self.PANELS:
            values = [trial.metrics[key] for trial in trials]
            path   = self.out_dir / "images" / f"{key}.png"
            written.append(self._bar(labels, values, y_label, title, path))

        return written


class WindowComparisonReport:

    METRIC_COLUMNS = [
        ("valid_fraction",        "Valid frac"),
        ("contrast_db_median",    "Contrast dB"),
        ("speckle_cv_median",     "Speckle CV"),
        ("spurious_peaks_median", "Spurious"),
        ("autocorr_length_az",    "Corr len"),
    ]

    ORIENTATION = {
        "valid_fraction"        : "higher",
        "contrast_db_median"    : "higher",
        "speckle_cv_median"     : "lower",
        "spurious_peaks_median" : "lower",
        "autocorr_length_az"    : "lower",
    }

    def __init__(self, out_dir: Path, logger: Logger) -> None:
        self.out_dir = out_dir
        self.logger  = logger

    def _table(self, doc: MarkdownDoc, trials: list[WindowTrial]) -> None:
        rows = MetricTableRenderer.render(
            rows           = trials,
            leading        = [("Window", lambda trial: trial.window_label)],
            metric_columns = self.METRIC_COLUMNS,
            orientation    = self.ORIENTATION,
        )

        doc.heading("Per-window metrics", level=2)
        doc.raw("\n".join(rows))
        doc.blank()

    def _reading(self, doc: MarkdownDoc) -> None:
        doc.heading("Reading the bias-variance trade-off", level=2)
        doc.paragraph("Arrows mark each axis on its own (↑ better, ↓ better) and the bold value is the per-column extreme, not an overall winner. As the multilook window grows, estimation variance falls: peak-to-floor contrast rises, residual speckle (local CV) drops, and spurious competing peaks per profile decrease. The cost is spatial resolution: the azimuth correlation length grows as neighbouring scatterers are mixed, so the largest window bolds the variance axes while the smallest bolds the correlation length. There is no single best window. Select the knee where variance has settled before the correlation length climbs, keep the two strongest candidates, and break the tie downstream on the end metric with a matched receptive field.")

    def _plots(self, doc: MarkdownDoc, plots: list[Path]) -> None:
        doc.heading("Comparison plots", level=2)
        for path in plots:
            doc.image(path.stem, Path("images") / path.name)

    def write(self, trials: list[WindowTrial], plots: list[Path]) -> Path:
        doc = MarkdownDoc("Preprocessing window comparison")

        doc.paragraph(f"Compared {len(trials)} preprocessing trials differing by multilook window size.")

        self._table(doc, trials)
        if plots:
            self._plots(doc, plots)
        self._reading(doc)

        path = doc.save(self.out_dir / "report.md")
        self.logger.info(f"Report written to: {path}")
        return path


class PreprocessingComparisonPipeline:
    def __init__(self, config, out_dir: Path, logger: Logger) -> None:
        self.config  = config
        self.out_dir = out_dir
        self.logger  = logger

    def _collect(self) -> list[WindowTrial]:
        collector = WindowTrialCollector(Path(self.config.runs_dir), list(self.config.run_tags), self.logger)
        return collector.collect()

    def _measure(self, trials: list[WindowTrial]) -> None:
        metrics = WindowMetrics(self.config.block_size, self.config.range_chunk, self.logger)
        for trial in trials:
            trial.metrics = metrics.compute(trial)

    def _plot(self, trials: list[WindowTrial]) -> list[Path]:
        if not self.config.make_plots:
            return []
        return WindowComparisonPlots(self.out_dir).render(trials)

    def _report(self, trials: list[WindowTrial], plots: list[Path]) -> Path:
        return WindowComparisonReport(self.out_dir, self.logger).write(trials, plots)

    def run(self) -> Path:
        trials = self._collect()
        if not trials:
            raise RuntimeError("No preprocessing trials to compare")

        self._measure(trials)
        plots = self._plot(trials)

        return self._report(trials, plots)
