"""Consolidated plotting utilities for TomoSAR."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from scipy.interpolate import interpn
from scipy.ndimage import uniform_filter


@dataclass
class PlotStyle:
    """Centralised plot-style config.  Usable as a context manager."""
    figsize:      tuple[float, float] = (15, 10)
    title_size:   int  = 40
    label_size:   int  = 30
    tick_size:    int  = 30
    legend_size:  int  = 30
    colorbar:     bool = True
    origin:       str  = "lower"
    aspect:       str  = "auto"
    dpi:          int  = 150
    style:        str  = "seaborn-v0_8-darkgrid"
    save_format:  str  = "png"
    tight_layout: bool = True

    def __enter__(self):
        self._token = matplotlib.rcParams.copy()
        plt.style.use(self.style)
        matplotlib.rc("xtick", labelsize=self.tick_size)
        matplotlib.rc("ytick", labelsize=self.tick_size)
        return self

    def __exit__(self, *exc):
        matplotlib.rcParams.update(self._token)
        return False

    def new_figure(self, title: str = "", num: int | None = None) -> Figure:
        fig = plt.figure(num=num, figsize=self.figsize, dpi=self.dpi)
        if title:
            plt.title(title, fontsize=self.title_size)
        return fig

    def decorate(self, ax: plt.Axes | None = None, xlabel: str = "Range",
                 ylabel: str = "Azimuth", add_colorbar: bool | None = None, mappable=None) -> None:
        ax = ax or plt.gca()
        ax.set_xlabel(xlabel, fontsize=self.label_size)
        ax.set_ylabel(ylabel, fontsize=self.label_size)
        if (add_colorbar if add_colorbar is not None else self.colorbar) and mappable is not None:
            plt.colorbar(mappable, ax=ax)

    def save(self, fig: Figure, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), format=self.save_format, bbox_inches="tight")


DEFAULT_STYLE = PlotStyle()


class SignalProcessing:
    """Static SAR signal-processing helpers used by plot functions."""

    @staticmethod
    def lin2db(array: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return 20.0 * np.log10(np.where(array > 0, array, np.finfo(float).tiny))

    @staticmethod
    def smooth(array: np.ndarray, box: int | Sequence[int], phase: bool = False) -> np.ndarray:
        if np.iscomplexobj(array):
            return uniform_filter(array.real, box) + 1j * uniform_filter(array.imag, box)
        if phase:
            return np.angle(SignalProcessing.smooth(np.exp(1j * array), box))
        return uniform_filter(array.real, box)

    @staticmethod
    def coherence(slc1: np.ndarray, slc2: np.ndarray, win: Sequence[int] = (7, 7)) -> np.ndarray:
        sm = SignalProcessing.smooth
        num = sm(slc1 * np.conj(slc2), win)
        den = np.sqrt(sm(slc1 * np.conj(slc1), win) * sm(slc2 * np.conj(slc2), win))
        with np.errstate(divide="ignore", invalid="ignore"):
            coh = np.abs(num / den)
        return np.clip(np.nan_to_num(coh), 0.0, 1.0)

    @staticmethod
    def coherence_angle(slc1: np.ndarray, slc2: np.ndarray, win: Sequence[int] = (5, 5)) -> np.ndarray:
        sm = SignalProcessing.smooth
        num = sm(slc1 * np.conj(slc2), win)
        den = np.sqrt(sm(slc1 * np.conj(slc1), win) * sm(slc2 * np.conj(slc2), win))
        with np.errstate(divide="ignore", invalid="ignore"):
            ang = np.angle(num / den)
        return np.nan_to_num(ang)

    @staticmethod
    def normalize(x: np.ndarray, a: float = 0.0, b: float = 1.0) -> np.ndarray:
        mn, mx = np.amin(x), np.amax(x)
        if mx == mn:
            return np.full_like(x, (a + b) / 2.0)
        return (b - a) * (x - mn) / (mx - mn) + a

    @staticmethod
    def cmap_phase() -> ListedColormap:
        """TAXI-style cyclic phase colormap (256 entries)."""
        r = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,5,7,9,13,15,17,21,25,29,33,37,43,47,53,59,65,73,79,85,91,97,101,107,113,117,121,127,131,135,141,145,151,155,159,163,167,171,173,177,181,185,189,193,197,201,205,209,211,215,217,219,221,223,223,223,223,223,223,221,219,217,215,213,211,207,205,201,199,195,193,191,189,187,185,183,181,179,179,179,179,179,179,181,183,185,187,189,191,193,195,199,201,205,209,213,215,219,223,227,229,233,235,237,239,243,245,245,247,249,251,251,253,253,253,253,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,253,253,251,249,245,243,239,233,229,223,215,209,201,193,185,175,165,155,145,133,121,109,99,87,79,69,61,53,45,37,31,25,21,17,13,9,5,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        g = [249,249,249,247,245,245,243,239,237,235,233,229,227,223,219,215,211,205,201,197,193,189,185,181,177,173,171,167,163,159,155,151,145,141,135,131,127,121,117,113,107,101,97,91,85,79,73,65,59,53,47,43,37,33,29,25,21,17,15,13,9,7,5,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,5,5,7,9,11,13,15,17,19,23,25,29,33,35,41,45,49,53,57,59,63,67,71,75,79,81,85,89,93,97,101,103,107,111,113,117,119,123,125,129,131,135,137,141,145,147,151,155,159,163,167,171,173,177,181,185,189,193,197,201,205,207,211,213,217,217,219,219,219,219,219,217,215,213,211,209,205,203,199,195,193,189,187,185,181,179,179,177,175,175,175,175,177,177,179,181,183,187,189,193,195,199,201,205,207,211,215,217,221,223,227,229,233,235,237,239,241,243,245,245,247,249,249,249,251,251]
        b = [253,253,253,253,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,253,253,251,251,249,247,245,241,239,235,231,227,223,217,211,207,201,193,185,179,169,161,153,145,137,129,121,113,105,99,91,85,77,71,63,57,51,45,39,35,31,27,23,21,17,15,13,9,7,7,5,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,5,7,9,13,15,17,21,25,29,33,37,43,47,53,59,65,73,79,85,91,97,101,107,113,117,121,127,131,135,141,145,151,155,159,163,167,171,173,177,181,185,189,193,197,201,205,211,215,219,223,227,229,233,235,237,239,243,245,245,247,249,251,251]
        rgba = np.column_stack([r, g, b, [255] * 256]).astype(np.float64) / 255.0
        return ListedColormap(rgba, name="taxi_phase")


class ImagePlot:
    """2-D raster SAR image plots.  Every method returns (fig, ax)."""

    @staticmethod
    def amplitude(data: np.ndarray, *, scale: Literal["dB", "linear", "ref"] = "dB",
                  vmin: float | None = None, vmax: float | None = None, title: str = "",
                  cmap: str = "gray", style: PlotStyle = DEFAULT_STYLE,
                  ax: plt.Axes | None = None) -> tuple[Figure, plt.Axes]:
        img = np.abs(data) if np.iscomplexobj(data) else data
        if scale == "dB":
            img = SignalProcessing.lin2db(img)
            vmin = vmin if vmin is not None else -25
            vmax = vmax if vmax is not None else 0
        elif scale == "ref":
            vmin, vmax = 0, 3.0 * np.mean(img)
        else:
            vmin = vmin if vmin is not None else np.nanmin(img)
            vmax = vmax if vmax is not None else np.nanmax(img)
        fig, ax = _ensure_axes(ax, title, style)
        im = ax.imshow(img, origin=style.origin, aspect=style.aspect, cmap=cmap, vmin=vmin, vmax=vmax)
        style.decorate(ax, mappable=im)
        return fig, ax

    @staticmethod
    def phase(data: np.ndarray, *, title: str = "", style: PlotStyle = DEFAULT_STYLE,
              ax: plt.Axes | None = None) -> tuple[Figure, plt.Axes]:
        img = np.angle(data) if np.iscomplexobj(data) else data
        fig, ax = _ensure_axes(ax, title, style)
        im = ax.imshow(img, origin=style.origin, aspect=style.aspect,
                        cmap=SignalProcessing.cmap_phase(), vmin=-np.pi, vmax=np.pi)
        style.decorate(ax, mappable=im)
        return fig, ax

    @staticmethod
    def phase_degrees(data: np.ndarray, *, title: str = "", cmap: str = "gray",
                      vmin: float = -180, vmax: float = 180, style: PlotStyle = DEFAULT_STYLE,
                      ax: plt.Axes | None = None) -> tuple[Figure, plt.Axes]:
        img = np.angle(data, deg=True) if np.iscomplexobj(data) else data
        fig, ax = _ensure_axes(ax, title, style)
        im = ax.imshow(img, origin=style.origin, aspect=style.aspect,
                        cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
        style.decorate(ax, mappable=im)
        return fig, ax

    @staticmethod
    def coherence(slc1: np.ndarray, slc2: np.ndarray, *, win: Sequence[int] = (7, 7),
                  title: str = "", cmap: str = "gray", style: PlotStyle = DEFAULT_STYLE,
                  ax: plt.Axes | None = None) -> tuple[Figure, plt.Axes]:
        img = SignalProcessing.coherence(slc1, slc2, win=win)
        fig, ax = _ensure_axes(ax, title, style)
        im = ax.imshow(img, origin=style.origin, aspect=style.aspect, cmap=cmap, vmin=0.0, vmax=1.0)
        style.decorate(ax, mappable=im)
        return fig, ax

    @staticmethod
    def coherence_angle(slc1: np.ndarray, slc2: np.ndarray, *, win: Sequence[int] = (5, 5),
                        title: str = "", style: PlotStyle = DEFAULT_STYLE,
                        ax: plt.Axes | None = None) -> tuple[Figure, plt.Axes]:
        img = SignalProcessing.coherence_angle(slc1, slc2, win=win)
        fig, ax = _ensure_axes(ax, title, style)
        im = ax.imshow(img, origin=style.origin, aspect=style.aspect, cmap=SignalProcessing.cmap_phase())
        style.decorate(ax, mappable=im)
        return fig, ax

    @staticmethod
    def image(data: np.ndarray, *, title: str = "", cmap: str = "viridis",
              vmin: float | None = None, vmax: float | None = None,
              xlabel: str = "Range", ylabel: str = "Azimuth",
              style: PlotStyle = DEFAULT_STYLE, ax: plt.Axes | None = None) -> tuple[Figure, plt.Axes]:
        fig, ax = _ensure_axes(ax, title, style)
        im = ax.imshow(data, origin=style.origin, aspect=style.aspect, cmap=cmap, vmin=vmin, vmax=vmax)
        style.decorate(ax, xlabel=xlabel, ylabel=ylabel, mappable=im)
        return fig, ax


class HistogramPlot:
    """Amplitude / real / imaginary distribution histograms."""

    @staticmethod
    def histogram(data: np.ndarray, *, label: str = "", mode: Literal["bars", "line", "both"] = "bars",
                  scale: Literal["dB", "linear"] = "linear", num_bins: int = 100,
                  bin_range: tuple[float, float] = (0.0, 5.0), xlim: tuple[float, float] = (-100, 100),
                  xlabel: str = "Value (dB)", alpha: float = 0.5, title: str = "",
                  style: PlotStyle = DEFAULT_STYLE, ax: plt.Axes | None = None) -> tuple[Figure, plt.Axes]:
        img = SignalProcessing.lin2db(data) if scale == "dB" else data
        histo, bins = np.histogram(img.ravel(), bins=num_bins, range=bin_range, density=False)
        width = 0.7 * (bins[1] - bins[0])
        centre = 0.5 * (bins[:-1] + bins[1:])
        fig, ax = _ensure_axes(ax, title, style)
        if mode in ("bars", "both"):
            ax.bar(centre, histo, width=width, alpha=alpha, label=label)
        if mode in ("line", "both"):
            ax.plot(centre, histo, label=label)
        ax.set_ylabel("Frequency", fontsize=style.label_size)
        ax.set_xlabel(xlabel, fontsize=style.label_size)
        ax.set_xlim(xlim)
        ax.legend(loc="upper left", fontsize=style.legend_size)
        ax.grid(True)
        return fig, ax

    @staticmethod
    def compare(original: np.ndarray, estimated: np.ndarray, *,
                label_orig: str = "Original", label_est: str = "Estimated",
                mode: Literal["bars", "line", "both"] = "both",
                scale: Literal["dB", "linear"] = "linear", num_bins: int = 100,
                bin_range: tuple[float, float] = (0.0, 5.0), xlim: tuple[float, float] = (-100, 100),
                xlabel: str = "Value (dB)", title: str = "",
                style: PlotStyle = DEFAULT_STYLE, ax: plt.Axes | None = None) -> tuple[Figure, plt.Axes]:
        fig, ax = HistogramPlot.histogram(original, label=label_orig, mode=mode, scale=scale,
                                          num_bins=num_bins, bin_range=bin_range, xlim=xlim,
                                          xlabel=xlabel, title=title, style=style, ax=ax)
        HistogramPlot.histogram(estimated, label=label_est, mode=mode, scale=scale,
                                num_bins=num_bins, bin_range=bin_range, xlim=xlim,
                                xlabel=xlabel, style=style, ax=ax)
        return fig, ax


class ScatterPlot:
    """2-D comparison plots: scatter, density-coloured scatter, and heatmaps."""

    @staticmethod
    def scatter(original: np.ndarray, estimated: np.ndarray, *,
                scale: Literal["dB", "linear"] = "linear",
                xlim: tuple[float, float] = (-100, 100), ylim: tuple[float, float] = (-100, 100),
                xlabel: str = "Original", ylabel: str = "Estimated", dot_size: float = 0.1,
                diagonal: bool = True, title: str = "", style: PlotStyle = DEFAULT_STYLE,
                ax: plt.Axes | None = None) -> tuple[Figure, plt.Axes]:
        x, y = original.ravel(), estimated.ravel()
        if scale == "dB":
            x, y = SignalProcessing.lin2db(x), SignalProcessing.lin2db(y)
        fig, ax = _ensure_axes(ax, title, style)
        ax.scatter(x, y, c="blue", s=dot_size)
        if diagonal:
            d = np.array([xlim[0] - 50, xlim[1] + 50])
            ax.plot(d, d, ls="dashed", lw=2.0, color="r")
        ax.set_xlabel(xlabel, fontsize=style.label_size)
        ax.set_ylabel(ylabel, fontsize=style.label_size)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        return fig, ax

    @staticmethod
    def density_scatter(original: np.ndarray, estimated: np.ndarray, *,
                        scale: Literal["dB", "linear"] = "linear", bins: Sequence[int] = (30, 30),
                        hist_range: tuple[tuple[float, float], tuple[float, float]] = ((-100, 100), (-100, 100)),
                        xlim: tuple[float, float] = (-100, 100), ylim: tuple[float, float] = (-100, 100),
                        xlabel: str = "Original (dB)", ylabel: str = "Estimated (dB)",
                        dot_size: float = 0.1, cmap: str = "nipy_spectral",
                        vmin: float = 0.0, vmax: float = 5000.0, diagonal: bool = True,
                        title: str = "", style: PlotStyle = DEFAULT_STYLE,
                        ax: plt.Axes | None = None) -> tuple[Figure, plt.Axes]:
        x, y = original.ravel(), estimated.ravel()
        if scale == "dB":
            x, y = SignalProcessing.lin2db(x), SignalProcessing.lin2db(y)
        data, x_e, y_e = np.histogram2d(x, y, bins=bins, range=list(hist_range))
        z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
                    data, np.vstack([x, y]).T, method="splinef2d", bounds_error=False)
        z[np.isnan(z)] = 0.0
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        fig, ax = _ensure_axes(ax, title, style)
        im = ax.scatter(x, y, c=z, s=dot_size, cmap=cmap, vmin=vmin, vmax=vmax)
        if diagonal:
            d = np.array([xlim[0] - 50, xlim[1] + 50])
            ax.plot(d, d, ls="dashed", lw=2.0, color="brown")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel(xlabel, fontsize=style.label_size)
        ax.set_ylabel(ylabel, fontsize=style.label_size)
        plt.colorbar(im, ax=ax)
        return fig, ax

    @staticmethod
    def density_heatmap(original: np.ndarray, estimated: np.ndarray, *,
                        scale: Literal["dB", "linear"] = "linear", bins: Sequence[int] = (30, 30),
                        hist_range: tuple[tuple[float, float], tuple[float, float]] = ((-100, 100), (-100, 100)),
                        xlim: tuple[float, float] = (-100, 100), ylim: tuple[float, float] = (-100, 100),
                        xlabel: str = "Original (dB)", ylabel: str = "Estimated (dB)",
                        cmap: str = "nipy_spectral", diagonal: bool = True, probability: bool = False,
                        title: str = "", style: PlotStyle = DEFAULT_STYLE,
                        ax: plt.Axes | None = None) -> tuple[Figure, plt.Axes]:
        x, y = original.ravel(), estimated.ravel()
        if scale == "dB":
            x, y = SignalProcessing.lin2db(x), SignalProcessing.lin2db(y)
        data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=probability, range=list(hist_range))
        extent = [x_e[0], x_e[-1], y_e[0], y_e[-1]]
        fig, ax = _ensure_axes(ax, title, style)
        im = ax.imshow(data.T, extent=extent, origin="lower", cmap=cmap)
        if diagonal:
            d = np.array([xlim[0] - 50, xlim[1] + 50])
            ax.plot(d, d, ls="dashed", lw=2.0, color="brown")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel(xlabel, fontsize=style.label_size)
        ax.set_ylabel(ylabel, fontsize=style.label_size)
        plt.colorbar(im, ax=ax)
        return fig, ax


class ErrorPlot:
    """Error-analysis plots (amplitude vs phase error, etc.)."""

    @staticmethod
    def amplitude_vs_phase_error(slc_original: np.ndarray, slc_estimated: np.ndarray, *,
                                 mode: Literal["scatter", "heat"] = "heat",
                                 bins: Sequence[int] = (1000, 1000),
                                 xlim: tuple[float, float] = (0, 5),
                                 ylim: tuple[float, float] = (0, np.pi),
                                 hist_range: tuple[tuple[float, float], tuple[float, float]] | None = None,
                                 xlabel: str = "Amplitude", ylabel: str = "Phase error (rad)",
                                 title: str = "Amplitude vs Phase Error",
                                 style: PlotStyle = DEFAULT_STYLE,
                                 ax: plt.Axes | None = None) -> tuple[Figure, plt.Axes]:
        amp = np.abs(slc_estimated).ravel()
        error = np.abs(np.angle(slc_estimated.ravel()) - np.angle(slc_original.ravel()))
        idx = amp.argsort()
        amp, error = amp[idx], error[idx]
        hr = hist_range if hist_range is not None else (xlim, ylim)
        plotter = ScatterPlot.density_heatmap if mode == "heat" else ScatterPlot.density_scatter
        return plotter(amp, error, scale="linear", bins=bins, hist_range=hr,
                       xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel,
                       diagonal=False, title=title, style=style, ax=ax)


class ComparisonReport:
    """Full SLC comparison battery in one call."""

    def __init__(self, original: np.ndarray, estimated: np.ndarray, *,
                 suffix: str = "", output_dir: str | Path = ".", style: PlotStyle = DEFAULT_STYLE):
        self.original  = original
        self.estimated = estimated
        self.suffix    = suffix
        self.output_dir = Path(output_dir)
        self.style     = style
        self.figures: list[tuple[str, Figure]] = []

    def add_amplitude_images(self, scale: str = "dB", vmin: float = -25, vmax: float = 0):
        for data, tag in [(self.original, "Original"), (self.estimated, "Estimated")]:
            fig, _ = ImagePlot.amplitude(data, scale=scale, vmin=vmin, vmax=vmax,
                                         title=f"Amplitude ({tag})", style=self.style)
            self.figures.append((f"amp_{tag.lower()}", fig))
        return self

    def add_phase_images(self):
        for data, tag in [(self.original, "Original"), (self.estimated, "Estimated")]:
            fig, _ = ImagePlot.phase(data, title=f"Phase ({tag})", style=self.style)
            self.figures.append((f"pha_{tag.lower()}", fig))
        return self

    def add_coherence(self, win: Sequence[int] = (7, 7)):
        fig, _ = ImagePlot.coherence(self.original, self.estimated, win=win,
                                     title="Interferometric Coherence", style=self.style)
        self.figures.append(("coherence", fig))
        return self

    def add_coherence_angle(self, win: Sequence[int] = (5, 5)):
        fig, _ = ImagePlot.coherence_angle(self.original, self.estimated, win=win,
                                           title="Coherence Angle", style=self.style)
        self.figures.append(("coherence_angle", fig))
        return self

    def add_amplitude_histograms(self, num_bins: int = 100, bin_range: tuple = (0.0, 5.0), xlim: tuple = (-50, 20)):
        fig, _ = HistogramPlot.compare(np.abs(self.original), np.abs(self.estimated), scale="dB",
                                       num_bins=num_bins, bin_range=bin_range, xlim=xlim,
                                       xlabel="Amplitude (dB)", title="Amplitude Histogram (dB)", style=self.style)
        self.figures.append(("hist_amp_dB", fig))
        return self

    def add_real_histograms(self, num_bins: int = 100, bin_range: tuple = (-2.0, 2.0), xlim: tuple = (-2, 2)):
        o = self.original.real if np.iscomplexobj(self.original) else self.original
        e = self.estimated.real if np.iscomplexobj(self.estimated) else self.estimated
        fig, _ = HistogramPlot.compare(o, e, num_bins=num_bins, bin_range=bin_range, xlim=xlim,
                                       xlabel="Real part", title="Real Part Histogram", style=self.style)
        self.figures.append(("hist_real", fig))
        return self

    def add_imag_histograms(self, num_bins: int = 100, bin_range: tuple = (-2.0, 2.0), xlim: tuple = (-2, 2)):
        o = self.original.imag if np.iscomplexobj(self.original) else self.original
        e = self.estimated.imag if np.iscomplexobj(self.estimated) else self.estimated
        fig, _ = HistogramPlot.compare(o, e, num_bins=num_bins, bin_range=bin_range, xlim=xlim,
                                       xlabel="Imaginary part", title="Imaginary Part Histogram", style=self.style)
        self.figures.append(("hist_imag", fig))
        return self

    def add_density_heatmaps(self, bins: Sequence[int] = (30, 30)):
        amp_o, amp_e = np.abs(self.original), np.abs(self.estimated)
        fig, _ = ScatterPlot.density_heatmap(amp_o, amp_e, scale="dB", bins=bins,
                                             hist_range=((-50, 20), (-50, 20)), xlim=(-50, 20), ylim=(-50, 20),
                                             xlabel="Original (dB)", ylabel="Estimated (dB)",
                                             title="Density - Amplitude (dB)", style=self.style)
        self.figures.append(("density_amp_dB", fig))
        if np.iscomplexobj(self.original):
            for part, tag in [("real", "Real"), ("imag", "Imaginary")]:
                o, e = getattr(self.original, part), getattr(self.estimated, part)
                fig, _ = ScatterPlot.density_heatmap(o, e, bins=bins,
                                                     hist_range=((-2, 2), (-2, 2)), xlim=(-2, 2), ylim=(-2, 2),
                                                     xlabel=f"Original ({tag})", ylabel=f"Estimated ({tag})",
                                                     title=f"Density - {tag} Part", style=self.style)
                self.figures.append((f"density_{part}", fig))
        return self

    def add_error_analysis(self):
        fig, _ = ErrorPlot.amplitude_vs_phase_error(self.original, self.estimated,
                                                    title="Amplitude vs Phase Error", style=self.style)
        self.figures.append(("amp_vs_phase_error", fig))
        return self

    def standard(self):
        return (self.add_amplitude_images().add_phase_images().add_coherence()
                .add_coherence_angle().add_amplitude_histograms()
                .add_real_histograms().add_imag_histograms().add_density_heatmaps())

    def extended(self):
        return self.standard().add_error_analysis()

    def save_all(self) -> list[Path]:
        saved = []
        for name, fig in self.figures:
            fname = f"{name}_{self.suffix}.{self.style.save_format}" if self.suffix else f"{name}.{self.style.save_format}"
            path = self.output_dir / fname
            self.style.save(fig, path)
            saved.append(path)
        return saved

    def show(self):
        plt.show()

    def close_all(self):
        for _, fig in self.figures:
            plt.close(fig)
        self.figures.clear()


class PanelGrid:
    """Multi-panel subplot grid.  Pass grid.ax(r, c) to any plot method's ax parameter."""

    def __init__(self, nrows: int = 1, ncols: int = 1, *, title: str = "", style: PlotStyle = DEFAULT_STYLE):
        self.style = style
        w, h = style.figsize
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=(w * ncols * 0.6, h * nrows * 0.6), squeeze=False)
        if title:
            self.fig.suptitle(title, fontsize=style.title_size)

    def ax(self, row: int, col: int) -> plt.Axes:
        return self.axes[row, col]

    def save(self, path: str | Path):
        self.fig.tight_layout()
        self.style.save(self.fig, path)

    def show(self):
        self.fig.tight_layout()
        plt.show()


def _ensure_axes(ax: plt.Axes | None, title: str, style: PlotStyle) -> tuple[Figure, plt.Axes]:
    if ax is not None:
        fig = ax.get_figure()
        if title:
            ax.set_title(title, fontsize=style.title_size)
        return fig, ax
    fig = style.new_figure(title)
    return fig, fig.gca()
