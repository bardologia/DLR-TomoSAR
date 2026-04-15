import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from STEtools.ste_io import rrat

OUTPUT_DIR    = Path("/ste/rnd/User/vice_vi/Pruebas/tracks")
AZIMUTH_SLICE = slice(5010, 5290)


class BaselineProcessor:
    def __init__(self, track_paths, downsample_factor=4, azimuth_slice=AZIMUTH_SLICE):
        self.track_paths = track_paths
        self.factor = downsample_factor
        self.az_slice = azimuth_slice
        self.vertical = {}    # row index 3
        self.horizontal = {}  # row index 2

    @staticmethod
    def _downsample(x, factor):
        xp = np.r_[x, np.nan + np.zeros((-len(x) % factor,))]
        return np.nanmean(xp.reshape(-1, factor), axis=-1)

    def load(self):
        for label, path in self.track_paths.items():
            raw = rrat(path)
            self.vertical[label] = self._downsample(raw[3, :], self.factor)[self.az_slice]
            self.horizontal[label] = self._downsample(raw[2, :], self.factor)[self.az_slice]
        self.n_samples = len(next(iter(self.vertical.values())))
        return self


class BaselinePlotter:
    COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    def __init__(self, processor: BaselineProcessor):
        self.proc = processor
        self._apply_rc()

    @staticmethod
    def _apply_rc():
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.grid": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "mathtext.fontset": "cm",
        })

    def _azimuth_axis(self):
        return np.arange(self.proc.n_samples)

    def _build_filename(self, plot_type, ext="png"):
        passes = "_".join(self.proc.vertical.keys())
        return f"17SARTOM_FL01_{passes}_{plot_type}.{ext}"

    def _save(self, fig, plot_type, pad_inches=0.1):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fname = self._build_filename(plot_type)
        fig.savefig(OUTPUT_DIR / fname, bbox_inches="tight", pad_inches=pad_inches)
        print(f"Saved: {OUTPUT_DIR / fname}")

    def plot_3d(self, elev=28, azim=-55):
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_axes([0.15, 0.1, 0.65, 0.8], projection="3d")

        azimuth = self._azimuth_axis()
        labels = list(self.proc.vertical.keys())

        for i, label in enumerate(labels):
            v = self.proc.vertical[label]
            h = self.proc.horizontal[label]
            ax.plot(azimuth, h, v, color=self.COLORS[i % len(self.COLORS)], linewidth=1.4, label=label)

        ax.set_xlabel(r"Azimuth sample index", labelpad=14)
        ax.set_ylabel(r"Horizontal baseline $b_{\perp,\mathrm{h}}$ [m]", labelpad=14)
        ax.set_zlabel(r"Vertical baseline $b_{\perp,\mathrm{v}}$ [m]", labelpad=24)
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(axis="z", pad=10)
        ax.legend(loc="upper left", framealpha=0.9)
        self._save(fig, "baselines_3D", pad_inches=0.8)
        return fig, ax

    def plot_projections(self):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
        azimuth = self._azimuth_axis()
        labels = list(self.proc.vertical.keys())

        for i, label in enumerate(labels):
            c = self.COLORS[i % len(self.COLORS)]
            axes[0].plot(azimuth, self.proc.vertical[label], color=c, linewidth=1.2, label=label)
            axes[1].plot(azimuth, self.proc.horizontal[label], color=c, linewidth=1.2, label=label)

        axes[0].set_xlabel(r"Azimuth sample index")
        axes[0].set_ylabel(r"$b_{\perp,\mathrm{v}}$ [m]")
        axes[0].set_title("Vertical Baselines")
        axes[0].legend(framealpha=0.9)

        axes[1].set_xlabel(r"Azimuth sample index")
        axes[1].set_ylabel(r"$b_{\perp,\mathrm{h}}$ [m]")
        axes[1].set_title("Horizontal Baselines")
        axes[1].legend(framealpha=0.9)

        self._save(fig, "baselines_vert_horiz")
        return fig, axes

    def plot_baseline_scatter(self):
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        labels = list(self.proc.vertical.keys())

        for i, label in enumerate(labels):
            ax.scatter(self.proc.horizontal[label], self.proc.vertical[label], s=4, alpha=0.6, color=self.COLORS[i % len(self.COLORS)], label=label)

        ax.set_xlabel(r"$b_{\perp,\mathrm{h}}$ [m]")
        ax.set_ylabel(r"$b_{\perp,\mathrm{v}}$ [m]")
        ax.set_title("Baseline Distribution")
        ax.legend(markerscale=3, framealpha=0.9)
        ax.set_aspect("equal", adjustable="datalim")
        self._save(fig, "baselines_scatter")
        return fig, ax


if __name__ == "__main__":
    track_paths = {
        "PS03": "/ste/rnd/17SARTOM/FL01/PS03/T01L/INF/INF-TRACK/track_sar_resa_17sartom0103_Lhh_t01L.rat",
        "PS07": "/ste/rnd/17SARTOM/FL01/PS07/T01L/INF/INF-TRACK/track_sar_resa_17sartom0107_Lhh_t01L.rat",
        "PS11": "/ste/rnd/17SARTOM/FL01/PS11/T01L/INF/INF-TRACK/track_sar_resa_17sartom0111_Lhh_t01L.rat",
        "PS15": "/ste/rnd/17SARTOM/FL01/PS15/T01L/INF/INF-TRACK/track_sar_resa_17sartom0115_Lhh_t01L.rat",
    }

    processor = BaselineProcessor(track_paths).load()
    plotter = BaselinePlotter(processor)

    plotter.plot_3d()
    plotter.plot_projections()
    plotter.plot_baseline_scatter()

    plt.show()
