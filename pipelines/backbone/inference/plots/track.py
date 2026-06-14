from __future__ import annotations

from pathlib import Path
from typing  import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot    as plt
import numpy                as np

from pipelines.backbone.inference.plots.base import PlotTools


class TrackPlotter(PlotTools):
    def plot_track_geometry(self, baselines, out_path: Path) -> Path:
        fig, ax = plt.subplots(figsize=(5.4, 4.6))

        for index, label in enumerate(baselines.labels):
            color  = "black" if index == 0 else f"C{(index - 1) % 10}"
            marker = "*" if index == 0 else "o"
            size   = 140 if index == 0 else 60
            ax.scatter(baselines.horizontal[index], baselines.vertical[index], s=size, marker=marker, color=color, zorder=3)
            ax.annotate(label, (baselines.horizontal[index], baselines.vertical[index]), textcoords="offset points", xytext=(7, 5), fontsize=9)
            ax.errorbar(baselines.horizontal[index], baselines.vertical[index], xerr=baselines.horizontal_std[index], yerr=baselines.vertical_std[index], fmt="none", ecolor=color, elinewidth=0.7, capsize=2, alpha=0.6)

        ax.set_xlabel(r"horizontal baseline $b_{\perp,\mathrm{h}}$ [m]")
        ax.set_ylabel(r"vertical baseline $b_{\perp,\mathrm{v}}$ [m]")
        ax.set_title(f"Passes used  (reference {baselines.reference}, mean over azimuth window)")
        ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
        fig.tight_layout()

        return self._save(fig, out_path)

    def plot_track_profiles(self, profiles, out_dir: Path, split_azimuth: Optional[Tuple[int, int]] = None) -> List[Path]:
        azimuth = profiles.azimuth_axis
        paths   = []

        for component, data, symbol in (
            ("horizontal", profiles.relative_to_reference("horizontal"), r"$b_{\perp,\mathrm{h}}$"),
            ("vertical",   profiles.relative_to_reference("vertical"),   r"$b_{\perp,\mathrm{v}}$"),
        ):
            fig, ax = plt.subplots(figsize=(7.2, 3.6))

            for index, label in enumerate(profiles.labels):
                color = "black" if index == 0 else f"C{(index - 1) % 10}"
                ax.plot(azimuth, data[index], color=color, linewidth=1.0, label=label)

            if split_azimuth is not None:
                ax.axvspan(split_azimuth[0], split_azimuth[1], color="C7", alpha=0.18, label="inference split")

            ax.set_xlabel("azimuth sample index")
            ax.set_ylabel(f"{symbol} relative to {profiles.labels[0]} [m]")
            ax.set_title(f"Per-azimuth {component} baselines of the passes used")
            ax.legend(framealpha=0.9, fontsize=8, ncol=2)
            ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
            fig.tight_layout()

            paths.append(self._save(fig, out_dir / f"baseline_profiles_{component}.png"))

        return paths

    def plot_track_flight_3d(self, profiles, out_path: Path, elev: float = 28.0, azim: float = -55.0) -> Path:
        azimuth = profiles.azimuth_axis
        radii   = profiles.deviation_radii()
        h_mean  = np.nanmean(profiles.horizontal, axis=1)
        v_mean  = np.nanmean(profiles.vertical,   axis=1)

        step  = max(1, len(azimuth) // 200)
        theta = np.linspace(0.0, 2.0 * np.pi, 36)
        az_grid, th_grid = np.meshgrid(azimuth[::step], theta)

        fig = plt.figure(figsize=(12.0, 7.0))
        ax  = fig.add_axes([0.12, 0.08, 0.70, 0.84], projection="3d")

        for index, label in enumerate(profiles.labels):
            color = "black" if index == 0 else f"C{(index - 1) % 10}"

            ax.plot(azimuth, profiles.horizontal[index], profiles.vertical[index], color=color, linewidth=1.3, label=f"{label}  (RMS dev {radii[index]:.2f} m)", zorder=3)
            ax.plot_surface(az_grid, h_mean[index] + radii[index] * np.cos(th_grid), v_mean[index] + radii[index] * np.sin(th_grid), color=color, alpha=0.16, linewidth=0, antialiased=False, shade=False)

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("azimuth sample index", labelpad=14)
        ax.set_ylabel(r"horizontal baseline $b_{\perp,\mathrm{h}}$ [m]", labelpad=14)
        ax.set_zlabel(r"vertical baseline $b_{\perp,\mathrm{v}}$ [m]",   labelpad=18)
        ax.tick_params(axis="z", pad=8)
        ax.set_title("Flight tracks of the passes used  (tube radius = RMS planar deviation over azimuth)")
        ax.legend(loc="upper left", framealpha=0.9, fontsize=8)

        return self._save(fig, out_path)
