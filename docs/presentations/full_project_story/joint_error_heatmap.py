import sys
from pathlib import Path

_repo = next(p for p in Path(__file__).resolve().parents if (p / "tools" / "monitoring" / "logger.py").exists())
sys.path.insert(0, str(_repo))

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from tools.monitoring.logger import Logger

plt.style.use("default")


class JointErrorHeatmap:

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.logger     = Logger(log_dir=str(output_dir / "logs"), name="joint_error_heatmap")

        self.wide_shift = np.linspace(0.0, 4.0, 401)
        self.wide_scale = np.linspace(-1.0, 1.5, 401)
        self.zoom_shift = np.linspace(0.0, 0.5, 401)
        self.zoom_scale = np.linspace(-0.25, 0.25, 401)

        self.ramp   = LinearSegmentedColormap.from_list("deck_rose", ["#fdf5f7", "#f2ccd6", "#d97d97", "#b03052", "#701d34", "#38101c"])
        self.ink    = "#28282d"
        self.soft   = "#787880"
        self.good   = "#1e6e46"
        self.blue   = "#285fa5"

        plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})

    def render_landscape(self) -> plt.Figure:
        surface      = self.compute_surface(self.wide_shift, self.wide_scale)
        figure, axes = plt.subplots(figsize=(5.2, 2.9))
        mesh         = axes.pcolormesh(self.wide_shift, self.wide_scale, surface, cmap=self.ramp, shading="auto", vmin=0.0, rasterized=True)

        contours = axes.contour(self.wide_shift, self.wide_scale, surface, levels=[0.25, 0.5, 1.0, 3.0], colors=self.soft, linewidths=0.6)
        axes.clabel(contours, fmt=lambda v: f"{v:g}$E$", fontsize=7)

        ceiling = axes.contour(self.wide_shift, self.wide_scale, surface, levels=[2.0], colors="white", linewidths=1.5)
        axes.clabel(ceiling, fmt=lambda v: "2$E$", fontsize=8)

        valley = np.exp(-self.wide_shift ** 2 / 4.0) - 1.0
        axes.plot(self.wide_shift, valley, color=self.good, linewidth=1.5, linestyle="--")
        axes.annotate(r"$\partial/\partial\varepsilon = 0$", xy=(1.72, -0.46), color=self.good, fontsize=7.5)

        axes.plot([0.0, 0.0], [-1.0, 1.5], color=self.blue, linewidth=1.4, linestyle="-.", clip_on=False, zorder=4)
        axes.plot([0.0, 4.0], [-1.0, -1.0], color=self.blue, linewidth=1.4, linestyle="-.", clip_on=False, zorder=4)
        axes.annotate(r"$\partial/\partial\delta = 0$  ($1{+}\varepsilon = 0$)", xy=(0.42, -0.955), color=self.blue, fontsize=8)

        axes.plot(0.0, 0.0, marker="o", color=self.good, markersize=4.5, clip_on=False, zorder=5)

        self.decorate(figure, axes, mesh, [0, 1, 2, 3, 4], [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5], 10)
        return figure

    def render_convergence(self) -> plt.Figure:
        surface      = self.compute_surface(self.zoom_shift, self.zoom_scale)
        figure, axes = plt.subplots(figsize=(4.6, 2.6))
        mesh         = axes.pcolormesh(self.zoom_shift, self.zoom_scale, surface, cmap=self.ramp, shading="auto", vmin=0.0, rasterized=True)

        contours = axes.contour(self.zoom_shift, self.zoom_scale, surface, levels=[0.02, 0.05], colors=self.soft, linewidths=0.6)
        axes.clabel(contours, fmt=lambda v: f"{v:g}$E$", fontsize=7)

        outer = axes.contour(self.zoom_shift, self.zoom_scale, surface, levels=[0.1], colors=self.soft, linewidths=0.6)
        axes.clabel(outer, fmt=lambda v: f"{v:g}$E$", fontsize=7, manual=[(0.39, -0.20)])

        valley = np.exp(-self.zoom_shift ** 2 / 4.0) - 1.0
        axes.plot(self.zoom_shift, valley, color=self.good, linewidth=1.5, linestyle="--")

        axes.plot([0.0, 0.0], [-0.25, 0.25], color=self.blue, linewidth=1.4, linestyle="-.", clip_on=False, zorder=4)

        axes.plot(0.0, 0.0, marker="o", color=self.good, markersize=4.5, clip_on=False, zorder=5)

        axes.annotate(r"$\approx\; E\,(\varepsilon^2 + \delta^2/2\sigma^2)$", xy=(0.16, 0.185), color=self.ink, fontsize=8.5)

        self.decorate(figure, axes, mesh, [0, 0.25, 0.5], [-0.25, 0.0, 0.25], 8.5)
        return figure

    def render_gradient(self) -> plt.Figure:
        surface      = self.compute_surface(self.zoom_shift, self.zoom_scale)
        figure, axes = plt.subplots(figsize=(4.6, 2.6))
        mesh         = axes.pcolormesh(self.zoom_shift, self.zoom_scale, surface, cmap=self.ramp, shading="auto", vmin=0.0, rasterized=True)

        shift, scale = np.meshgrid(np.linspace(0.035, 0.485, 10), np.linspace(-0.22, 0.22, 9))
        overlap      = np.exp(-shift ** 2 / 4.0)
        pull_shift   = -(1.0 + scale) * shift * overlap
        pull_scale   = -2.0 * (1.0 + scale - overlap)

        axes.quiver(shift, scale, pull_shift, pull_scale, angles="xy", scale_units="xy", scale=16.0, width=0.0042, color=self.ink, alpha=0.85, zorder=4)
        axes.annotate(r"$-\nabla$ (descent)", xy=(0.025, 0.205), color=self.ink, fontsize=8.5, zorder=6, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.2))

        axes.plot(0.0, 0.0, marker="o", color=self.good, markersize=4.5, clip_on=False, zorder=5)

        self.decorate(figure, axes, mesh, [0, 0.25, 0.5], [-0.25, 0.0, 0.25], 8.5)
        return figure

    def compute_surface(self, shift_grid: np.ndarray, scale_grid: np.ndarray) -> np.ndarray:
        shift, scale = np.meshgrid(shift_grid, scale_grid)
        overlap      = np.exp(-shift ** 2 / 4.0)
        surface      = (1.0 + scale) ** 2 + 1.0 - 2.0 * (1.0 + scale) * overlap

        self.logger.info(f"surface range: {surface.min():.4f}E to {surface.max():.4f}E")
        return surface

    def decorate(self, figure: plt.Figure, axes: plt.Axes, mesh, xticks: list, yticks: list, colorbar_fontsize: float) -> None:
        colorbar = figure.colorbar(mesh, ax=axes, pad=0.02)
        colorbar.set_label(r"$\|(1+\varepsilon)\,\mathcal{N}_{\mu+\delta}-\mathcal{N}_{\mu}\|^{2}\;/\;E$", fontsize=colorbar_fontsize)
        colorbar.outline.set_linewidth(0.6)
        colorbar.ax.tick_params(width=0.6, labelsize=8)

        axes.set_xlabel(r"misplacement $\delta/\sigma$ (widths)")
        axes.set_ylabel(r"relative scale error $\varepsilon$")
        axes.set_xticks(xticks)
        axes.set_yticks(yticks)
        axes.tick_params(width=0.6, labelsize=8)

        for spine in axes.spines.values():
            spine.set_linewidth(0.6)

    def save(self, figure: plt.Figure, name: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / name

        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)

        self.logger.info(f"saved {path}")
        return path

    def build(self) -> list:
        landscape   = self.render_landscape()
        convergence = self.render_convergence()
        gradient    = self.render_gradient()

        paths = [self.save(landscape, "joint_error_heatmap.png"), self.save(convergence, "joint_error_convergence.png"), self.save(gradient, "joint_error_gradient.png")]
        return paths


class ProfileVsGroundTruth:

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.logger     = Logger(log_dir=str(output_dir / "logs"), name="profile_vs_ground_truth")

        self.z = np.linspace(-3.5, 4.5, 601)

        self.ink    = "#28282d"
        self.accent = "#b03052"

        plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})

    def gaussian(self, mu: float, amplitude: float = 1.0, sigma: float = 1.0) -> np.ndarray:
        return amplitude * np.exp(-(self.z - mu) ** 2 / (2.0 * sigma ** 2))

    def render_shift_profile(self) -> plt.Figure:
        truth        = self.gaussian(mu=0.0)
        pred         = self.gaussian(mu=1.4)
        figure, axes = plt.subplots(figsize=(2.1, 1.65))

        self.decorate(figure, axes, truth, pred, "shift only", y_max=1.15, gt_xy=(0.05, 1.03), pred_xy=(2.4, 0.78))
        return figure

    def render_scale_profile(self) -> plt.Figure:
        truth        = self.gaussian(mu=0.0)
        pred         = self.gaussian(mu=0.0, amplitude=1.55)
        figure, axes = plt.subplots(figsize=(2.1, 1.65))

        self.decorate(figure, axes, truth, pred, "scale only", y_max=1.75, gt_xy=(-0.7, 0.55), pred_xy=(1.0, 1.30))
        return figure

    def decorate(self, figure: plt.Figure, axes: plt.Axes, truth: np.ndarray, pred: np.ndarray, title: str, y_max: float, gt_xy: tuple, pred_xy: tuple) -> None:
        axes.fill_between(self.z, truth, pred, color=self.accent, alpha=0.28, linewidth=0, interpolate=True)
        axes.plot(self.z, truth, color=self.ink, linewidth=1.6)
        axes.plot(self.z, pred, color=self.accent, linewidth=1.6, linestyle="--")

        axes.annotate("GT", xy=gt_xy, color=self.ink, fontsize=7.5)
        axes.annotate("pred", xy=pred_xy, color=self.accent, fontsize=7.5)

        axes.set_title(title, fontsize=8.5, color=self.ink)
        axes.set_xlabel("$z$", fontsize=8)
        axes.set_ylabel("$a$", fontsize=8)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_ylim(0.0, y_max)

        for spine in axes.spines.values():
            spine.set_linewidth(0.6)

        figure.tight_layout(pad=0.3)

    def save(self, figure: plt.Figure, name: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / name

        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)

        self.logger.info(f"saved {path}")
        return path

    def build(self) -> list:
        shift_profile = self.render_shift_profile()
        scale_profile = self.render_scale_profile()

        paths = [self.save(shift_profile, "shift_only_profile.png"), self.save(scale_profile, "scale_only_profile.png")]
        return paths


if __name__ == "__main__":
    output_dir = _repo / "results" / "error_landscape"

    JointErrorHeatmap(output_dir).build()
    ProfileVsGroundTruth(output_dir).build()
