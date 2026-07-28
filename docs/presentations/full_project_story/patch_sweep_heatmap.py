import re
import sys
from pathlib import Path

_vault = next(p for p in Path(__file__).resolve().parents if (p / "code" / "tools" / "logger.py").exists())
sys.path.insert(0, str(_vault / "code"))

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from tools.logger import Logger

plt.style.use("default")


class PatchSweepLossHeatmap:

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir
        self.logger      = Logger(log_dir=str(results_dir / "logs"), name="patch_sweep_heatmap")

        self.reach_az = 52
        self.reach_rg = 24

        self.ramp   = LinearSegmentedColormap.from_list("deck_rose", ["#fdf5f7", "#f2ccd6", "#d97d97", "#b03052", "#701d34", "#38101c"])
        self.ink    = "#28282d"
        self.soft   = "#787880"
        self.good   = "#1e6e46"

        plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})

    def parse_overview(self) -> dict:
        text  = (self.results_dir / "overview.md").read_text()
        cells = {}

        for match in re.finditer(r"-p(\d{3})x(\d{3})`[^|]*\|\s*\d+\s*\|[^|]*\|[^|]*\|\s*([0-9.]+)\s*±\s*([0-9.e-]+)", text):
            az, rg, mean, std = int(match.group(1)), int(match.group(2)), float(match.group(3)), float(match.group(4))
            cells[(az, rg)]   = (mean, std)

        self.logger.info(f"parsed {len(cells)} cells: {sorted(cells)}")
        return cells

    def tie_set(self, cells: dict) -> set:
        best_key         = min(cells, key=lambda k: cells[k][0])
        best_mean, best_std = cells[best_key]

        ties = {key for key, (mean, std) in cells.items() if mean - best_mean <= np.hypot(std, best_std)}

        self.logger.info(f"best cell {best_key} = {best_mean:.5f}; ties inside combined seed noise: {sorted(ties)}")
        return ties

    def render(self, cells: dict) -> plt.Figure:
        az_values = sorted({az for az, _ in cells})
        rg_values = sorted({rg for _, rg in cells})
        step      = az_values[1] - az_values[0]

        grid = np.array([[cells[(az, rg)][0] for az in az_values] for rg in rg_values])

        az_edges = np.array([az - step / 2 for az in az_values] + [az_values[-1] + step / 2])
        rg_edges = np.array([rg - step / 2 for rg in rg_values] + [rg_values[-1] + step / 2])

        figure, axes = plt.subplots(figsize=(5.9, 3.55))
        mesh         = axes.pcolormesh(az_edges, rg_edges, grid, cmap=self.ramp, shading="flat", rasterized=True)

        threshold = grid.min() + 0.55 * (grid.max() - grid.min())
        ties      = self.tie_set(cells)
        best_key  = min(cells, key=lambda k: cells[k][0])

        for (az, rg), (mean, std) in cells.items():
            color    = "white" if mean > threshold else self.ink
            shared   = min(az, self.reach_az) * min(rg, self.reach_rg)
            coverage = 100.0 * shared / (self.reach_az * self.reach_rg)
            outside  = 100.0 * (1.0 - shared / (az * rg))

            axes.annotate(f"{mean:.4f}", xy=(az, rg + 5.2), ha="center", va="center", color=color, fontsize=8.5)
            axes.annotate(rf"$\pm${f'{std:.4f}'.lstrip('0')}", xy=(az, rg + 1.9), ha="center", va="center", color=color, fontsize=6.0, alpha=0.85)
            axes.annotate(f"{coverage:.0f} % of reach", xy=(az, rg - 1.8), ha="center", va="center", color=color, fontsize=6.2, alpha=0.95)
            axes.annotate(f"{outside:.0f} % outside", xy=(az, rg - 5.2), ha="center", va="center", color=color, fontsize=6.2, alpha=0.95)

        for az, rg in ties:
            style = dict(fill=False, edgecolor=self.good, zorder=5)
            width = 1.8 if (az, rg) == best_key else 1.1
            dash  = "-" if (az, rg) == best_key else (0, (4, 2.4))
            axes.add_patch(plt.Rectangle((az - step / 2, rg - step / 2), step, step, linewidth=width, linestyle=dash, **style))

        colorbar = figure.colorbar(mesh, ax=axes, pad=0.02)
        colorbar.set_label("best validation loss (five-seed mean)", fontsize=9)
        colorbar.outline.set_linewidth(0.6)
        colorbar.ax.tick_params(width=0.6, labelsize=8)

        axes.set_xlabel(r"azimuth patch $W_a$ [px]")
        axes.set_ylabel(r"range patch $W_r$ [px]")
        axes.set_xticks(az_values)
        axes.set_yticks(rg_values)
        axes.tick_params(width=0.6, labelsize=9)

        for spine in axes.spines.values():
            spine.set_linewidth(0.6)

        figure.tight_layout(pad=0.4)
        return figure

    def save(self, figure: plt.Figure, name: str) -> Path:
        path = self.results_dir / name

        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)

        self.logger.info(f"saved {path}")
        return path

    def build(self) -> Path:
        cells  = self.parse_overview()
        figure = self.render(cells)
        return self.save(figure, "patch_sweep_loss_heatmap.png")


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[2]
    PatchSweepLossHeatmap(repo / "results" / "Patch").build()
