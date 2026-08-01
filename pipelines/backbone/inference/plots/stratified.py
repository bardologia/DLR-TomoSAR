from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from pipelines.backbone.inference.plots.base import PlotTools


class StratifiedErrorPlotter(PlotTools):

    COVARIATE_LABELS = {
        "gt_active_count"   : "GT active Gaussian count",
        "primary_amplitude" : "Primary SLC amplitude",
        "coherence"         : "Mean coherence across secondaries",
        "dem_slope"         : "DEM slope magnitude [m/px]",
        "label_r2"          : "Label fit R²",
    }

    def plot_error_curve(self, rows: list[dict], covariate: str, out_path: Path, discrete: bool) -> Path:
        self._apply_style()

        centers = [row["center"] for row in rows]
        medians = [row["median"] for row in rows]
        q25     = [row["q25"] for row in rows]
        q75     = [row["q75"] for row in rows]

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))
        ax.fill_between(centers, q25, q75, alpha=0.25, color="#0072B2", linewidth=0)
        ax.plot(centers, medians, marker="o", color="#0072B2", linewidth=1.4)

        ax.set_xlabel(self.COVARIATE_LABELS.get(covariate, covariate))
        ax.set_ylabel("Pixel curve MSE (median, IQR band)")
        ax.set_yscale("log")
        ax.set_title(f"Error stratified by {self.COVARIATE_LABELS.get(covariate, covariate).lower()}")
        if discrete:
            ax.set_xticks(centers)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._save(fig, out_path)
