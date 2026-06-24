from __future__ import annotations

from pathlib import Path
from typing  import List, Optional

import matplotlib.pyplot as plt
import numpy             as np
import torch

from tools.reporting.plotting import PlotBase

from configuration.diagnostics.weight_xray_config import WeightXrayConfig


class WeightXrayPlots(PlotBase):
    def __init__(self, config: WeightXrayConfig) -> None:
        self.config     = config
        self.output_dir = config.plots_directory

    def _series_plot(self, values: List[float], y_label: str, title: str, filename: str, log_y: bool = False, threshold: Optional[float] = None) -> Optional[Path]:
        points = [(index, value) for index, value in enumerate(values) if value is not None]
        if not points:
            return None

        xs = [index for index, _ in points]
        ys = [value for _, value in points]

        self._apply_style()
        fig, ax = plt.subplots(figsize=(7.4, 3.8))

        ax.plot(xs, ys, color="#1f4e79", linewidth=1.0, marker="o", markersize=2.6, markerfacecolor="#1f4e79", markeredgecolor="none")

        if threshold is not None:
            ax.axhline(threshold, color="#c0392b", linewidth=0.9, linestyle="--", label="threshold")
            ax.legend(loc="best", frameon=False)

        if log_y:
            ax.set_yscale("log")

        ax.set_xlabel("Tensor index (forward order)")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.margins(x=0.01)

        return self._save(fig, self.output_dir / filename)

    def _histogram_plot(self, values: np.ndarray, title: str, filename: str) -> Optional[Path]:
        if values.size == 0:
            return None

        self._apply_style()
        fig, ax = plt.subplots(figsize=(6.6, 4.0))

        ax.hist(values, bins=160, color="#1f4e79", edgecolor="none")
        ax.set_yscale("log")

        ax.set_xlabel("Weight value")
        ax.set_ylabel("Count (log scale)")
        ax.set_title(title)

        return self._save(fig, self.output_dir / filename)

    def _global_sample(self, reports, state_dict: dict) -> np.ndarray:
        chunks = []
        for report in reports:
            if report.role != "weight":
                continue

            flat   = state_dict[report.name].detach().to(torch.float32).cpu().numpy().reshape(-1)
            finite = flat[np.isfinite(flat)]
            chunks.append(finite)

        if not chunks:
            return np.empty(0, dtype=np.float32)

        pooled = np.concatenate(chunks)
        if pooled.size > self.config.histogram_sample_max:
            index  = np.linspace(0, pooled.size - 1, self.config.histogram_sample_max).astype(np.int64)
            pooled = pooled[index]

        return pooled

    def _flagged_histograms(self, reports, state_dict: dict) -> List[Path]:
        flagged = [report for report in reports if report.role == "weight" and report.severity in ("warning", "critical")]
        flagged = flagged[: self.config.max_layer_histograms]

        paths = []
        for report in flagged:
            flat   = state_dict[report.name].detach().to(torch.float32).cpu().numpy().reshape(-1)
            finite = flat[np.isfinite(flat)]
            safe   = report.name.replace(".", "_").replace("/", "_")

            path = self._histogram_plot(finite, f"{report.name}  ({report.severity})", str(Path("layer_histograms") / f"{safe}.png"))
            if path is not None:
                paths.append(path)

        return paths

    def render(self, reports, state_dict: dict) -> List[Path]:
        weight_reports = [report for report in reports if report.role == "weight"]
        thresholds     = self.config.thresholds

        produced = [
            self._series_plot([report.l2_norm   for report in weight_reports], "L2 norm",              "Per-tensor weight L2 norm",        "layer_l2_norm.png",   log_y=True),
            self._series_plot([report.std       for report in reports],        "Standard deviation",   "Per-tensor weight dispersion",     "layer_std.png",       log_y=True),
            self._series_plot([report.frac_dead for report in reports],        "Dead fraction",        "Per-tensor dead-weight fraction",  "layer_sparsity.png",  threshold=thresholds.dead_fraction_warn),
            self._series_plot([report.rank_ratio for report in weight_reports], "Effective rank ratio", "Per-tensor effective rank ratio",  "layer_rank_ratio.png", threshold=thresholds.rank_ratio_warn),
            self._histogram_plot(self._global_sample(reports, state_dict), "Pooled weight distribution", "weight_histogram.png"),
        ]

        produced = [path for path in produced if path is not None]
        produced.extend(self._flagged_histograms(reports, state_dict))
        return produced
