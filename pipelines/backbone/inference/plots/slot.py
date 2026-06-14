from __future__ import annotations

from pathlib import Path
from typing  import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot    as plt
import numpy                as np
from matplotlib.patches     import Patch

from pipelines.backbone.inference.plots.base import PlotTools


class SlotPlotter(PlotTools):
    def plot_slot_mu_distributions(
        self,
        global_metrics : dict,
        n_gaussians    : int,
        out_dir        : Path,
    ) -> List[Path]:

        slots = list(range(n_gaussians))
        x     = np.arange(n_gaussians)
        width = 0.35

        pred_means = np.array([global_metrics[f"slot_{k}_mu_pred_mean"] for k in slots])
        pred_stds  = np.array([global_metrics[f"slot_{k}_mu_pred_std"]  for k in slots])
        gt_means   = np.array([global_metrics[f"slot_{k}_mu_gt_mean"]   for k in slots])
        gt_stds    = np.array([global_metrics[f"slot_{k}_mu_gt_std"]    for k in slots])

        paths = []

        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        ax.bar(x - width / 2, gt_means,   width, yerr=gt_stds,   color="C0", alpha=0.75, capsize=4, label="GT")
        ax.bar(x + width / 2, pred_means, width, yerr=pred_stds, color="C3", alpha=0.75, capsize=4, label="Pred")
        ax.set_xticks(x)
        ax.set_xticklabels([f"g{k + 1}" for k in slots])
        ax.set_xlabel("Gaussian slot")
        ax.set_ylabel("µ  [m]")
        ax.set_title("Mean µ per slot  (active pixels, mean ± std)")
        ax.legend(framealpha=0.9)
        ax.grid(True, axis="y", which="major", linewidth=0.3, alpha=0.5)
        fig.tight_layout()
        paths.append(self._save(fig, out_dir / "mu_means.png"))

        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        ax.bar(x - width / 2, gt_stds,   width, color="C0", alpha=0.75, label="GT")
        ax.bar(x + width / 2, pred_stds, width, color="C3", alpha=0.75, label="Pred")
        ax.set_xticks(x)
        ax.set_xticklabels([f"g{k + 1}" for k in slots])
        ax.set_xlabel("Gaussian slot")
        ax.set_ylabel("std(µ)  [m]")
        ax.set_title("Std of µ per slot  (spread across pixels)")
        ax.legend(framealpha=0.9)
        ax.grid(True, axis="y", which="major", linewidth=0.3, alpha=0.5)
        fig.tight_layout()
        paths.append(self._save(fig, out_dir / "mu_stds.png"))

        return paths

    def plot_placeholder_detection(
        self,
        global_metrics : dict,
        n_gaussians    : int,
        out_dir        : Path,
    ) -> List[Path]:

        slots  = list(range(n_gaussians))
        labels = [f"g{k + 1}" for k in slots] + ["all"]
        x      = np.arange(len(labels))
        width  = 0.25

        precisions = [global_metrics[f"slot_{k}_placeholder_precision"] for k in slots] + [global_metrics["placeholder_precision"]]
        recalls    = [global_metrics[f"slot_{k}_placeholder_recall"]    for k in slots] + [global_metrics["placeholder_recall"]]
        f1s        = [global_metrics[f"slot_{k}_placeholder_f1"]        for k in slots] + [global_metrics["placeholder_f1"]]
        gt_rates   = [global_metrics[f"slot_{k}_placeholder_gt_rate"]   for k in slots] + [global_metrics.get("placeholder_gt_rate", np.nan)]

        paths = []

        fig, ax = plt.subplots(figsize=(max(5.6, 1.6 * len(labels)), 3.8))
        ax.bar(x - width, precisions, width, color="C0", alpha=0.80, label="Precision")
        ax.bar(x,         recalls,    width, color="C2", alpha=0.80, label="Recall")
        ax.bar(x + width, f1s,        width, color="C3", alpha=0.80, label="F1")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.08)
        ax.set_xlabel("Gaussian slot")
        ax.set_ylabel("score")
        ax.set_title("Inactive-Gaussian detection  (Precision / Recall / F1)")
        ax.legend(framealpha=0.9)
        ax.grid(True, axis="y", which="major", linewidth=0.3, alpha=0.5)
        for xi, (p, r, f) in enumerate(zip(precisions, recalls, f1s)):
            for val, offset in ((p, -width), (r, 0.0), (f, width)):
                if np.isfinite(val):
                    ax.text(xi + offset, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=7)
        fig.tight_layout()
        paths.append(self._save(fig, out_dir / "placeholder_scores.png"))

        fig, ax = plt.subplots(figsize=(max(5.6, 1.2 * len(labels)), 3.8))
        colors  = [f"C{k % 10}" for k in range(len(labels))]
        bars    = ax.bar(x, gt_rates, color=colors, alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.08)
        ax.set_xlabel("Gaussian slot")
        ax.set_ylabel("fraction of pixels")
        ax.set_title("GT placeholder rate per slot")
        ax.grid(True, axis="y", which="major", linewidth=0.3, alpha=0.5)
        for bar, val in zip(bars, gt_rates):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        paths.append(self._save(fig, out_dir / "placeholder_gt_rate.png"))

        return paths

    def plot_slot_ordering_summary(
        self,
        global_metrics : dict,
        n_gaussians    : int,
        out_dir        : Path,
    ) -> List[Path]:

        ordering_rate = float(global_metrics["mu_ordering_rate"])
        dominant_frac = float(global_metrics["permutation_consensus_dominant_frac"])
        identity_frac = float(global_metrics["permutation_consensus_identity_frac"])

        slots        = list(range(n_gaussians))
        active_rates = [1.0 - float(global_metrics[f"slot_{k}_placeholder_gt_rate"]) for k in slots]

        paths = []

        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        labels  = ["µ ordering\nrate", "consensus\ndominant", "consensus\nidentity"]
        values  = [ordering_rate, dominant_frac, identity_frac]
        bars    = ax.barh(labels, values, color=["C0", "C2", "C3"], alpha=0.80)
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("fraction")
        ax.set_title("Slot organisation scalars")
        ax.axvline(1.0, color="black", linewidth=0.7, linestyle="--")
        ax.grid(True, axis="x", which="major", linewidth=0.3, alpha=0.5)
        for bar, val in zip(bars, values):
            if np.isfinite(val):
                ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)
        fig.tight_layout()
        paths.append(self._save(fig, out_dir / "ordering_scalars.png"))

        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        x = np.arange(n_gaussians)
        ax.bar(x, active_rates, color=[f"C{k % 10}" for k in slots], alpha=0.78)
        ax.set_xticks(x)
        ax.set_xticklabels([f"g{k + 1}" for k in slots])
        ax.set_ylim(0, 1.08)
        ax.set_xlabel("Gaussian slot")
        ax.set_ylabel("active-pixel fraction")
        ax.set_title("GT activation rate per slot  (1 − placeholder rate)")
        ax.grid(True, axis="y", which="major", linewidth=0.3, alpha=0.5)
        for xi, val in enumerate(active_rates):
            if np.isfinite(val):
                ax.text(xi, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        paths.append(self._save(fig, out_dir / "activation_rate.png"))

        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        pred_means = np.array([global_metrics[f"slot_{k}_mu_pred_mean"] for k in slots])
        gt_means   = np.array([global_metrics[f"slot_{k}_mu_gt_mean"]   for k in slots])
        pred_stds  = np.array([global_metrics[f"slot_{k}_mu_pred_std"]  for k in slots])
        gt_stds    = np.array([global_metrics[f"slot_{k}_mu_gt_std"]    for k in slots])

        ax.errorbar(slots, gt_means,   yerr=gt_stds,   fmt="o-",  color="C0", capsize=5, linewidth=1.2, label="GT",   markersize=6)
        ax.errorbar(slots, pred_means, yerr=pred_stds, fmt="s--", color="C3", capsize=5, linewidth=1.2, label="Pred", markersize=6)
        ax.set_xticks(slots)
        ax.set_xticklabels([f"g{k + 1}" for k in slots])
        ax.set_xlabel("Gaussian slot")
        ax.set_ylabel("µ  [m]")
        ax.set_title("µ centre per slot  (mean ± std)")
        ax.legend(framealpha=0.9)
        ax.grid(True, which="major", linewidth=0.3, alpha=0.5)
        fig.tight_layout()
        paths.append(self._save(fig, out_dir / "mu_centres.png"))

        return paths

    def plot_active_count_map(
        self,
        params_pred  : np.ndarray,
        params_gt    : np.ndarray,
        n_gaussians  : int,
        out_dir      : Path,
        az_offset    : int,
        rg_offset    : int,
        amp_threshold: float = 1e-3,
    ) -> List[Path]:

        gt_count   = np.zeros(params_gt  .shape[-2:], dtype=np.int32)
        pred_count = np.zeros(params_pred.shape[-2:], dtype=np.int32)

        for k in range(n_gaussians):
            gt_count   += (params_gt  [3 * k] >= amp_threshold).astype(np.int32)
            pred_count += (params_pred[3 * k] >= amp_threshold).astype(np.int32)

        diff = pred_count - gt_count

        H, W = diff.shape
        rgb  = np.zeros((H, W, 3), dtype=np.float32)
        rgb[diff == 0] = [0.20, 0.75, 0.20]
        rgb[diff <  0] = [0.20, 0.45, 0.90]
        rgb[diff >  0] = [0.90, 0.25, 0.25]

        extent = [rg_offset, rg_offset + W, az_offset + H, az_offset]

        n_total   = H * W
        n_correct = int((diff == 0).sum())
        n_under   = int((diff <  0).sum())
        n_over    = int((diff >  0).sum())

        paths = []

        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        ax.imshow(rgb, aspect="auto", interpolation="nearest", extent=extent)
        ax.set_xlabel("range index")
        ax.set_ylabel("azimuth index")
        ax.set_title("Active-count agreement per pixel")

        legend_els = [
            Patch(facecolor=[0.20, 0.75, 0.20], label=f"correct  ({n_correct / n_total * 100:.1f}%)"),
            Patch(facecolor=[0.20, 0.45, 0.90], label=f"under    ({n_under   / n_total * 100:.1f}%)"),
            Patch(facecolor=[0.90, 0.25, 0.25], label=f"over     ({n_over    / n_total * 100:.1f}%)"),
        ]
        ax.legend(handles=legend_els, loc="lower right", framealpha=0.9, fontsize=9)
        fig.tight_layout()
        paths.append(self._save(fig, out_dir / "active_count_agreement.png"))

        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        vabs = max(1, int(np.abs(diff).max()))
        im   = ax.imshow(diff, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto", interpolation="nearest", extent=extent)
        ax.set_xlabel("range index")
        ax.set_ylabel("azimuth index")
        ax.set_title("Signed count difference  (pred − GT)")
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("pred − GT  [#Gaussians]")
        cb.set_ticks(range(-vabs, vabs + 1))
        fig.tight_layout()
        paths.append(self._save(fig, out_dir / "active_count_difference.png"))

        return paths
