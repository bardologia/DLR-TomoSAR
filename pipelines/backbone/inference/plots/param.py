from __future__ import annotations

from pathlib import Path
from typing  import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot    as plt
import numpy                as np

from pipelines.backbone.inference.plots.base import PlotTools
from tools.metrics.gaussian_matching          import GaussianMatcher


class ParamPlotter(PlotTools):
    def plot_param_maps(
        self,
        params_pred : np.ndarray,
        params_gt   : Optional[np.ndarray],
        n_gaussians : int,
        out_dir     : Path,
        az_offset   : int,
        rg_offset   : int,
    ) -> List[Path]:

        H, W   = params_pred.shape[-2:]
        extent = [rg_offset, rg_offset + W, az_offset + H, az_offset]
        paths  = []

        for k in range(n_gaussians):
            for j, (fname, short) in enumerate(zip(("a", "mu", "sigma"), self.PARAM_SHORT)):
                ch       = 3 * k + j
                arr_pred = params_pred[ch]
                vmin, vmax = self._shared_clim(arr_pred if params_gt is None else np.stack([arr_pred, params_gt[ch]]))

                paths.append(self._imshow_panel(
                    data       = arr_pred,
                    title      = f"Pred {short} — g{k + 1}",
                    x_label    = "range index",
                    y_label    = "azimuth index",
                    cbar_label = short,
                    extent     = extent,
                    cmap       = "jet",
                    vmin       = vmin,
                    vmax       = vmax,
                    origin     = "upper",
                    path       = out_dir / f"g{k + 1}_{fname}_pred.png",
                ))

                if params_gt is not None and ch < params_gt.shape[0]:
                    paths.append(self._imshow_panel(
                        data       = params_gt[ch],
                        title      = f"GT {short} — g{k + 1}",
                        x_label    = "range index",
                        y_label    = "azimuth index",
                        cbar_label = short,
                        extent     = extent,
                        cmap       = "jet",
                        vmin       = vmin,
                        vmax       = vmax,
                        origin     = "upper",
                        path       = out_dir / f"g{k + 1}_{fname}_gt.png",
                    ))

        return paths

    def plot_param_distributions(
        self,
        params_pred : np.ndarray,
        params_gt   : Optional[np.ndarray],
        n_gaussians : int,
        out_dir     : Path,
        bins        : int = 80,
    ) -> List[Path]:

        matched     = params_gt is not None
        pred_source = GaussianMatcher().aligned_prediction(params_pred, params_gt, n_gaussians) if matched else params_pred
        paths       = []

        for k in range(n_gaussians):
            amp_ch = 3 * k
            if matched and amp_ch < params_gt.shape[0]:
                gt_amp_flat = params_gt[amp_ch].reshape(-1)
                active_mask = np.isfinite(gt_amp_flat) & (gt_amp_flat >= 1e-3)
            else:
                active_mask = None

            for j, (fname, lbl) in enumerate(self.PARAM_LABELS):
                ch    = 3 * k + j
                short = self.PARAM_SHORT[j]

                pred_flat = pred_source[ch].reshape(-1)
                if active_mask is not None:
                    pred_flat = pred_flat[active_mask]
                pred = pred_flat[np.isfinite(pred_flat)]

                gt = np.empty(0, dtype=np.float64)
                if params_gt is not None and ch < params_gt.shape[0]:
                    gt_flat = params_gt[ch].reshape(-1)
                    if active_mask is not None:
                        gt_flat = gt_flat[active_mask]
                    gt = gt_flat[np.isfinite(gt_flat)]

                has_pred = pred.size > 0
                has_gt   = gt.size > 0
                if not has_pred and not has_gt:
                    continue

                combined = np.concatenate([arr for arr in (pred, gt) if arr.size])
                is_amp   = j == 0

                if is_amp:
                    positive  = combined[combined > 0]
                    if positive.size == 0:
                        continue
                    lo        = max(float(np.percentile(positive, 0.5)), 1e-6)
                    hi        = float(positive.max()) * 1.02
                    bin_edges = np.geomspace(lo, hi, bins + 1)
                else:
                    lo        = float(np.percentile(combined, 0.5))
                    hi        = float(np.percentile(combined, 99.5))
                    bin_edges = np.linspace(lo, hi, bins + 1)

                fig, ax = plt.subplots(figsize=(4.8, 3.4))

                if has_gt:
                    ax.hist(gt, bins=bin_edges, density=True, color="C0", alpha=0.55, label="GT", edgecolor="none")
                    ax.axvline(float(np.median(gt)), color="C0", linestyle="--", linewidth=0.9, label=f"med GT={np.median(gt):.3g}")

                if has_pred:
                    pred_label = "Matched pred" if matched else "Pred"
                    ax.hist(pred, bins=bin_edges, density=True, color="C3", alpha=0.55, label=pred_label, edgecolor="none")
                    ax.axvline(float(np.median(pred)), color="C3", linestyle="--", linewidth=0.9, label=f"med {pred_label}={np.median(pred):.3g}")

                tag   = "GT g" if matched else "g"
                scope = "matched, active" if matched else "pred only"

                if is_amp:
                    ax.set_xscale("log")
                    ax.set_title(f"{tag}{k + 1} — {lbl}  ({scope}, max={float(combined.max()):.3g})", fontsize=10)
                else:
                    ax.set_title(f"{tag}{k + 1} — {lbl}  ({scope})", fontsize=10)

                ax.set_xlabel(short)
                ax.set_ylabel("density")
                ax.legend(fontsize=7, framealpha=0.9)
                ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
                fig.tight_layout()

                paths.append(self._save(fig, out_dir / f"g{k + 1}_{fname}.png"))

        return paths

    @staticmethod
    def _point_density(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
        bins = int(np.clip(np.sqrt(gt.size / 2.0), 16, 128))

        counts, gt_edges, pred_edges = np.histogram2d(gt, pred, bins=bins)
        gt_bin   = np.clip(np.digitize(gt,   gt_edges)   - 1, 0, bins - 1)
        pred_bin = np.clip(np.digitize(pred, pred_edges) - 1, 0, bins - 1)

        return counts[gt_bin, pred_bin]

    def _scatter_panel(self, ax, gt: np.ndarray, pred: np.ndarray, label: str):
        r2      = self._r2_value(gt, pred)
        density = self._point_density(gt, pred)
        order   = np.argsort(density)

        return ax.scatter(
            gt[order], pred[order], c=density[order], s=4, alpha=0.7,
            cmap="viridis", edgecolors="none", rasterized=True, label=f"{label}  R²={r2:.3f}",
        )

    @staticmethod
    def _robust_square_limits(*arrays: np.ndarray, q: float = 0.5) -> tuple:
        flat = np.concatenate([arr for arr in arrays])
        lo   = float(np.percentile(flat, q))
        hi   = float(np.percentile(flat, 100.0 - q))
        pad  = (hi - lo) * 0.04 if hi > lo else 1.0

        return lo - pad, hi + pad

    @staticmethod
    def _identity_line(ax, lo: float, hi: float) -> None:
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.9, linestyle="--", label="identity")

    def plot_param_scatter(
        self,
        params_pred : np.ndarray,
        params_gt   : np.ndarray,
        n_gaussians : int,
        out_dir     : Path,
        max_points  : int = 8_000,
        seed        : int = 0,
    ) -> List[Path]:

        aligned = GaussianMatcher().aligned_prediction(params_pred, params_gt, n_gaussians)
        paths   = []

        for k in range(n_gaussians):
            gt_amp_flat = params_gt[3 * k].reshape(-1)
            is_active   = np.isfinite(gt_amp_flat) & (gt_amp_flat >= 1e-3)

            for j, (fname, lbl) in enumerate(self.PARAM_LABELS):
                ch    = 3 * k + j
                short = self.PARAM_SHORT[j]

                if ch >= params_gt.shape[0] or ch >= aligned.shape[0]:
                    continue

                gt_all   = params_gt[ch].reshape(-1)
                pred_all = aligned  [ch].reshape(-1)
                matched  = is_active & np.isfinite(pred_all)

                gt, pred = self._paired_subsample([gt_all[matched], pred_all[matched]], max_points, seed)
                if gt.size == 0:
                    continue

                fig, ax = plt.subplots(figsize=(5.2, 4.4))
                sc      = self._scatter_panel(ax, gt, pred, "matched")
                lo, hi  = self._robust_square_limits(gt, pred)
                self._identity_line(ax, lo, hi)

                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.set_aspect("equal", adjustable="box")

                fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02).set_label("point density (count per 2-D bin)")

                r2_str = self._r2_value(gt, pred)
                ax.set_title(f"GT g{k + 1} — {lbl}  (matched, R²={r2_str:.3f})", fontsize=10)
                ax.set_xlabel(f"GT {short}")
                ax.set_ylabel(f"Matched pred {short}")
                ax.legend(fontsize=7, framealpha=0.9)
                ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
                fig.tight_layout()

                paths.append(self._save(fig, out_dir / f"g{k + 1}_{fname}.png"))

        return paths

    @staticmethod
    def _r2_value(gt: np.ndarray, pred: np.ndarray) -> float:
        ss_res = float(np.sum((gt - pred) ** 2))
        ss_tot = float(np.sum((gt - np.mean(gt)) ** 2))

        return 1.0 - ss_res / (ss_tot + 1e-12)

    def plot_param_error_maps(
        self,
        params_pred : np.ndarray,
        params_gt   : np.ndarray,
        n_gaussians : int,
        out_dir     : Path,
        az_offset   : int,
        rg_offset   : int,
    ) -> List[Path]:

        aligned = GaussianMatcher().aligned_prediction(params_pred, params_gt, n_gaussians)
        H, W    = params_pred.shape[-2:]
        extent  = [rg_offset, rg_offset + W, az_offset + H, az_offset]
        paths   = []

        for k in range(n_gaussians):
            gt_active = params_gt[3 * k] >= 1e-3

            for j, (fname, _) in enumerate(self.PARAM_LABELS):
                ch    = 3 * k + j
                short = self.PARAM_SHORT[j]

                if ch >= params_gt.shape[0] or ch >= aligned.shape[0]:
                    continue

                err       = np.where(gt_active, np.abs(aligned[ch] - params_gt[ch]), np.nan).astype(np.float32)
                valid_err = err[np.isfinite(err)]
                if valid_err.size == 0:
                    continue

                vmax = float(np.percentile(valid_err, 99.0))

                paths.append(self._imshow_panel(
                    data       = err,
                    title      = f"|Δ{short}| — GT g{k + 1}  (matched, p99={vmax:.3g})",
                    x_label    = "range index",
                    y_label    = "azimuth index",
                    cbar_label = f"|Δ{short}|",
                    extent     = extent,
                    cmap       = self.err_cmap,
                    vmin       = 0.0,
                    vmax       = vmax,
                    origin     = "upper",
                    path       = out_dir / f"g{k + 1}_{fname}.png",
                ))

        return paths
