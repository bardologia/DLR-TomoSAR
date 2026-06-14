from __future__ import annotations

from pathlib import Path
from typing  import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot    as plt
import numpy                as np

from pipelines.backbone.inference.plots.base import PlotTools


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
                ch         = 3 * k + j
                arr_pred   = params_pred[ch]
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

        paths = []

        for k in range(n_gaussians):
            amp_ch = 3 * k
            if params_gt is not None and amp_ch < params_gt.shape[0]:
                gt_amp_flat = params_gt[amp_ch].reshape(-1)
                active_mask = np.isfinite(gt_amp_flat) & (gt_amp_flat >= 1e-3)
            else:
                active_mask = None

            for j, (fname, lbl) in enumerate(self.PARAM_LABELS):
                ch    = 3 * k + j
                short = self.PARAM_SHORT[j]

                pred_flat = params_pred[ch].reshape(-1)
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
                    ax.hist(pred, bins=bin_edges, density=True, color="C3", alpha=0.55, label="Pred", edgecolor="none")
                    ax.axvline(float(np.median(pred)), color="C3", linestyle="--", linewidth=0.9, label=f"med Pred={np.median(pred):.3g}")

                if is_amp:
                    ax.set_xscale("log")
                    ax.set_title(f"g{k + 1} — {lbl}  (active pixels, full range, max={float(combined.max()):.3g})", fontsize=10)
                else:
                    ax.set_title(f"g{k + 1} — {lbl}  (active pixels only)", fontsize=10)

                ax.set_xlabel(short)
                ax.set_ylabel("density")
                ax.legend(fontsize=7, framealpha=0.9)
                ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
                fig.tight_layout()

                paths.append(self._save(fig, out_dir / f"g{k + 1}_{fname}.png"))

        return paths

    def _scatter_panel(self, ax, gt: np.ndarray, pred: np.ndarray, label: str) -> None:
        r2 = self._r2_value(gt, pred)

        ax.scatter(gt, pred, s=2, alpha=0.25, color="C0", rasterized=True, label=f"{label}  R²={r2:.3f}")

    @staticmethod
    def _identity_line(ax, *arrays: np.ndarray) -> None:
        lo = min(float(arr.min()) for arr in arrays)
        hi = max(float(arr.max()) for arr in arrays)

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

        paths = []

        for k in range(n_gaussians):
            amp_ch         = 3 * k
            gt_amp_flat    = params_gt[amp_ch].reshape(-1)
            is_active      = np.isfinite(gt_amp_flat) & (gt_amp_flat >= 1e-3)
            is_placeholder = np.isfinite(gt_amp_flat) & (gt_amp_flat <  1e-3)

            for j, (fname, lbl) in enumerate(self.PARAM_LABELS):
                ch    = 3 * k + j
                short = self.PARAM_SHORT[j]

                if ch >= params_gt.shape[0] or ch >= params_pred.shape[0]:
                    continue

                gt_all   = params_gt  [ch].reshape(-1)
                pred_all = params_pred[ch].reshape(-1)

                fig, ax = plt.subplots(figsize=(4.6, 4.2))

                if j == 0:
                    g_act,  p_act  = self._paired_subsample([gt_all[is_active],      pred_all[is_active]],      max_points, seed)
                    g_phld, p_phld = self._paired_subsample([gt_all[is_placeholder], pred_all[is_placeholder]], max_points, seed)

                    if g_act.size == 0 and g_phld.size == 0:
                        plt.close(fig)
                        continue

                    if g_act.size > 0:
                        self._scatter_panel(ax, g_act, p_act, "active")

                    if g_phld.size > 0:
                        ax.scatter(g_phld, p_phld, s=2, alpha=0.35, color="C1", rasterized=True, label=f"placeholder (n={g_phld.size})")

                    self._identity_line(ax, np.concatenate([g_act, g_phld]), np.concatenate([p_act, p_phld]))
                    ax.set_title(f"g{k + 1} — {lbl}", fontsize=10)

                else:
                    gt, pred = self._paired_subsample([gt_all[is_active], pred_all[is_active]], max_points, seed)

                    if gt.size == 0:
                        plt.close(fig)
                        continue

                    self._scatter_panel(ax, gt, pred, "active")
                    self._identity_line(ax, gt, pred)
                    r2_str = self._r2_value(gt, pred)
                    ax.set_title(f"g{k + 1} — {lbl}  (R²={r2_str:.3f}, active only)", fontsize=10)

                ax.set_xlabel(f"GT {short}")
                ax.set_ylabel(f"Pred {short}")
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

        H, W   = params_pred.shape[-2:]
        extent = [rg_offset, rg_offset + W, az_offset + H, az_offset]
        paths  = []

        for k in range(n_gaussians):
            for j, (fname, _) in enumerate(self.PARAM_LABELS):
                ch    = 3 * k + j
                short = self.PARAM_SHORT[j]

                if ch >= params_gt.shape[0] or ch >= params_pred.shape[0]:
                    continue

                err       = np.abs(params_pred[ch] - params_gt[ch]).astype(np.float32)
                valid_err = err[np.isfinite(err)]
                if valid_err.size == 0:
                    continue

                vmax = float(np.percentile(valid_err, 99.0))

                paths.append(self._imshow_panel(
                    data       = err,
                    title      = f"|Δ{short}| — g{k + 1}  (p99={vmax:.3g})",
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
