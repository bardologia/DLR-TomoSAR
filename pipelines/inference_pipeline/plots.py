from __future__ import annotations

from pathlib import Path
from typing  import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm        as cm
import matplotlib.pyplot    as plt
import numpy                as np

from pipelines.shared.plotting import PlotBase
from tools.gaussians           import GaussianReconstructor


class PlotTools(PlotBase):
    PARAM_LABELS = (("a", "amplitude (a)"), ("mu", "mean (μ)"), ("sigma", "std-dev (σ)"))
    PARAM_SHORT  = ("a", "μ", "σ")

    @staticmethod
    def _gaussian_components(params: np.ndarray, x_axis: np.ndarray, n_gaussians: int) -> List[np.ndarray]:
        return GaussianReconstructor.components(params, x_axis, n_gaussians)


class Ploter(PlotTools):
    def __init__(
        self,
        cmap     : str  = "jet",
        err_cmap : str  = "magma",
        normalize: bool = False,
        fig_dpi  : int  = 150,
        save_dpi : int  = 150,
    ) -> None:

        self.cmap       = cmap
        self.err_cmap   = err_cmap
        self.normalize  = normalize
        self.fig_dpi    = fig_dpi
        self.save_dpi   = save_dpi
        self._apply_style()

    def _maybe_normalize(self, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
        if self.normalize:
            return tuple(self._normalize_01(a) for a in arrays)
        return arrays

    @property
    def _int_label(self) -> str:
        return "intensity [0-1]" if self.normalize else "intensity"

    def _imshow_figure(
        self,
        data       : np.ndarray,
        title      : str,
        x_label    : str,
        y_label    : str,
        cbar_label : str,
        extent     : list,
        cmap       : str,
        vmin       : float,
        vmax       : float,
        origin     : str,
        path       : Path,
        figsize    : Tuple[float, float] = (6.2, 4.4),
    ) -> Path:

        fig, ax = plt.subplots(figsize=figsize)
        im      = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, aspect="auto", origin=origin, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).set_label(cbar_label)
        fig.tight_layout()

        return self._save(fig, path)

    def plot_profiles(
        self,
        pred_curves   : np.ndarray,
        gt_curves     : np.ndarray,
        params_pred   : np.ndarray,
        x_axis        : np.ndarray,
        pixels        : np.ndarray,
        tag           : str,
        out_dir       : Path,
        n_gaussians   : int,
        pixel_metrics : Dict[str, np.ndarray],
        az_offset     : int,
        rg_offset     : int,
    ) -> List[Path]:

        base_colors = [cm.tab10(i) for i in range(10)]
        paths       = []

        for p, (y, x) in enumerate(pixels):
            gt    = gt_curves  [:, y, x]
            pred  = pred_curves[:, y, x]
            comps = self._gaussian_components(params_pred[:, y, x], x_axis, n_gaussians)

            fig, ax = plt.subplots(figsize=(5.2, 3.4))
            ax.plot(x_axis, gt,   color="black", linewidth=1.4, label="GT",   zorder=3)
            ax.plot(x_axis, pred, color="C3",    linewidth=1.2, label="Pred", linestyle="--", zorder=4)

            for k, comp in enumerate(comps):
                ax.plot(x_axis, comp, color=base_colors[k % len(base_colors)], linewidth=0.8, alpha=0.75, label=f"$g_{{{k+1}}}$")

            ax.fill_between(x_axis, pred - gt, 0.0, color="C0", alpha=0.10, linewidth=0)

            mse = float(pixel_metrics["mse"][y, x])
            r2  = float(pixel_metrics["r2"] [y, x])
            cos = float(pixel_metrics["cos"][y, x])

            ax.set_title(f"az={y + az_offset}, rg={x + rg_offset}   MSE={mse:.3g}  R²={r2:.3f}  cos={cos:.3f}", fontsize=9)
            ax.set_xlabel("elevation [m]")
            ax.set_ylabel("intensity")
            ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
            ax.legend(loc="best", framealpha=0.85, fontsize=7, ncol=2)
            fig.tight_layout()

            paths.append(self._save(fig, out_dir / f"{tag}_{p + 1:02d}_az{y + az_offset}_rg{x + rg_offset}.png"))

        return paths

    def plot_pixel_metric_map(
        self,
        metric_map : np.ndarray,
        title      : str,
        label      : str,
        out_path   : Path,
        az_offset  : int,
        rg_offset  : int,
        cmap       : str   = "jet",
        log        : bool  = False,
        q_low      : float = 1.0,
        q_high     : float = 99.0,
    ) -> Path:

        H, W   = metric_map.shape
        extent = [rg_offset, rg_offset + W, az_offset + H, az_offset]

        data = metric_map.copy()
        if log:
            data = np.log10(np.clip(data, 1e-12, None))

        vmin, vmax = self._shared_clim(data, q_low=q_low, q_high=q_high)

        return self._imshow_figure(
            data       = data,
            title      = title,
            x_label    = "range index",
            y_label    = "azimuth index",
            cbar_label = label + (" [log10]" if log else ""),
            extent     = extent,
            cmap       = cmap,
            vmin       = vmin,
            vmax       = vmax,
            origin     = "upper",
            path       = out_path,
            figsize    = (7, 5),
        )

    def plot_tomogram_slice(
        self,
        pred_cube  : np.ndarray,
        gt_cube    : np.ndarray,
        axis       : str,
        index      : int,
        x_axis     : np.ndarray,
        out_dir    : Path,
        stem       : str,
        az_offset  : int,
        rg_offset  : int,
        ssim_value : Optional[float] = None,
    ) -> List[Path]:

        if axis == "range":
            pred_slice               = pred_cube[:, :, index]
            gt_slice                 = gt_cube  [:, :, index]
            x_label                  = "azimuth index"
            x_extent_lo, x_extent_hi = az_offset, az_offset + pred_slice.shape[1]
            title_pos                = f"range = {index + rg_offset}"

        elif axis == "azimuth":
            pred_slice               = pred_cube[:, index, :]
            gt_slice                 = gt_cube  [:, index, :]
            x_label                  = "range index"
            x_extent_lo, x_extent_hi = rg_offset, rg_offset + pred_slice.shape[1]
            title_pos                = f"azimuth = {index + az_offset}"

        else:
            raise ValueError(f"axis must be 'range' or 'azimuth', got {axis!r}")

        err_gt_slice = np.abs(pred_slice - gt_slice)

        sort_idx      = np.argsort(x_axis)
        x_axis_sorted = x_axis[sort_idx]
        pred_slice    = pred_slice  [sort_idx]
        gt_slice      = gt_slice    [sort_idx]
        err_gt_slice  = err_gt_slice[sort_idx]

        extent = [x_extent_lo, x_extent_hi, float(x_axis_sorted[0]), float(x_axis_sorted[-1])]

        pred_slice, gt_slice, err_gt_slice = self._maybe_normalize(pred_slice, gt_slice, err_gt_slice)

        vmin, vmax = self._shared_clim(gt_slice, pred_slice)
        emax_gt    = float(np.percentile(err_gt_slice, 99.0))
        ssim_str   = f"   SSIM = {ssim_value:.4f}" if ssim_value is not None and np.isfinite(ssim_value) else ""

        panels = [
            (gt_slice,     f"GT (Gaussian) — {title_pos}",         self.cmap,     vmin, vmax,    self._int_label, "gt"),
            (pred_slice,   f"Prediction — {title_pos}{ssim_str}",  self.cmap,     vmin, vmax,    self._int_label, "pred"),
            (err_gt_slice, f"|Pred − GT| — {title_pos}",           self.err_cmap, 0.0,  emax_gt, "|error|",       "error"),
        ]

        return [
            self._imshow_figure(
                data       = data,
                title      = title,
                x_label    = x_label,
                y_label    = "elevation [m]",
                cbar_label = cbar,
                extent     = extent,
                cmap       = cmap_used,
                vmin       = vlo,
                vmax       = vhi,
                origin     = "lower",
                path       = out_dir / f"{stem}_{kind}.png",
            )
            for data, title, cmap_used, vlo, vhi, cbar, kind in panels
        ]

    def plot_elevation_intensity_slice(
        self,
        pred_cube  : np.ndarray,
        gt_cube    : np.ndarray,
        elev_idx   : int,
        x_axis     : np.ndarray,
        out_dir    : Path,
        stem       : str,
        az_offset  : int,
        rg_offset  : int,
        ssim_value : Optional[float] = None,
    ) -> List[Path]:

        pred_slice   = pred_cube[elev_idx]
        gt_slice     = gt_cube  [elev_idx]
        err_gt_slice = np.abs(pred_slice - gt_slice)

        pred_slice, gt_slice, err_gt_slice = self._maybe_normalize(pred_slice, gt_slice, err_gt_slice)

        H, W       = pred_slice.shape
        extent     = [rg_offset, rg_offset + W, az_offset + H, az_offset]
        vmin, vmax = self._shared_clim(gt_slice, pred_slice)
        emax_gt    = float(np.percentile(err_gt_slice, 99.0))

        title_pos = f"elev = {x_axis[elev_idx]:.2f} m (idx {elev_idx})"
        ssim_str  = f"   SSIM = {ssim_value:.4f}" if ssim_value is not None and np.isfinite(ssim_value) else ""

        panels = [
            (gt_slice,     f"GT (Gaussian) — {title_pos}",        self.cmap,     vmin, vmax,    self._int_label, "gt"),
            (pred_slice,   f"Prediction — {title_pos}{ssim_str}", self.cmap,     vmin, vmax,    self._int_label, "pred"),
            (err_gt_slice, f"|Pred − GT| — {title_pos}",          self.err_cmap, 0.0,  emax_gt, "|error|",       "error"),
        ]

        return [
            self._imshow_figure(
                data       = data,
                title      = title,
                x_label    = "range index",
                y_label    = "azimuth index",
                cbar_label = cbar,
                extent     = extent,
                cmap       = cmap_used,
                vmin       = vlo,
                vmax       = vhi,
                origin     = "upper",
                path       = out_dir / f"{stem}_{kind}.png",
            )
            for data, title, cmap_used, vlo, vhi, cbar, kind in panels
        ]

    def plot_metric_histograms(self, metric_arrays: Dict[str, np.ndarray], out_dir: Path) -> List[Path]:
        paths = []

        for name, arr in metric_arrays.items():
            flat = arr.reshape(-1)
            flat = flat[np.isfinite(flat)]
            if flat.size == 0:
                continue

            lo, hi  = np.percentile(flat, [0.5, 99.5])
            fig, ax = plt.subplots(figsize=(4.8, 3.4))
            ax.hist(flat, bins=80, range=(lo, hi), color="C0", edgecolor="white", linewidth=0.3)
            ax.axvline(float(np.median(flat)), color="black", linestyle="--", linewidth=1.0, label=f"median={np.median(flat):.3g}")
            ax.set_title(f"{name} (denorm)")
            ax.set_xlabel(name)
            ax.set_ylabel("count")
            ax.set_yscale("log")
            ax.legend(framealpha=0.9)
            ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
            fig.tight_layout()

            paths.append(self._save(fig, out_dir / f"{name}.png"))

        return paths

    def plot_ssim_curves(
        self,
        global_metrics : dict,
        axis           : str,
        out_path       : Path,
        n_slices       : int,
        slice_indices  : np.ndarray,
        ax_offset      : int = 0,
    ) -> Path:

        vals_gt = np.array([global_metrics.get(f"ssim_gt_{axis}_{i}", float("nan")) for i in range(n_slices)], dtype=np.float64)
        x_phys  = slice_indices.astype(np.float64) + ax_offset

        mean_gt = float(np.nanmean(vals_gt))

        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.plot(x_phys, vals_gt, color="C0", linewidth=0.9, label="pred × GT (Gaussian)", alpha=0.9)
        ax.axhline(mean_gt, color="C0", linestyle="--", linewidth=1.0, label=f"mean = {mean_gt:.4f}")

        tick_locs = np.linspace(x_phys[0], x_phys[-1], min(20, n_slices)).round().astype(int)
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_locs, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel(f"{axis} index")
        ax.set_ylabel("SSIM")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"SSIM per {axis}-slice  (n={n_slices})")
        ax.legend(framealpha=0.9)
        ax.grid(True, axis="y", which="major", linewidth=0.3, alpha=0.5)
        fig.tight_layout()

        return self._save(fig, out_path)

    def plot_elev_metric_curves(
        self,
        global_metrics : dict,
        out_dir        : Path,
        n_elev         : int,
        x_axis         : np.ndarray,
    ) -> List[Path]:

        metric_specs = [
            ("elev_mae",  "mae",           "MAE",           "mean absolute error"),
            ("elev_rmse", "rmse",          "RMSE",          "root mean squared error"),
            ("elev_r2",   "r2",            "R²",            "coefficient of determination"),
            ("elev_ce",   "cross_entropy", "cross-entropy", "cross-entropy (normalised profiles)"),
        ]

        paths = []
        for key, fname, ylabel, desc in metric_specs:
            vals_gt = np.array([global_metrics.get(f"{key}_gt_{i}", float("nan")) for i in range(n_elev)], dtype=np.float64)
            mean_gt = float(np.nanmean(vals_gt))

            fig, ax = plt.subplots(figsize=(5.8, 3.6))
            ax.plot(x_axis, vals_gt, color="C0", linewidth=0.9, label="pred × GT (Gaussian)", alpha=0.9)
            ax.axhline(mean_gt, color="C0", linestyle="--", linewidth=1.0, label=f"mean = {mean_gt:.4g}")

            ax.set_xlabel("elevation [m]")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} per elevation bin — {desc}")
            ax.legend(framealpha=0.9)
            ax.grid(True, axis="y", which="major", linewidth=0.3, alpha=0.5)
            fig.tight_layout()

            paths.append(self._save(fig, out_dir / f"{fname}.png"))

        return paths

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

                paths.append(self._imshow_figure(
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
                    paths.append(self._imshow_figure(
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

    def plot_param_scatter(
        self,
        params_pred : np.ndarray,
        params_gt   : np.ndarray,
        n_gaussians : int,
        out_dir     : Path,
        max_points  : int = 8_000,
        seed        : int = 0,
    ) -> List[Path]:

        rng   = np.random.default_rng(seed)
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
                    def _subsample(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                        m = mask & np.isfinite(gt_all) & np.isfinite(pred_all)
                        g, p = gt_all[m], pred_all[m]
                        if g.size > max_points:
                            idx  = rng.choice(g.size, max_points, replace=False)
                            g, p = g[idx], p[idx]
                        return g, p

                    g_act,  p_act  = _subsample(is_active)
                    g_phld, p_phld = _subsample(is_placeholder)

                    if g_act.size == 0 and g_phld.size == 0:
                        plt.close(fig)
                        continue

                    all_g = np.concatenate([g_act, g_phld]) if g_phld.size else g_act
                    all_p = np.concatenate([p_act, p_phld]) if p_phld.size else p_act
                    lo    = min(float(all_g.min()), float(all_p.min()))
                    hi    = max(float(all_g.max()), float(all_p.max()))

                    if g_act.size > 0:
                        ss_res = float(np.sum((g_act - p_act) ** 2))
                        ss_tot = float(np.sum((g_act - np.mean(g_act)) ** 2))
                        r2_act = 1.0 - ss_res / (ss_tot + 1e-12)
                        ax.scatter(g_act, p_act, s=2, alpha=0.25, color="C0", rasterized=True, label=f"active (R²={r2_act:.3f})")

                    if g_phld.size > 0:
                        ax.scatter(g_phld, p_phld, s=2, alpha=0.35, color="C1", rasterized=True, label=f"placeholder (n={g_phld.size})")

                    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.9, linestyle="--", label="identity")
                    ax.set_title(f"g{k + 1} — {lbl}", fontsize=10)

                else:
                    mask     = is_active & np.isfinite(gt_all) & np.isfinite(pred_all)
                    gt, pred = gt_all[mask], pred_all[mask]

                    if gt.size == 0:
                        plt.close(fig)
                        continue

                    if gt.size > max_points:
                        idx      = rng.choice(gt.size, max_points, replace=False)
                        gt, pred = gt[idx], pred[idx]

                    ss_res = float(np.sum((gt - pred) ** 2))
                    ss_tot = float(np.sum((gt - np.mean(gt)) ** 2))
                    r2     = 1.0 - ss_res / (ss_tot + 1e-12)
                    lo     = min(float(gt.min()), float(pred.min()))
                    hi     = max(float(gt.max()), float(pred.max()))

                    ax.scatter(gt, pred, s=2, alpha=0.25, color="C0", rasterized=True, label=f"active  R²={r2:.3f}")
                    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.9, linestyle="--", label="identity")
                    ax.set_title(f"g{k + 1} — {lbl}  (R²={r2:.3f}, active only)", fontsize=10)

                ax.set_xlabel(f"GT {short}")
                ax.set_ylabel(f"Pred {short}")
                ax.legend(fontsize=7, framealpha=0.9)
                ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
                fig.tight_layout()

                paths.append(self._save(fig, out_dir / f"g{k + 1}_{fname}.png"))

        return paths

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

                paths.append(self._imshow_figure(
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

    def plot_slot_mu_distributions(
        self,
        global_metrics : dict,
        n_gaussians    : int,
        out_dir        : Path,
    ) -> List[Path]:

        slots = list(range(n_gaussians))
        x     = np.arange(n_gaussians)
        width = 0.35

        pred_means = np.array([global_metrics.get(f"slot_{k}_mu_pred_mean", np.nan) for k in slots])
        pred_stds  = np.array([global_metrics.get(f"slot_{k}_mu_pred_std",  np.nan) for k in slots])
        gt_means   = np.array([global_metrics.get(f"slot_{k}_mu_gt_mean",   np.nan) for k in slots])
        gt_stds    = np.array([global_metrics.get(f"slot_{k}_mu_gt_std",    np.nan) for k in slots])

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

        def _get(key: str) -> float:
            return float(global_metrics.get(key, np.nan))

        precisions = [_get(f"slot_{k}_placeholder_precision") for k in slots] + [_get("placeholder_precision")]
        recalls    = [_get(f"slot_{k}_placeholder_recall")    for k in slots] + [_get("placeholder_recall")]
        f1s        = [_get(f"slot_{k}_placeholder_f1")        for k in slots] + [_get("placeholder_f1")]
        gt_rates   = [_get(f"slot_{k}_placeholder_gt_rate")   for k in slots] + [_get("placeholder_gt_rate") if "placeholder_gt_rate" in global_metrics else np.nan]

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

        ordering_rate = float(global_metrics.get("mu_ordering_rate",                    np.nan))
        dominant_frac = float(global_metrics.get("permutation_consensus_dominant_frac", np.nan))
        identity_frac = float(global_metrics.get("permutation_consensus_identity_frac", np.nan))

        slots        = list(range(n_gaussians))
        active_rates = [1.0 - float(global_metrics.get(f"slot_{k}_placeholder_gt_rate", np.nan)) for k in slots]

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
        pred_means = np.array([global_metrics.get(f"slot_{k}_mu_pred_mean", np.nan) for k in slots])
        gt_means   = np.array([global_metrics.get(f"slot_{k}_mu_gt_mean",   np.nan) for k in slots])
        pred_stds  = np.array([global_metrics.get(f"slot_{k}_mu_pred_std",  np.nan) for k in slots])
        gt_stds    = np.array([global_metrics.get(f"slot_{k}_mu_gt_std",    np.nan) for k in slots])

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

        from matplotlib.patches import Patch
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
