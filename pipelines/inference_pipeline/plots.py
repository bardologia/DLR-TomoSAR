from __future__ import annotations

from pathlib import Path
from typing  import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm        as cm
import matplotlib.pyplot    as plt
import numpy                as np

from pipelines.inference_pipeline.reconstruction import GaussianReconstructor


class PlotTools:
    SCIENTIFIC_RC: dict = {
        "font.family"         : "serif",
        "font.serif"          : ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset"    : "dejavuserif",
        "font.size"           : 11,
        "axes.titlesize"      : 12,
        "axes.labelsize"      : 11,
        "xtick.labelsize"     : 10,
        "ytick.labelsize"     : 10,
        "legend.fontsize"     : 9,
        "axes.linewidth"      : 0.8,
        "xtick.direction"     : "in",
        "ytick.direction"     : "in",
        "xtick.top"           : True,
        "ytick.right"         : True,
        "xtick.minor.visible" : True,
        "ytick.minor.visible" : True,
        "image.interpolation" : "nearest",
        "savefig.bbox"        : "tight",
    }

    @staticmethod
    def _gaussian_components(params: np.ndarray, x_axis: np.ndarray, n_gaussians: int) -> List[np.ndarray]:
        return GaussianReconstructor.components(params, x_axis, n_gaussians)

    @staticmethod
    def _triple_panel(
        fig,
        axes,
        panels    : List[Tuple[np.ndarray, str, str, float, float]],
        x_label   : str,
        int_label : str,
        extent    : list,
        origin    : str,
    ) -> None:

        for ax_i, (data, label, cm_used, vlo, vhi) in zip(axes, panels):
            im = ax_i.imshow(data, cmap=cm_used, vmin=vlo, vmax=vhi, extent=extent, aspect="auto", origin=origin)
            ax_i.set_title(label)
            ax_i.set_xlabel(x_label)
            lbl_cb = int_label if cm_used == panels[0][2] else "|error|"
            fig.colorbar(im, ax=ax_i, fraction=0.045, pad=0.02).set_label(lbl_cb)

    @staticmethod
    def _shared_clim(*arrays: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> Tuple[float, float]:
        flat = np.concatenate([a.reshape(-1) for a in arrays])
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return (0.0, 1.0)
        return float(np.percentile(flat, q_low)), float(np.percentile(flat, q_high))

    @staticmethod
    def _save(fig: plt.Figure, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        return path

    @staticmethod
    def _normalize_01(arr: np.ndarray) -> np.ndarray:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if hi - lo < 1e-12:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - lo) / (hi - lo)).astype(np.float32)


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

    def _apply_style(self) -> None:
        plt.rcParams.update(self.SCIENTIFIC_RC)
        plt.rcParams["figure.dpi"]  = self.fig_dpi
        plt.rcParams["savefig.dpi"] = self.save_dpi

    def _maybe_normalize(self, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
        if self.normalize:
            return tuple(self._normalize_01(a) for a in arrays)
        return arrays

    @property
    def _int_label(self) -> str:
        return "intensity [0-1]" if self.normalize else "intensity"

    def plot_profile_panel(
        self,
        pred_curves   : np.ndarray,
        gt_curves     : np.ndarray,
        params_pred   : np.ndarray,
        x_axis        : np.ndarray,
        pixels        : np.ndarray,
        title         : str,
        out_path      : Path,
        n_gaussians   : int,
        pixel_metrics : Dict[str, np.ndarray],
        az_offset     : int,
        rg_offset     : int,
        n_cols        : int = 4,
    ) -> Path:
        
        P = len(pixels)
        if P == 0:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, f"{title}\n(no pixels selected)", ha="center", va="center")
            ax.set_axis_off()
            return self._save(fig, out_path)

        n_cols    = min(n_cols, P)
        n_rows    = int(np.ceil(P / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.6 * n_rows), squeeze=False)

        base_colors = [cm.tab10(i) for i in range(10)]

        for ax in axes.ravel():
            ax.set_visible(False)

        for p, (y, x) in enumerate(pixels):
            r, c = divmod(p, n_cols)
            ax   = axes[r, c]
            ax.set_visible(True)

            gt    = gt_curves  [:, y, x]
            pred  = pred_curves[:, y, x]
            comps = self._gaussian_components(params_pred[:, y, x], x_axis, n_gaussians)

            ax.plot(x_axis, gt,   color="black", linewidth=1.4, label="GT",   zorder=3)
            ax.plot(x_axis, pred, color="C3",    linewidth=1.2, label="Pred", linestyle="--", zorder=4)
            
            for k, comp in enumerate(comps):
                ax.plot(x_axis, comp, color=base_colors[k % len(base_colors)], linewidth=0.8, alpha=0.75, label=f"$g_{{{k+1}}}$" if p == 0 else None)
            
            ax.fill_between(x_axis, pred - gt, 0.0, color="C0", alpha=0.10, linewidth=0)

            mse = float(pixel_metrics["mse"][y, x])
            r2  = float(pixel_metrics["r2"] [y, x])
            cos = float(pixel_metrics["cos"][y, x])

            ax.set_title(
                f"az={y + az_offset}, rg={x + rg_offset}\n"
                f"MSE={mse:.3g}  R²={r2:.3f}  cos={cos:.3f}",
                fontsize=8,
            )

            ax.set_xlabel("elevation [m]")
            ax.set_ylabel("intensity")
            ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
            if p == 0:
                ax.legend(loc="best", framealpha=0.85)

        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        
        return self._save(fig, out_path)

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
        fig, ax    = plt.subplots(figsize=(7, 5))
        im         = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, aspect="auto", interpolation="nearest")
        ax.set_xlabel("range index")
        ax.set_ylabel("azimuth index")
        ax.set_title(title)
        cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
        cb.set_label(label + (" [log10]" if log else ""))
        
        return self._save(fig, out_path)

    def plot_tomogram_slice(
        self,
        pred_cube  : np.ndarray,
        gt_cube    : np.ndarray,
        axis       : str,
        index      : int,
        x_axis     : np.ndarray,
        out_path   : Path,
        az_offset  : int,
        rg_offset  : int,
        ssim_value : Optional[float] = None,
    ) -> Path:

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

        extent_int = [x_extent_lo, x_extent_hi, float(x_axis_sorted[0]), float(x_axis_sorted[-1])]

        pred_slice, gt_slice, err_gt_slice = self._maybe_normalize(pred_slice, gt_slice, err_gt_slice)

        vmin, vmax = self._shared_clim(gt_slice, pred_slice)
        emax_gt    = float(np.percentile(err_gt_slice, 99.0))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

        panels = [
            (gt_slice,    "GT (Gaussian)", self.cmap,     vmin,  vmax),
            (pred_slice,  "Prediction",    self.cmap,     vmin,  vmax),
            (err_gt_slice, "|Pred - GT|",  self.err_cmap, 0.0,   emax_gt),
        ]

        self._triple_panel(fig, axes, panels, x_label, self._int_label, extent_int, origin="lower")

        axes[0].set_ylabel("elevation [m]")

        ssim_str = f"   SSIM = {ssim_value:.4f}" if ssim_value is not None and np.isfinite(ssim_value) else ""
        fig.suptitle(f"Tomogram slice ({axis}-cut) - {title_pos}{ssim_str}", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        return self._save(fig, out_path)

    def plot_elevation_intensity_slice(
        self,
        pred_cube  : np.ndarray,
        gt_cube    : np.ndarray,
        elev_idx   : int,
        x_axis     : np.ndarray,
        out_path   : Path,
        az_offset  : int,
        rg_offset  : int,
        ssim_value : Optional[float] = None,
    ) -> Path:

        pred_slice   = pred_cube[elev_idx]
        gt_slice     = gt_cube  [elev_idx]
        err_gt_slice = np.abs(pred_slice - gt_slice)

        pred_slice, gt_slice, err_gt_slice = self._maybe_normalize(pred_slice, gt_slice, err_gt_slice)

        H, W       = pred_slice.shape
        extent     = [rg_offset, rg_offset + W, az_offset + H, az_offset]
        vmin, vmax = self._shared_clim(gt_slice, pred_slice)
        emax_gt    = float(np.percentile(err_gt_slice, 99.0))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        panels = [
            (gt_slice,     "GT (Gaussian)", self.cmap,     vmin, vmax),
            (pred_slice,   "Prediction",    self.cmap,     vmin, vmax),
            (err_gt_slice, "|Pred - GT|",   self.err_cmap, 0.0,  emax_gt),
        ]

        self._triple_panel(fig, axes, panels, "range index", self._int_label, extent, origin="upper")

        for ax_i in axes:
            ax_i.set_ylabel("azimuth index")

        ssim_str = f"   SSIM = {ssim_value:.4f}" if ssim_value is not None and np.isfinite(ssim_value) else ""
        fig.suptitle(f"Elevation slice (elev = {x_axis[elev_idx]:.2f} m, idx={elev_idx}){ssim_str}", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        return self._save(fig, out_path)

    def plot_metric_histogram(self, metric_arrays: Dict[str, np.ndarray], out_path: Path) -> Path:
        fig, axes = plt.subplots(1, len(metric_arrays), figsize=(4.5 * len(metric_arrays), 3.5))
        if len(metric_arrays) == 1:
            axes = [axes]

        for ax, (name, arr) in zip(axes, metric_arrays.items()):
            flat = arr.reshape(-1)
            flat = flat[np.isfinite(flat)]
            if flat.size == 0:
                ax.set_axis_off()
                ax.set_title(f"{name} (empty)")
                continue

            lo, hi = np.percentile(flat, [0.5, 99.5])
            ax.hist(flat, bins=80, range=(lo, hi), color="C0", edgecolor="white", linewidth=0.3)
            ax.axvline(float(np.median(flat)), color="black", linestyle="--", linewidth=1.0, label=f"median={np.median(flat):.3g}")
            ax.set_title(name)
            ax.set_xlabel(name)
            ax.set_ylabel("count")
            ax.set_yscale("log")
            ax.legend(framealpha=0.9)
            ax.grid(True, which="both", linewidth=0.3, alpha=0.4)

        fig.tight_layout()
        
        return self._save(fig, out_path)

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

        fig, ax = plt.subplots(figsize=(12, 4.2))
        ax.plot(x_phys, vals_gt, color="C0", linewidth=0.9, label="pred × GT (Gaussian)", alpha=0.9)
        ax.axhline(mean_gt, color="C0", linestyle="--", linewidth=1.0, label=f"mean = {mean_gt:.4f}")

        tick_locs = np.linspace(x_phys[0], x_phys[-1], min(40, n_slices)).round().astype(int)
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
        out_path       : Path,
        n_elev         : int,
        x_axis         : np.ndarray,
    ) -> Path:
    
        metric_specs = [
            ("elev_mae",  "MAE",           "mean absolute error"),
            ("elev_rmse", "RMSE",          "root mean squared error"),
            ("elev_r2",   "R²",            "coefficient of determination"),
            ("elev_ce",   "cross-entropy", "cross-entropy (normalised profiles)"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(13, 7))

        for ax, (key, ylabel, desc) in zip(axes.ravel(), metric_specs):
            vals_gt = np.array([global_metrics.get(f"{key}_gt_{i}", float("nan")) for i in range(n_elev)], dtype=np.float64)
            mean_gt = float(np.nanmean(vals_gt))

            ax.plot(x_axis, vals_gt, color="C0", linewidth=0.9, label="pred × GT (Gaussian)", alpha=0.9)
            ax.axhline(mean_gt, color="C0", linestyle="--", linewidth=1.0, label=f"mean = {mean_gt:.4g}")

            ax.set_xlabel("elevation [m]")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} per elevation bin  —  {desc}")
            ax.legend(framealpha=0.9)
            ax.grid(True, axis="y", which="major", linewidth=0.3, alpha=0.5)

        fig.suptitle("Per-elevation-bin metrics  (aggregated over all az × rg pixels)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        return self._save(fig, out_path)

    def plot_param_maps(
        self,
        params_pred : np.ndarray,
        params_gt   : Optional[np.ndarray],
        n_gaussians : int,
        out_path    : Path,
        az_offset   : int,
        rg_offset   : int,
    ) -> Path:
        
        cols      = 2 if params_gt is not None else 1
        fig, axes = plt.subplots(n_gaussians * 3, cols, figsize=(6.5 * cols, 3.0 * n_gaussians * 3), squeeze=False)

        H, W   = params_pred.shape[-2:]
        extent = [rg_offset, rg_offset + W, az_offset + H, az_offset]

        for k in range(n_gaussians):
            for j, sub in enumerate(("a", "mu", "sig")):
                ch         = 3 * k + j
                row        = k * 3 + j
                arr_pred   = params_pred[ch]
                vmin, vmax = self._shared_clim(arr_pred if params_gt is None else np.stack([arr_pred, params_gt[ch]]))

                ax = axes[row, 0]
                im = ax.imshow(arr_pred, cmap="jet", vmin=vmin, vmax=vmax, extent=extent, aspect="auto")
                ax.set_title(f"Pred {sub}{k+1}")
                ax.set_xlabel("range")
                ax.set_ylabel("azimuth")
                fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

                if params_gt is not None and ch < params_gt.shape[0]:
                    ax2 = axes[row, 1]
                    im2 = ax2.imshow(params_gt[ch], cmap="jet", vmin=vmin, vmax=vmax, extent=extent, aspect="auto")
                    ax2.set_title(f"GT {sub}{k+1}")
                    ax2.set_xlabel("range")
                    fig.colorbar(im2, ax=ax2, fraction=0.045, pad=0.02)

        fig.suptitle("Gaussian parameter maps", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        
        return self._save(fig, out_path)

    def plot_param_distributions(
        self,
        params_pred : np.ndarray,
        params_gt   : Optional[np.ndarray],
        n_gaussians : int,
        out_path    : Path,
        bins        : int = 80,
    ) -> Path:
        fig, axes = plt.subplots(n_gaussians, 3, figsize=(13.5, 3.0 * n_gaussians), squeeze=False)

        for k in range(n_gaussians):
            amp_ch   = 3 * k         
            if params_gt is not None and amp_ch < params_gt.shape[0]:
                gt_amp_flat = params_gt[amp_ch].reshape(-1)
                active_mask = np.isfinite(gt_amp_flat) & (gt_amp_flat >= 1e-3)
            else:
                active_mask = None

            for j, (lbl, short) in enumerate(zip(("amplitude (a)", "mean (μ)", "std-dev (σ)"), ("a", "μ", "σ"))):
                ch   = 3 * k + j
                ax   = axes[k, j]

                pred_flat = params_pred[ch].reshape(-1)
                if active_mask is not None:
                    pred_flat = pred_flat[active_mask]
                pred = pred_flat[np.isfinite(pred_flat)]

                has_pred = pred.size > 0
                lo = float(np.percentile(pred, 0.5))  if has_pred else 0.0
                hi = float(np.percentile(pred, 99.5)) if has_pred else 1.0

                has_gt = False
                if params_gt is not None and ch < params_gt.shape[0]:
                    gt_flat = params_gt[ch].reshape(-1)
                    if active_mask is not None:
                        gt_flat = gt_flat[active_mask]
                    gt     = gt_flat[np.isfinite(gt_flat)]
                    has_gt = gt.size > 0
                    if has_gt:
                        lo = min(lo, float(np.percentile(gt, 0.5)))  if has_pred else float(np.percentile(gt, 0.5))
                        hi = max(hi, float(np.percentile(gt, 99.5))) if has_pred else float(np.percentile(gt, 99.5))
                        ax.hist(gt, bins=bins, range=(lo, hi), density=True, color="C0", alpha=0.55, label="GT", edgecolor="none")
                        ax.axvline(float(np.median(gt)), color="C0", linestyle="--", linewidth=0.9, label=f"med GT={np.median(gt):.3g}")

                if has_pred:
                    ax.hist(pred, bins=bins, range=(lo, hi), density=True, color="C3", alpha=0.55, label="Pred", edgecolor="none")
                    ax.axvline(float(np.median(pred)), color="C3", linestyle="--", linewidth=0.9, label=f"med Pred={np.median(pred):.3g}")

                ax.set_title(f"g{k+1} — {lbl}  (active pixels only)")
                ax.set_xlabel(short)
                ax.set_ylabel("density")
                if has_pred or has_gt:
                    ax.legend(fontsize=7, framealpha=0.9)
                ax.grid(True, which="both", linewidth=0.3, alpha=0.4)

        fig.suptitle("Gaussian parameter distributions (GT vs Pred, placeholder slots excluded)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))

        return self._save(fig, out_path)

    def plot_param_scatter(
        self,
        params_pred : np.ndarray,
        params_gt   : np.ndarray,
        n_gaussians : int,
        out_path    : Path,
        max_points  : int = 8_000,
        seed        : int = 0,
    ) -> Path:
        fig, axes = plt.subplots(n_gaussians, 3, figsize=(13.5, 3.5 * n_gaussians), squeeze=False)
        rng       = np.random.default_rng(seed)

        for k in range(n_gaussians):
            amp_ch          = 3 * k
            gt_amp_flat     = params_gt[amp_ch].reshape(-1)
            is_active       = np.isfinite(gt_amp_flat) & (gt_amp_flat >= 1e-3)
            is_placeholder  = np.isfinite(gt_amp_flat) & (gt_amp_flat <  1e-3)

            for j, (lbl, short) in enumerate(zip(("amplitude (a)", "mean (μ)", "std-dev (σ)"), ("a", "μ", "σ"))):
                ch  = 3 * k + j
                ax  = axes[k, j]

                if ch >= params_gt.shape[0] or ch >= params_pred.shape[0]:
                    ax.set_axis_off()
                    continue

                gt_all   = params_gt  [ch].reshape(-1)
                pred_all = params_pred[ch].reshape(-1)

                if j == 0:
                    def _subsample(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                        m = mask & np.isfinite(gt_all) & np.isfinite(pred_all)
                        g, p = gt_all[m], pred_all[m]
                        if g.size > max_points:
                            idx = rng.choice(g.size, max_points, replace=False)
                            g, p = g[idx], p[idx]
                        return g, p

                    g_act,  p_act  = _subsample(is_active)
                    g_phld, p_phld = _subsample(is_placeholder)

                    has_data = g_act.size > 0 or g_phld.size > 0
                    if not has_data:
                        ax.set_title(f"g{k+1} — {lbl}  (No Data)")
                        ax.set_axis_off()
                        continue

                    all_g    = np.concatenate([g_act, g_phld]) if g_phld.size else g_act
                    all_p    = np.concatenate([p_act, p_phld]) if p_phld.size else p_act
                    lo       = min(float(all_g.min()), float(all_p.min()))
                    hi       = max(float(all_g.max()), float(all_p.max()))

                    if g_act.size > 0:
                        ss_res = float(np.sum((g_act - p_act) ** 2))
                        ss_tot = float(np.sum((g_act - np.mean(g_act)) ** 2))
                        r2_act = 1.0 - ss_res / (ss_tot + 1e-12)
                        ax.scatter(g_act, p_act, s=2, alpha=0.25, color="C0", rasterized=True, label=f"active (R²={r2_act:.3f})")

                    if g_phld.size > 0:
                        ax.scatter(g_phld, p_phld, s=2, alpha=0.35, color="C1", rasterized=True, label=f"placeholder (n={g_phld.size})")

                    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.9, linestyle="--", label="identity")
                    ax.set_title(f"g{k+1} — {lbl}")

                else:
                    # ── mu / sigma: active pixels only (placeholders are NaN'd) ─
                    mask = is_active & np.isfinite(gt_all) & np.isfinite(pred_all)
                    gt, pred = gt_all[mask], pred_all[mask]

                    if gt.size == 0:
                        ax.set_title(f"g{k+1} — {lbl}  (No active data)")
                        ax.set_axis_off()
                        continue

                    if gt.size > max_points:
                        idx  = rng.choice(gt.size, max_points, replace=False)
                        gt, pred = gt[idx], pred[idx]

                    ss_res = float(np.sum((gt - pred) ** 2))
                    ss_tot = float(np.sum((gt - np.mean(gt)) ** 2))
                    r2     = 1.0 - ss_res / (ss_tot + 1e-12)
                    lo     = min(float(gt.min()), float(pred.min()))
                    hi     = max(float(gt.max()), float(pred.max()))

                    ax.scatter(gt, pred, s=2, alpha=0.25, color="C0", rasterized=True,
                               label=f"active  R²={r2:.3f}")
                    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.9, linestyle="--", label="identity")
                    ax.set_title(f"g{k+1} — {lbl}  (R²={r2:.3f}, active only)")

                ax.set_xlabel(f"GT {short}")
                ax.set_ylabel(f"Pred {short}")
                ax.legend(fontsize=7, framealpha=0.9)
                ax.grid(True, which="both", linewidth=0.3, alpha=0.4)

        fig.suptitle("Gaussian parameter scatter: GT vs Pred", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))

        return self._save(fig, out_path)

    def plot_param_error_maps(
        self,
        params_pred : np.ndarray,
        params_gt   : np.ndarray,
        n_gaussians : int,
        out_path    : Path,
        az_offset   : int,
        rg_offset   : int,
    ) -> Path:

        H, W      = params_pred.shape[-2:]
        extent    = [rg_offset, rg_offset + W, az_offset + H, az_offset]
        fig, axes = plt.subplots(n_gaussians, 3, figsize=(16.5, 3.5 * n_gaussians), squeeze=False)

        for k in range(n_gaussians):
            for j, (lbl, short) in enumerate(zip(("amplitude (a)", "mean (μ)", "std-dev (σ)"), ("a", "μ", "σ"))):
                ch  = 3 * k + j
                ax  = axes[k, j]

                if ch >= params_gt.shape[0] or ch >= params_pred.shape[0]:
                    ax.set_axis_off()
                    continue

                err = np.abs(params_pred[ch] - params_gt[ch]).astype(np.float32)

                valid_err = err[np.isfinite(err)]
                
                if valid_err.size == 0:
                    ax.set_title(f"|Δ{short}| — g{k+1}  (No Data)")
                    ax.set_axis_off()
                    continue
                    
                vmax = float(np.percentile(valid_err, 99.0))
                im   = ax.imshow(err, cmap=self.err_cmap, vmin=0.0, vmax=vmax, extent=extent, aspect="auto", interpolation="nearest")
                ax.set_title(f"|Δ{short}| — g{k+1}  (p99={vmax:.3g})")
                ax.set_xlabel("range index")
                ax.set_ylabel("azimuth index")
                fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).set_label(f"|Δ{short}|")

        fig.suptitle("Gaussian parameter absolute-error maps  |Pred − GT|", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))

        return self._save(fig, out_path)

    def plot_slot_mu_distributions(
        self,
        global_metrics : dict,
        n_gaussians    : int,
        out_path       : Path,
    ) -> Path:
        slots      = list(range(n_gaussians))
        x          = np.arange(n_gaussians)
        width      = 0.35

        pred_means = np.array([global_metrics.get(f"slot_{k}_mu_pred_mean", np.nan) for k in slots])
        pred_stds  = np.array([global_metrics.get(f"slot_{k}_mu_pred_std",  np.nan) for k in slots])
        gt_means   = np.array([global_metrics.get(f"slot_{k}_mu_gt_mean",   np.nan) for k in slots])
        gt_stds    = np.array([global_metrics.get(f"slot_{k}_mu_gt_std",    np.nan) for k in slots])

        fig, axes = plt.subplots(1, 2, figsize=(5.5 * n_gaussians, 4.5))

        ax = axes[0]
        ax.bar(x - width / 2, gt_means,   width, yerr=gt_stds,   color="C0", alpha=0.75, capsize=4, label="GT")
        ax.bar(x + width / 2, pred_means, width, yerr=pred_stds, color="C3", alpha=0.75, capsize=4, label="Pred")
        ax.set_xticks(x)
        ax.set_xticklabels([f"g{k+1}" for k in slots])
        ax.set_xlabel("Gaussian slot")
        ax.set_ylabel("µ  [m]")
        ax.set_title("Mean µ per slot  (active pixels, mean ± std)")
        ax.legend(framealpha=0.9)
        ax.grid(True, axis="y", which="major", linewidth=0.3, alpha=0.5)

        ax = axes[1]
        ax.bar(x - width / 2, gt_stds,   width, color="C0", alpha=0.75, label="GT")
        ax.bar(x + width / 2, pred_stds, width, color="C3", alpha=0.75, label="Pred")
        ax.set_xticks(x)
        ax.set_xticklabels([f"g{k+1}" for k in slots])
        ax.set_xlabel("Gaussian slot")
        ax.set_ylabel("std(µ)  [m]")
        ax.set_title("Std of µ per slot  (spread across pixels)")
        ax.legend(framealpha=0.9)
        ax.grid(True, axis="y", which="major", linewidth=0.3, alpha=0.5)

        fig.suptitle("Per-slot µ statistics  (active pixels only)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        return self._save(fig, out_path)

    def plot_placeholder_detection(
        self,
        global_metrics : dict,
        n_gaussians    : int,
        out_path       : Path,
    ) -> Path:
        slots      = list(range(n_gaussians))
        labels     = [f"g{k+1}" for k in slots] + ["all"]
        x          = np.arange(len(labels))
        width      = 0.25

        def _get(key: str) -> float:
            return float(global_metrics.get(key, np.nan))

        precisions = [_get(f"slot_{k}_placeholder_precision") for k in slots] + [_get("placeholder_precision")]
        recalls    = [_get(f"slot_{k}_placeholder_recall")    for k in slots] + [_get("placeholder_recall")]
        f1s        = [_get(f"slot_{k}_placeholder_f1")        for k in slots] + [_get("placeholder_f1")]
        gt_rates   = [_get(f"slot_{k}_placeholder_gt_rate")   for k in slots] + [_get("placeholder_gt_rate") if "placeholder_gt_rate" in global_metrics else np.nan]

        fig, axes = plt.subplots(1, 2, figsize=(max(8, 3.0 * len(labels)), 4.5))

        ax = axes[0]
        ax.bar(x - width,     precisions, width, color="C0", alpha=0.80, label="Precision")
        ax.bar(x,             recalls,    width, color="C2", alpha=0.80, label="Recall")
        ax.bar(x + width,     f1s,        width, color="C3", alpha=0.80, label="F1")
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

        # ── right: GT placeholder rate per slot ──────────────────────────────────
        ax = axes[1]
        colors = [f"C{k % 10}" for k in range(len(labels))]
        bars   = ax.bar(x, gt_rates, color=colors, alpha=0.75)
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

        fig.suptitle("Inactive-Gaussian detection metrics", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        return self._save(fig, out_path)

    def plot_slot_ordering_summary(
        self,
        global_metrics : dict,
        n_gaussians    : int,
        out_path       : Path,
    ) -> Path:
        ordering_rate   = float(global_metrics.get("mu_ordering_rate",                            np.nan))
        dominant_frac   = float(global_metrics.get("permutation_consensus_dominant_frac",         np.nan))
        identity_frac   = float(global_metrics.get("permutation_consensus_identity_frac",         np.nan))

        slots        = list(range(n_gaussians))
        active_rates = [1.0 - float(global_metrics.get(f"slot_{k}_placeholder_gt_rate", np.nan)) for k in slots]

        fig = plt.figure(figsize=(13, 4.5))
        gs  = fig.add_gridspec(1, 3, wspace=0.38)

        ax1     = fig.add_subplot(gs[0])
        labels  = ["µ ordering\nrate", "consensus\ndominant", "consensus\nidentity"]
        values  = [ordering_rate, dominant_frac, identity_frac]
        colors  = ["C0", "C2", "C3"]

        bars = ax1.barh(labels, values, color=colors, alpha=0.80)
        ax1.set_xlim(0, 1.08)
        ax1.set_xlabel("fraction")
        ax1.set_title("Slot organisation scalars")
        ax1.axvline(1.0, color="black", linewidth=0.7, linestyle="--")
        ax1.grid(True, axis="x", which="major", linewidth=0.3, alpha=0.5)
        for bar, val in zip(bars, values):
            if np.isfinite(val):
                ax1.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                         f"{val:.3f}", va="center", fontsize=9)

        ax2 = fig.add_subplot(gs[1])
        x   = np.arange(n_gaussians)
        ax2.bar(x, active_rates, color=[f"C{k % 10}" for k in slots], alpha=0.78)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"g{k+1}" for k in slots])
        ax2.set_ylim(0, 1.08)
        ax2.set_xlabel("Gaussian slot")
        ax2.set_ylabel("active-pixel fraction")
        ax2.set_title("GT activation rate per slot\n(1 − placeholder rate)")
        ax2.grid(True, axis="y", which="major", linewidth=0.3, alpha=0.5)
        for xi, val in enumerate(active_rates):
            if np.isfinite(val):
                ax2.text(xi, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        ax3 = fig.add_subplot(gs[2])
        pred_means = np.array([global_metrics.get(f"slot_{k}_mu_pred_mean", np.nan) for k in slots])
        gt_means   = np.array([global_metrics.get(f"slot_{k}_mu_gt_mean",   np.nan) for k in slots])
        pred_stds  = np.array([global_metrics.get(f"slot_{k}_mu_pred_std",  np.nan) for k in slots])
        gt_stds    = np.array([global_metrics.get(f"slot_{k}_mu_gt_std",    np.nan) for k in slots])

        ax3.errorbar(slots, gt_means,   yerr=gt_stds,   fmt="o-", color="C0", capsize=5, linewidth=1.2, label="GT",   markersize=6)
        ax3.errorbar(slots, pred_means, yerr=pred_stds, fmt="s--", color="C3", capsize=5, linewidth=1.2, label="Pred", markersize=6)
        ax3.set_xticks(slots)
        ax3.set_xticklabels([f"g{k+1}" for k in slots])
        ax3.set_xlabel("Gaussian slot")
        ax3.set_ylabel("µ  [m]")
        ax3.set_title("µ centre per slot  (mean ± std)")
        ax3.legend(framealpha=0.9)
        ax3.grid(True, which="major", linewidth=0.3, alpha=0.5)

        fig.suptitle("Slot organisation summary", fontsize=13)

        return self._save(fig, out_path)

    def plot_active_count_map(
        self,
        params_pred : np.ndarray,
        params_gt   : np.ndarray,
        n_gaussians : int,
        out_path    : Path,
        az_offset   : int,
        rg_offset   : int,
        amp_threshold: float = 1e-3,
    ) -> Path:

        gt_count   = np.zeros(params_gt  .shape[-2:], dtype=np.int32)
        pred_count = np.zeros(params_pred.shape[-2:], dtype=np.int32)
        
        for k in range(n_gaussians):
            gt_count   += (params_gt  [3 * k] >= amp_threshold).astype(np.int32)
            pred_count += (params_pred[3 * k] >= amp_threshold).astype(np.int32)

        diff = pred_count - gt_count   

        H, W  = diff.shape
        rgb   = np.zeros((H, W, 3), dtype=np.float32)
        rgb[diff == 0]  = [0.20, 0.75, 0.20]   
        rgb[diff <  0]  = [0.20, 0.45, 0.90]   
        rgb[diff >  0]  = [0.90, 0.25, 0.25]   

        extent = [az_offset, az_offset + W, az_offset + H, az_offset]

        n_total   = H * W
        n_correct = int((diff == 0).sum())
        n_under   = int((diff <  0).sum())
        n_over    = int((diff >  0).sum())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.imshow(rgb, aspect="auto", interpolation="nearest", extent=[az_offset, az_offset + W, az_offset + H, az_offset])
        ax.set_xlabel("range index")
        ax.set_ylabel("azimuth index")
        ax.set_title("Active-count agreement per pixel")

        from matplotlib.patches import Patch
        legend_els = [
            Patch(facecolor=[0.20, 0.75, 0.20], label=f"correct  ({n_correct/n_total*100:.1f}%)"),
            Patch(facecolor=[0.20, 0.45, 0.90], label=f"under    ({n_under  /n_total*100:.1f}%)"),
            Patch(facecolor=[0.90, 0.25, 0.25], label=f"over     ({n_over   /n_total*100:.1f}%)"),
        ]
        ax.legend(handles=legend_els, loc="lower right", framealpha=0.9, fontsize=9)

        ax2  = axes[1]
        vabs = max(1, int(np.abs(diff).max()))
        im   = ax2.imshow(diff, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto", interpolation="nearest", extent=[az_offset, az_offset + W, az_offset + H, az_offset])
        ax2.set_xlabel("range index")
        ax2.set_ylabel("azimuth index")
        ax2.set_title("Signed count difference  (pred − GT)")
        cb = fig.colorbar(im, ax=ax2, fraction=0.045, pad=0.02)
        cb.set_label("pred − GT  [#Gaussians]")
        cb.set_ticks(range(-vabs, vabs + 1))

        fig.suptitle(
            f"Active-Gaussian count  |  correct {n_correct/n_total*100:.1f}%  "
            f"under {n_under/n_total*100:.1f}%  over {n_over/n_total*100:.1f}%",
            fontsize=13,
        )
        
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        return self._save(fig, out_path)
