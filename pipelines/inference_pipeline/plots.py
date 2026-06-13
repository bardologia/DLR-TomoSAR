from __future__ import annotations

from pathlib import Path
from typing  import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm        as cm
import matplotlib.pyplot    as plt
import numpy                as np
from matplotlib.patches     import Patch

from pipelines.shared.plotting import PlotBase
from tools.gaussians           import GaussianReconstructor


class PlotTools(PlotBase):
    PARAM_LABELS = (("a", "amplitude (a)"), ("mu", "mean (μ)"), ("sigma", "std-dev (σ)"))
    PARAM_SHORT  = ("a", "μ", "σ")

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

    @staticmethod
    def _gaussian_components(params: np.ndarray, x_axis: np.ndarray, n_gaussians: int) -> List[np.ndarray]:
        return GaussianReconstructor.components(params, x_axis, n_gaussians)

    def _intensity_scale(self, reference: np.ndarray) -> float:
        if not self.normalize:
            return 1.0

        ref   = reference[np.isfinite(reference)]
        scale = float(ref.max()) if ref.size else 0.0

        return scale if scale > 1e-12 else 1.0

    @staticmethod
    def _rescale(arr: np.ndarray, scale: float) -> np.ndarray:
        return (arr / scale).astype(np.float32)

    @property
    def _int_label(self) -> str:
        return "intensity (GT-peak normalised)" if self.normalize else "intensity"

    @property
    def _err_label(self) -> str:
        return "|error| (GT-peak normalised)" if self.normalize else "|error| (intensity)"

    def _imshow_panel(
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

        return self._imshow_figure(
            data,
            x_label        = x_label,
            y_label        = y_label,
            title          = title,
            cmap           = cmap,
            vmin           = vmin,
            vmax           = vmax,
            extent         = extent,
            origin         = origin,
            colorbar_label = cbar_label,
            figsize        = figsize,
            path           = path,
        )


class SlicePlotter(PlotTools):
    def plot_profiles(
        self,
        pred_curves    : np.ndarray,
        gt_curves      : np.ndarray,
        params_pred    : Optional[np.ndarray],
        x_axis         : np.ndarray,
        pixels         : np.ndarray,
        tag            : str,
        out_dir        : Path,
        n_gaussians    : int,
        pixel_metrics  : Dict[str, np.ndarray],
        az_offset      : int,
        rg_offset      : int,
        reduced_curves : Optional[np.ndarray] = None,
        full_curves    : Optional[np.ndarray] = None,
    ) -> List[Path]:

        base_colors = [cm.tab10(i) for i in range(10)]
        paths       = []

        for p, (y, x) in enumerate(pixels):
            gt    = gt_curves  [:, y, x]
            pred  = pred_curves[:, y, x]
            comps = self._gaussian_components(params_pred[:, y, x], x_axis, n_gaussians) if params_pred is not None else []

            fig, ax = plt.subplots(figsize=(5.2, 3.4))

            if full_curves is not None:
                ax.plot(x_axis, full_curves[:, y, x], color="0.55", linewidth=1.0, label="Full tomo (raw)", zorder=1)

            ax.plot(x_axis, gt,   color="black", linewidth=1.4, label="GT",   zorder=3)
            ax.plot(x_axis, pred, color="C3",    linewidth=1.2, label="Pred", linestyle="--", zorder=4)

            if reduced_curves is not None:
                reduced   = reduced_curves[:, y, x]
                red_peak  = float(np.nanmax(reduced))
                gt_peak   = float(np.nanmax(gt))
                red_scale = (gt_peak / red_peak) if red_peak > 1e-12 else 1.0
                ax.plot(x_axis, reduced * red_scale, color="C2", linewidth=1.1, label="Reduced (GT-peak)", linestyle=":", zorder=2)

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

        return self._imshow_panel(
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

    def plot_input_channels(
        self,
        complex_inputs : np.ndarray,
        n_secondaries  : int,
        labels         : Optional[List[str]],
        out_dir        : Path,
        az_offset      : int,
        rg_offset      : int,
        primary_label  : str = "primary",
    ) -> List[Path]:

        H, W   = complex_inputs.shape[-2:]
        extent = [rg_offset, rg_offset + W, az_offset + H, az_offset]
        names  = list(labels) if labels else [f"secondary {i + 1}" for i in range(n_secondaries)]
        paths  = []

        paths.append(self._amplitude_panel(complex_inputs[0], f"Primary {primary_label} — amplitude", out_dir / "pass_primary_amplitude.png", extent))

        for i in range(n_secondaries):
            paths.append(self._amplitude_panel(complex_inputs[1 + i], f"Secondary {names[i]} — amplitude", out_dir / f"pass_{names[i]}_amplitude.png", extent))

        for i in range(n_secondaries):
            phase = np.angle(complex_inputs[1 + n_secondaries + i]).astype(np.float32)

            paths.append(self._imshow_panel(
                data       = phase,
                title      = f"Interferogram {primary_label} × {names[i]} — phase",
                x_label    = "range index",
                y_label    = "azimuth index",
                cbar_label = "phase [rad]",
                extent     = extent,
                cmap       = "twilight",
                vmin       = -np.pi,
                vmax       = np.pi,
                origin     = "upper",
                path       = out_dir / f"interferogram_{names[i]}_phase.png",
            ))

        return paths

    def _amplitude_panel(self, channel: np.ndarray, title: str, path: Path, extent: list) -> Path:
        amplitude = np.abs(channel).astype(np.float32)
        vmax      = float(np.percentile(amplitude, 99.0))

        return self._imshow_panel(
            data       = amplitude,
            title      = title,
            x_label    = "range index",
            y_label    = "azimuth index",
            cbar_label = "amplitude",
            extent     = extent,
            cmap       = "gray",
            vmin       = 0.0,
            vmax       = max(vmax, 1e-6),
            origin     = "upper",
            path       = path,
            figsize    = (7, 5),
        )

    def _render_slice_panels(
        self,
        pred_slice    : np.ndarray,
        gt_slice      : np.ndarray,
        extent        : list,
        x_label       : str,
        y_label       : str,
        origin        : str,
        title_pos     : str,
        ssim_value    : Optional[float],
        out_dir       : Path,
        stem          : str,
        ref_title     : str = "GT (Gaussian)",
        pred_title    : str = "Prediction",
        err_title     : str = "|Pred − GT|",
        full_slice    : Optional[np.ndarray] = None,
        full_title    : str = "Full tomogram (raw)",
    ) -> List[Path]:

        err_gt_slice = np.abs(pred_slice - gt_slice)

        scale        = self._intensity_scale(gt_slice)
        gt_slice     = self._rescale(gt_slice,     scale)
        pred_slice   = self._rescale(pred_slice,   scale)
        err_gt_slice = self._rescale(err_gt_slice, scale)

        vmin, vmax = self._shared_clim(gt_slice, pred_slice)
        emax_gt    = float(np.percentile(err_gt_slice, 99.0))
        ssim_str   = f"   SSIM = {ssim_value:.4f}" if ssim_value is not None and np.isfinite(ssim_value) else ""

        panels = [
            (gt_slice,     f"{ref_title} — {title_pos}",          self.cmap,     vmin, vmax,    self._int_label, "gt"),
            (pred_slice,   f"{pred_title} — {title_pos}{ssim_str}", self.cmap,   vmin, vmax,    self._int_label, "pred"),
            (err_gt_slice, f"{err_title} — {title_pos}",           self.err_cmap, 0.0,  emax_gt, self._err_label, "error"),
        ]

        if full_slice is not None:
            full_rescaled = self._rescale(full_slice, scale)
            panels.insert(0, (full_rescaled, f"{full_title} — {title_pos}", self.cmap, vmin, vmax, self._int_label, "full"))

        return [
            self._imshow_panel(
                data       = data,
                title      = title,
                x_label    = x_label,
                y_label    = y_label,
                cbar_label = cbar,
                extent     = extent,
                cmap       = cmap_used,
                vmin       = vlo,
                vmax       = vhi,
                origin     = origin,
                path       = out_dir / f"{stem}_{kind}.png",
            )
            for data, title, cmap_used, vlo, vhi, cbar, kind in panels
        ]

    def plot_tomogram_slice(
        self,
        pred_cube    : np.ndarray,
        gt_cube      : np.ndarray,
        axis         : str,
        index        : int,
        x_axis       : np.ndarray,
        out_dir      : Path,
        stem         : str,
        az_offset    : int,
        rg_offset    : int,
        ssim_value   : Optional[float] = None,
        ref_title    : str = "GT (Gaussian)",
        pred_title   : str = "Prediction",
        err_title    : str = "|Pred − GT|",
        full_cube    : Optional[np.ndarray] = None,
        full_title   : str = "Full tomogram (raw)",
    ) -> List[Path]:

        if axis == "range":
            pred_slice               = pred_cube[:, :, index]
            gt_slice                 = gt_cube  [:, :, index]
            full_slice               = full_cube[:, :, index] if full_cube is not None else None
            x_label                  = "azimuth index"
            x_extent_lo, x_extent_hi = az_offset, az_offset + pred_slice.shape[1]
            title_pos                = f"range = {index + rg_offset}"

        elif axis == "azimuth":
            pred_slice               = pred_cube[:, index, :]
            gt_slice                 = gt_cube  [:, index, :]
            full_slice               = full_cube[:, index, :] if full_cube is not None else None
            x_label                  = "range index"
            x_extent_lo, x_extent_hi = rg_offset, rg_offset + pred_slice.shape[1]
            title_pos                = f"azimuth = {index + az_offset}"

        else:
            raise ValueError(f"axis must be 'range' or 'azimuth', got {axis!r}")

        sort_idx      = np.argsort(x_axis)
        x_axis_sorted = x_axis[sort_idx]
        pred_slice    = pred_slice[sort_idx]
        gt_slice      = gt_slice  [sort_idx]
        full_slice    = full_slice[sort_idx] if full_slice is not None else None

        extent = [x_extent_lo, x_extent_hi, float(x_axis_sorted[0]), float(x_axis_sorted[-1])]

        return self._render_slice_panels(
            pred_slice = pred_slice,
            gt_slice   = gt_slice,
            extent     = extent,
            x_label    = x_label,
            y_label    = "elevation [m]",
            origin     = "lower",
            title_pos  = title_pos,
            ssim_value = ssim_value,
            out_dir    = out_dir,
            stem       = stem,
            ref_title  = ref_title,
            pred_title = pred_title,
            err_title  = err_title,
            full_slice = full_slice,
            full_title = full_title,
        )

    def plot_elevation_intensity_slice(
        self,
        pred_cube    : np.ndarray,
        gt_cube      : np.ndarray,
        elev_idx     : int,
        x_axis       : np.ndarray,
        out_dir      : Path,
        stem         : str,
        az_offset    : int,
        rg_offset    : int,
        ssim_value   : Optional[float] = None,
        ref_title    : str = "GT (Gaussian)",
        pred_title   : str = "Prediction",
        err_title    : str = "|Pred − GT|",
        full_cube    : Optional[np.ndarray] = None,
        full_title   : str = "Full tomogram (raw)",
    ) -> List[Path]:

        pred_slice = pred_cube[elev_idx]
        gt_slice   = gt_cube  [elev_idx]
        full_slice = full_cube[elev_idx] if full_cube is not None else None

        H, W      = pred_slice.shape
        extent    = [rg_offset, rg_offset + W, az_offset + H, az_offset]
        title_pos = f"elev = {x_axis[elev_idx]:.2f} m (idx {elev_idx})"

        return self._render_slice_panels(
            pred_slice = pred_slice,
            gt_slice   = gt_slice,
            extent     = extent,
            x_label    = "range index",
            y_label    = "azimuth index",
            origin     = "upper",
            title_pos  = title_pos,
            ssim_value = ssim_value,
            out_dir    = out_dir,
            stem       = stem,
            ref_title  = ref_title,
            pred_title = pred_title,
            err_title  = err_title,
            full_slice = full_slice,
            full_title = full_title,
        )

    def plot_metric_histograms(self, metric_arrays: Dict[str, np.ndarray], out_dir: Path) -> List[Path]:
        paths = []

        for name, arr in metric_arrays.items():
            flat = arr.reshape(-1)
            flat = flat[np.isfinite(flat)]
            if flat.size == 0:
                continue

            lo, hi    = np.percentile(flat, [0.5, 99.5])
            clipped   = np.clip(flat, lo, hi)
            n_outside = int(np.count_nonzero((flat < lo) | (flat > hi)))
            fig, ax   = plt.subplots(figsize=(4.8, 3.4))
            ax.hist(clipped, bins=80, color="C0", edgecolor="white", linewidth=0.3)
            ax.axvline(float(np.median(flat)), color="black", linestyle="--", linewidth=1.0, label=f"median={np.median(flat):.3g}")
            ax.set_title(f"{name} (denorm)  [{n_outside} of {flat.size} outliers piled at edges]", fontsize=8)
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
        prefix         : str = "gt",
        series_label   : str = "pred × GT (Gaussian)",
    ) -> Path:

        vals_gt = np.array([global_metrics[f"ssim_{prefix}_{axis}_{i}"]  for i in range(n_slices)], dtype=np.float64)
        x_phys  = slice_indices.astype(np.float64) + ax_offset

        mean_gt = float(global_metrics[f"ssim_{prefix}_{axis}_mean"])

        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.plot(x_phys, vals_gt, color="C0", linewidth=0.9, label=series_label, alpha=0.9)
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
        suffix         : str = "gt",
        series_label   : str = "pred × GT (Gaussian)",
    ) -> List[Path]:

        metric_specs = [
            ("elev_mae",  "mae",           "MAE",           "mean absolute error"),
            ("elev_rmse", "rmse",          "RMSE",          "root mean squared error"),
            ("elev_r2",   "r2",            "R²",            "coefficient of determination"),
            ("elev_ce",   "cross_entropy", "cross-entropy", "cross-entropy (normalised profiles)"),
        ]

        paths = []
        for key, fname, ylabel, desc in metric_specs:
            vals_gt = np.array([global_metrics[f"{key}_{suffix}_{i}"]  for i in range(n_elev)], dtype=np.float64)
            mean_gt = float(global_metrics[f"{key}_{suffix}_mean"])

            fig, ax = plt.subplots(figsize=(5.8, 3.6))
            ax.plot(x_axis, vals_gt, color="C0", linewidth=0.9, label=series_label, alpha=0.9)
            ax.axhline(mean_gt, color="C0", linestyle="--", linewidth=1.0, label=f"mean = {mean_gt:.4g}")

            ax.set_xlabel("elevation [m]")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} per elevation bin — {desc}")
            ax.legend(framealpha=0.9)
            ax.grid(True, axis="y", which="major", linewidth=0.3, alpha=0.5)
            fig.tight_layout()

            paths.append(self._save(fig, out_dir / f"{fname}.png"))

        return paths


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

        step             = max(1, len(azimuth) // 200)
        theta            = np.linspace(0.0, 2.0 * np.pi, 36)
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


class Ploter(PlotTools):
    def __init__(
        self,
        cmap     : str  = "jet",
        err_cmap : str  = "magma",
        normalize: bool = False,
        fig_dpi  : int  = 150,
        save_dpi : int  = 150,
    ) -> None:

        super().__init__(cmap=cmap, err_cmap=err_cmap, normalize=normalize, fig_dpi=fig_dpi, save_dpi=save_dpi)

        self.slice = SlicePlotter(cmap=cmap, err_cmap=err_cmap, normalize=normalize, fig_dpi=fig_dpi, save_dpi=save_dpi)
        self.param = ParamPlotter(cmap=cmap, err_cmap=err_cmap, normalize=normalize, fig_dpi=fig_dpi, save_dpi=save_dpi)
        self.slot  = SlotPlotter( cmap=cmap, err_cmap=err_cmap, normalize=normalize, fig_dpi=fig_dpi, save_dpi=save_dpi)
        self.track = TrackPlotter(cmap=cmap, err_cmap=err_cmap, normalize=normalize, fig_dpi=fig_dpi, save_dpi=save_dpi)
