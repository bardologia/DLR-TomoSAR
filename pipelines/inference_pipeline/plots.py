from __future__ import annotations

from pathlib import Path
from typing  import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm        as cm
import matplotlib.pyplot    as plt
import numpy                as np


SCIENTIFIC_RC = {
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


def apply_style(fig_dpi: int = 150, save_dpi: int = 300) -> None:
    plt.rcParams.update(SCIENTIFIC_RC)
    plt.rcParams["figure.dpi"]  = fig_dpi
    plt.rcParams["savefig.dpi"] = save_dpi


def _gaussian_components(params: np.ndarray, x_axis: np.ndarray, n_gaussians: int) -> List[np.ndarray]:
    out = []
    for k in range(n_gaussians):
        a    = float(params[3 * k])
        mu   = float(params[3 * k + 1])
        sig  = float(params[3 * k + 2])
        comp = a * np.exp(-((x_axis - mu) ** 2) / (2.0 * sig * sig + 1e-8))
        out.append(comp)
    return out


def _shared_clim(*arrays: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> Tuple[float, float]:
    flat = np.concatenate([a.reshape(-1) for a in arrays])
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return (0.0, 1.0)
    return float(np.percentile(flat, q_low)), float(np.percentile(flat, q_high))


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_profile_panel(
    pred_curves   : np.ndarray,
    gt_curves     : np.ndarray,
    raw_curves    : np.ndarray,
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
        return _save(fig, out_path)

    n_cols     = min(n_cols, P)
    n_rows     = int(np.ceil(P / n_cols))
    fig, axes  = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.6 * n_rows), squeeze=False)

    base_colors = [cm.tab10(i) for i in range(10)]

    for ax in axes.ravel():
        ax.set_visible(False)

    for p, (y, x) in enumerate(pixels):
        r, c = divmod(p, n_cols)
        ax   = axes[r, c]
        ax.set_visible(True)

        raw   = raw_curves [:, y, x]
        gt    = gt_curves  [:, y, x]
        pred  = pred_curves[:, y, x]
        comps = _gaussian_components(params_pred[:, y, x], x_axis, n_gaussians)

        ax.plot(x_axis, raw,  color="C0",    linewidth=1.0, label="Raw",  linestyle=":",  zorder=2)
        ax.plot(x_axis, gt,   color="black", linewidth=1.4, label="GT",   zorder=3)
        ax.plot(x_axis, pred, color="C3",    linewidth=1.2, label="Pred", linestyle="--", zorder=4)
        for k, comp in enumerate(comps):
            ax.plot(x_axis, comp, color=base_colors[k % len(base_colors)],
                    linewidth=0.8, alpha=0.75, label=f"$g_{{{k+1}}}$" if p == 0 else None)
        ax.fill_between(x_axis, pred - gt, 0.0, color="C0", alpha=0.10, linewidth=0)

        mse     = float(pixel_metrics["mse"]    [y, x])
        r2      = float(pixel_metrics["r2"]     [y, x])
        cos     = float(pixel_metrics["cos"]    [y, x])
        mse_raw = float(pixel_metrics["mse_raw"][y, x])
        r2_raw  = float(pixel_metrics["r2_raw"] [y, x])
        cos_raw = float(pixel_metrics["cos_raw"][y, x])
        ax.set_title(
            f"az={y + az_offset}, rg={x + rg_offset}\n"
            f"pred/gt  MSE={mse:.3g}  R²={r2:.3f}  cos={cos:.3f}\n"
            f"pred/raw MSE={mse_raw:.3g}  R²={r2_raw:.3f}  cos={cos_raw:.3f}",
            fontsize=8,
        )
        ax.set_xlabel("elevation [m]")
        ax.set_ylabel("intensity")
        ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
        if p == 0:
            ax.legend(loc="best", framealpha=0.85)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out_path)


def plot_pixel_metric_map(
    metric_map : np.ndarray,
    title      : str,
    label      : str,
    out_path   : Path,
    az_offset  : int,
    rg_offset  : int,
    cmap       : str   = "viridis",
    log        : bool  = False,
    q_low      : float = 1.0,
    q_high     : float = 99.0,
) -> Path:
    H, W   = metric_map.shape
    extent = [rg_offset, rg_offset + W, az_offset + H, az_offset]

    data = metric_map.copy()
    if log:
        data = np.log10(np.clip(data, 1e-12, None))

    vmin, vmax = _shared_clim(data, q_low=q_low, q_high=q_high)
    fig, ax    = plt.subplots(figsize=(7, 5))
    im         = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, aspect="auto", interpolation="nearest")
    ax.set_xlabel("range index")
    ax.set_ylabel("azimuth index")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    cb.set_label(label + (" [log10]" if log else ""))
    return _save(fig, out_path)


def plot_tomogram_slice(
    pred_cube  : np.ndarray,
    gt_cube    : np.ndarray,
    raw_cube   : np.ndarray,
    axis       : str,
    index      : int,
    x_axis     : np.ndarray,
    out_path   : Path,
    az_offset  : int,
    rg_offset  : int,
    cmap       : str            = "viridis",
    err_cmap   : str            = "magma",
    ssim_value : Optional[float] = None,
) -> Path:
    if axis == "range":
        pred_slice               = pred_cube[:, :, index]
        gt_slice                 = gt_cube  [:, :, index]
        raw_slice                = raw_cube [:, :, index]
        x_label                  = "azimuth index"
        x_extent_lo, x_extent_hi = az_offset, az_offset + pred_slice.shape[1]
        title_pos                = f"range = {index + rg_offset}"
    elif axis == "azimuth":
        pred_slice               = pred_cube[:, index, :]
        gt_slice                 = gt_cube  [:, index, :]
        raw_slice                = raw_cube [:, index, :]
        x_label                  = "range index"
        x_extent_lo, x_extent_hi = rg_offset, rg_offset + pred_slice.shape[1]
        title_pos                = f"azimuth = {index + az_offset}"
    else:
        raise ValueError(f"axis must be 'range' or 'azimuth', got {axis!r}")

    err_gt_slice  = np.abs(pred_slice - gt_slice)
    err_raw_slice = np.abs(pred_slice - raw_slice)
    extent_int    = [x_extent_lo, x_extent_hi, float(x_axis[-1]), float(x_axis[0])]

    vmin, vmax = _shared_clim(raw_slice, gt_slice, pred_slice)
    emax_gt    = float(np.percentile(err_gt_slice,  99.0))
    emax_raw   = float(np.percentile(err_raw_slice, 99.0))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.0), sharey=True)
    top_panels = [
        (raw_slice,     "Raw Tomogram",  cmap,     vmin, vmax),
        (gt_slice,      "GT (Gaussian)", cmap,     vmin, vmax),
        (pred_slice,    "Prediction",    cmap,     vmin, vmax),
    ]
    bot_panels = [
        (err_gt_slice,  "|Pred - GT|",   err_cmap, 0.0,  emax_gt),
        (err_raw_slice, "|Pred - Raw|",  err_cmap, 0.0,  emax_raw),
    ]
    for ax_i, (data, label, cm_used, vlo, vhi) in zip(axes[0], top_panels):
        im = ax_i.imshow(data, aspect="auto", extent=extent_int, cmap=cm_used, vmin=vlo, vmax=vhi)
        ax_i.set_title(label)
        ax_i.set_xlabel(x_label)
        fig.colorbar(im, ax=ax_i, fraction=0.045, pad=0.02).set_label("intensity")
    for ax_i, (data, label, cm_used, vlo, vhi) in zip(axes[1], bot_panels):
        im = ax_i.imshow(data, aspect="auto", extent=extent_int, cmap=cm_used, vmin=vlo, vmax=vhi)
        ax_i.set_title(label)
        ax_i.set_xlabel(x_label)
        fig.colorbar(im, ax=ax_i, fraction=0.045, pad=0.02).set_label("|error|")
    axes[1, 2].set_visible(False)
    axes[0, 0].set_ylabel("elevation [m]")
    axes[1, 0].set_ylabel("elevation [m]")
    ssim_str = f"   SSIM = {ssim_value:.4f}" if ssim_value is not None and np.isfinite(ssim_value) else ""
    fig.suptitle(f"Tomogram slice ({axis}-cut) — {title_pos}{ssim_str}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out_path)


def plot_elevation_intensity_slice(
    pred_cube  : np.ndarray,
    gt_cube    : np.ndarray,
    raw_cube   : np.ndarray,
    elev_idx   : int,
    x_axis     : np.ndarray,
    out_path   : Path,
    az_offset  : int,
    rg_offset  : int,
    cmap       : str             = "viridis",
    err_cmap   : str             = "magma",
    ssim_value : Optional[float] = None,
) -> Path:
    pred_slice    = pred_cube[elev_idx]
    gt_slice      = gt_cube  [elev_idx]
    raw_slice     = raw_cube [elev_idx]
    err_gt_slice  = np.abs(pred_slice - gt_slice)
    err_raw_slice = np.abs(pred_slice - raw_slice)

    H, W       = pred_slice.shape
    extent     = [rg_offset, rg_offset + W, az_offset + H, az_offset]
    vmin, vmax = _shared_clim(raw_slice, gt_slice, pred_slice)
    emax_gt    = float(np.percentile(err_gt_slice,  99.0))
    emax_raw   = float(np.percentile(err_raw_slice, 99.0))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    top_panels = [
        (raw_slice,     "Raw Tomogram",  cmap,     vmin, vmax),
        (gt_slice,      "GT (Gaussian)", cmap,     vmin, vmax),
        (pred_slice,    "Prediction",    cmap,     vmin, vmax),
    ]
    bot_panels = [
        (err_gt_slice,  "|Pred - GT|",   err_cmap, 0.0,  emax_gt),
        (err_raw_slice, "|Pred - Raw|",  err_cmap, 0.0,  emax_raw),
    ]
    for ax_i, (data, label, cm_used, vlo, vhi) in zip(axes[0], top_panels):
        im = ax_i.imshow(data, cmap=cm_used, vmin=vlo, vmax=vhi, extent=extent, aspect="auto")
        ax_i.set_title(label)
        ax_i.set_xlabel("range index")
        ax_i.set_ylabel("azimuth index")
        fig.colorbar(im, ax=ax_i, fraction=0.045, pad=0.02).set_label("intensity")
    for ax_i, (data, label, cm_used, vlo, vhi) in zip(axes[1], bot_panels):
        im = ax_i.imshow(data, cmap=cm_used, vmin=vlo, vmax=vhi, extent=extent, aspect="auto")
        ax_i.set_title(label)
        ax_i.set_xlabel("range index")
        ax_i.set_ylabel("azimuth index")
        fig.colorbar(im, ax=ax_i, fraction=0.045, pad=0.02).set_label("|error|")
    axes[1, 2].set_visible(False)
    ssim_str = f"   SSIM = {ssim_value:.4f}" if ssim_value is not None and np.isfinite(ssim_value) else ""
    fig.suptitle(f"Elevation slice (elev = {x_axis[elev_idx]:.2f} m, idx={elev_idx}){ssim_str}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out_path)


def plot_metric_histogram(metric_arrays: Dict[str, np.ndarray], out_path: Path) -> Path:
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
        ax.axvline(float(np.median(flat)), color="black", linestyle="--", linewidth=1.0,
                   label=f"median={np.median(flat):.3g}")
        ax.set_title(name)
        ax.set_xlabel(name)
        ax.set_ylabel("count")
        ax.set_yscale("log")
        ax.legend(framealpha=0.9)
        ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    return _save(fig, out_path)


_SSIM_MAX_XTICKS = 40   # max labelled x-ticks


def plot_ssim_curves(
    global_metrics : dict,
    axis           : str,
    out_path       : Path,
    n_slices       : int,
    slice_indices  : np.ndarray,
    ax_offset      : int = 0,
) -> Path:
    """Line chart of SSIM per slice for both pred×gt and pred×raw comparisons."""
    vals_gt  = np.array([global_metrics.get(f"ssim_{axis}_{i}",     float("nan")) for i in range(n_slices)], dtype=np.float64)
    vals_raw = np.array([global_metrics.get(f"raw_ssim_{axis}_{i}", float("nan")) for i in range(n_slices)], dtype=np.float64)
    x_phys   = slice_indices.astype(np.float64) + ax_offset

    mean_gt  = float(np.nanmean(vals_gt))
    mean_raw = float(np.nanmean(vals_raw))

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(x_phys, vals_gt,  color="C0", linewidth=0.9, label="pred × GT (Gaussian)", alpha=0.9)
    ax.plot(x_phys, vals_raw, color="C3", linewidth=0.9, label="pred × Raw",           alpha=0.9)
    ax.axhline(mean_gt,  color="C0", linestyle="--", linewidth=1.0, label=f"mean GT  = {mean_gt:.4f}")
    ax.axhline(mean_raw, color="C3", linestyle="--", linewidth=1.0, label=f"mean Raw = {mean_raw:.4f}")

    tick_locs = np.linspace(x_phys[0], x_phys[-1], min(_SSIM_MAX_XTICKS, n_slices)).round().astype(int)
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_locs, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel(f"{axis} index")
    ax.set_ylabel("SSIM")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"SSIM per {axis}-slice  (n={n_slices})")
    ax.legend(framealpha=0.9)
    ax.grid(True, axis="y", which="major", linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_param_maps(
    params_pred : np.ndarray,
    params_gt   : Optional[np.ndarray],
    n_gaussians : int,
    param_names : List[str],
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
            vmin, vmax = _shared_clim(arr_pred if params_gt is None else np.stack([arr_pred, params_gt[ch]]))

            ax = axes[row, 0]
            im = ax.imshow(arr_pred, cmap="viridis", vmin=vmin, vmax=vmax, extent=extent, aspect="auto")
            ax.set_title(f"Pred {sub}{k+1}")
            ax.set_xlabel("range")
            ax.set_ylabel("azimuth")
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

            if params_gt is not None and ch < params_gt.shape[0]:
                ax2 = axes[row, 1]
                im2 = ax2.imshow(params_gt[ch], cmap="viridis", vmin=vmin, vmax=vmax, extent=extent, aspect="auto")
                ax2.set_title(f"GT {sub}{k+1}")
                ax2.set_xlabel("range")
                fig.colorbar(im2, ax=ax2, fraction=0.045, pad=0.02)

    fig.suptitle("Gaussian parameter maps", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    return _save(fig, out_path)
