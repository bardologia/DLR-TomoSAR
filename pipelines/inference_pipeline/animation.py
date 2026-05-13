from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot    as plt
import numpy                as np

from pipelines.inference_pipeline.plots import _shared_clim, apply_style


def make_walk_gif(
    pred_cube  : np.ndarray,
    gt_cube    : np.ndarray,
    raw_cube   : np.ndarray,
    axis       : str,
    out_path   : Path,
    *,
    x_axis     : np.ndarray,
    az_offset  : int,
    rg_offset  : int,
    fps        : int = 12,
    max_frames : int = 150,
    dpi        : int = 110,
    cmap       : str = "viridis",
    err_cmap   : str = "magma",
) -> Path:
    apply_style(fig_dpi=dpi, save_dpi=dpi)

    N_elev, az, rg = pred_cube.shape

    if axis == "elevation":
        n_total = N_elev
        def get_slice(i):
            return pred_cube[i], gt_cube[i], raw_cube[i]
        extent           = [rg_offset, rg_offset + rg, az_offset + az, az_offset]
        x_label, y_label = "range index", "azimuth index"
        title_fn         = lambda i: f"elevation = {x_axis[i]:.2f} m  (idx {i}/{N_elev - 1})"
    elif axis == "range":
        n_total = rg
        def get_slice(i):
            return pred_cube[:, :, i], gt_cube[:, :, i], raw_cube[:, :, i]
        extent           = [az_offset, az_offset + az, float(x_axis[-1]), float(x_axis[0])]
        x_label, y_label = "azimuth index", "elevation [m]"
        title_fn         = lambda i: f"range = {i + rg_offset}"
    elif axis == "azimuth":
        n_total = az
        def get_slice(i):
            return pred_cube[:, i, :], gt_cube[:, i, :], raw_cube[:, i, :]
        extent           = [rg_offset, rg_offset + rg, float(x_axis[-1]), float(x_axis[0])]
        x_label, y_label = "range index", "elevation [m]"
        title_fn         = lambda i: f"azimuth = {i + az_offset}"
    else:
        raise ValueError(f"axis must be elevation|range|azimuth, got {axis!r}")

    if n_total > max_frames:
        frame_indices = np.linspace(0, n_total - 1, max_frames).round().astype(int)
    else:
        frame_indices = np.arange(n_total)

    sample_idx  = frame_indices[:: max(1, len(frame_indices) // 16)]
    pred_sample = np.stack([get_slice(int(i))[0] for i in sample_idx])
    gt_sample   = np.stack([get_slice(int(i))[1] for i in sample_idx])
    raw_sample  = np.stack([get_slice(int(i))[2] for i in sample_idx])
    vmin, vmax  = _shared_clim(pred_sample, gt_sample, raw_sample)
    emax_gt     = float(np.percentile(np.abs(pred_sample - gt_sample),  99.0))
    emax_raw    = float(np.percentile(np.abs(pred_sample - raw_sample), 99.0))
    if emax_gt  <= 0.0: emax_gt  = 1.0
    if emax_raw <= 0.0: emax_raw = 1.0

    fig, axes          = plt.subplots(2, 3, figsize=(18, 8.0))
    p0, g0, r0         = get_slice(int(frame_indices[0]))
    eg0                = np.abs(p0 - g0)
    er0                = np.abs(p0 - r0)

    im_raw  = axes[0, 0].imshow(r0,  cmap=cmap,     vmin=vmin,  vmax=vmax,    extent=extent, aspect="auto")
    im_gt   = axes[0, 1].imshow(g0,  cmap=cmap,     vmin=vmin,  vmax=vmax,    extent=extent, aspect="auto")
    im_pred = axes[0, 2].imshow(p0,  cmap=cmap,     vmin=vmin,  vmax=vmax,    extent=extent, aspect="auto")
    im_egt  = axes[1, 0].imshow(eg0, cmap=err_cmap, vmin=0.0,   vmax=emax_gt, extent=extent, aspect="auto")
    im_eraw = axes[1, 1].imshow(er0, cmap=err_cmap, vmin=0.0,   vmax=emax_raw,extent=extent, aspect="auto")
    axes[1, 2].set_visible(False)

    for ax, label in zip(axes[0], ("Raw Tomogram", "GT (Gaussian)", "Prediction")):
        ax.set_title(label)
        ax.set_xlabel(x_label)
    for ax, label in zip(axes[1, :2], ("|Pred - GT|", "|Pred - Raw|")):
        ax.set_title(label)
        ax.set_xlabel(x_label)
    axes[0, 0].set_ylabel(y_label)
    axes[1, 0].set_ylabel(y_label)

    fig.colorbar(im_raw,  ax=axes[0, 0], fraction=0.045, pad=0.02).set_label("intensity")
    fig.colorbar(im_gt,   ax=axes[0, 1], fraction=0.045, pad=0.02).set_label("intensity")
    fig.colorbar(im_pred, ax=axes[0, 2], fraction=0.045, pad=0.02).set_label("intensity")
    fig.colorbar(im_egt,  ax=axes[1, 0], fraction=0.045, pad=0.02).set_label("|error|")
    fig.colorbar(im_eraw, ax=axes[1, 1], fraction=0.045, pad=0.02).set_label("|error|")

    suptitle = fig.suptitle(title_fn(int(frame_indices[0])), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    def update(frame_pos):
        i          = int(frame_indices[frame_pos])
        p, g, r    = get_slice(i)
        eg         = np.abs(p - g)
        er         = np.abs(p - r)
        im_raw.set_data(r)
        im_gt.set_data(g)
        im_pred.set_data(p)
        im_egt.set_data(eg)
        im_eraw.set_data(er)
        suptitle.set_text(title_fn(i))
        return im_raw, im_gt, im_pred, im_egt, im_eraw, suptitle

    ani = animation.FuncAnimation(fig, update, frames=len(frame_indices), blit=False, interval=1000.0 / max(1, fps))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)
    ani.save(str(out_path), writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path
