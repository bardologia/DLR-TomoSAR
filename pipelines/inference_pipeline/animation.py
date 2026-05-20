from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from io                 import BytesIO
from pathlib            import Path
from typing             import Any

import matplotlib

matplotlib.use("Agg")
import numpy             as np
from PIL                 import Image
from tqdm                import tqdm

from pipelines.inference_pipeline.plots import Ploter


def _render_frame(args: tuple) -> tuple[int, bytes]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from io import BytesIO

    frame_order, g, p, vmin, vmax, emax_gt, extent, x_label, y_label, cmap, err_cmap, dpi, _origin, title = args

    eg = np.abs(p - g)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0))

    ims = [
        axes[0].imshow(g,  cmap=cmap,     vmin=vmin,    vmax=vmax,    extent=extent, aspect="auto", origin=_origin),
        axes[1].imshow(p,  cmap=cmap,     vmin=vmin,    vmax=vmax,    extent=extent, aspect="auto", origin=_origin),
        axes[2].imshow(eg, cmap=err_cmap, vmin=0.0,     vmax=emax_gt, extent=extent, aspect="auto", origin=_origin),
    ]

    for ax, label in zip(axes, ("GT (Gaussian)", "Prediction", "|Pred - GT|")):
        ax.set_title(label)
        ax.set_xlabel(x_label)

    axes[0].set_ylabel(y_label)

    int_label = "intensity"
    fig.colorbar(ims[0], ax=axes[0], fraction=0.045, pad=0.02).set_label(int_label)
    fig.colorbar(ims[1], ax=axes[1], fraction=0.045, pad=0.02).set_label(int_label)
    fig.colorbar(ims[2], ax=axes[2], fraction=0.045, pad=0.02).set_label("|error|")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    
    return frame_order, buf.read()


def make_walk_gif(
    pred_cube  : np.ndarray,
    gt_cube    : np.ndarray,
    axis       : str,
    out_path   : Path,
    *,
    x_axis     : np.ndarray,
    az_offset  : int,
    rg_offset  : int,
    fps        : int = 12,
    max_frames : int = 150,
    dpi        : int = 110,
    cmap       : str = "jet",
    err_cmap   : str = "magma",
    num_workers: int | None = None,
) -> Path:
    Ploter(fig_dpi=dpi, save_dpi=dpi)._apply_style()

    N_elev, az, rg = pred_cube.shape

    _sort_idx = None
    if axis in ("range", "azimuth"):
        _sort_idx = np.argsort(x_axis)
        x_axis    = x_axis[_sort_idx]

    if axis == "elevation":
        n_total = N_elev
        def get_slice(i):
            return pred_cube[i], gt_cube[i]
        
        extent           = [rg_offset, rg_offset + rg, az_offset + az, az_offset]
        x_label, y_label = "range index", "azimuth index"
        title_fn         = lambda i: f"elevation = {x_axis[i]:.2f} m  (idx {i}/{N_elev - 1})"
    
    elif axis == "range":
        n_total = rg
        def get_slice(i):
            p, g = pred_cube[:, :, i], gt_cube[:, :, i]
            if _sort_idx is not None:
                p, g = p[_sort_idx], g[_sort_idx]
        
            return p, g
        
        extent           = [az_offset, az_offset + az, float(x_axis[0]), float(x_axis[-1])]
        x_label, y_label = "azimuth index", "elevation [m]"
        title_fn         = lambda i: f"range = {i + rg_offset}"
    
    elif axis == "azimuth":
        n_total = az
        def get_slice(i):
            p, g = pred_cube[:, i, :], gt_cube[:, i, :]
            if _sort_idx is not None:
                p, g = p[_sort_idx], g[_sort_idx]
            
            return p, g
        
        extent           = [rg_offset, rg_offset + rg, float(x_axis[0]), float(x_axis[-1])]
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
    vmin, vmax  = Ploter._shared_clim(pred_sample, gt_sample)
    emax_gt     = float(np.percentile(np.abs(pred_sample - gt_sample), 99.0))
   
    if emax_gt <= 0.0: emax_gt = 1.0

    _origin = "lower" if axis in ("range", "azimuth") else "upper"

    tasks: list[tuple[Any, ...]] = []
    for frame_order, fi in enumerate(frame_indices):
        i    = int(fi)
        p, g = get_slice(i)
        tasks.append((
            frame_order,
            g.copy(), p.copy(),
            vmin, vmax,
            emax_gt,
            extent,
            x_label, y_label,
            cmap, err_cmap,
            dpi,
            _origin,
            title_fn(i),
        ))

    n_workers = num_workers if num_workers is not None else min(len(tasks), os.cpu_count() or 1)
    png_bytes: dict[int, bytes] = {}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_render_frame, t): t[0] for t in tasks}
        with tqdm(total=len(futures), desc=f"Rendering {axis} frames", unit="frame") as pbar:
            for fut in as_completed(futures):
                order, data = fut.result()
                png_bytes[order] = data
                pbar.update(1)

    frames: list[Image.Image] = [Image.open(BytesIO(png_bytes[k])).convert("P", dither=Image.Dither.NONE) for k in sorted(png_bytes)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / max(1, fps)))
    
    frames[0].save(
        fp            = str(out_path),
        format        = "GIF",
        save_all      = True,
        append_images = frames[1:],
        loop          = 0,
        duration      = duration_ms,
        optimize      = False,
    )
   
    return out_path
