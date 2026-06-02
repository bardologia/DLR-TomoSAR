from __future__ import annotations

import os
import warnings
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
from tools.logger                       import Logger


def _init_worker() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot
    import numpy
    import io


def _render_frame(args: tuple) -> tuple[int, bytes]:
    import matplotlib.pyplot as plt
    import numpy as np
    from io import BytesIO

    frame_order, n_frames, g, p, vmin, vmax, emax_gt, extent, x_label, y_label, cmap, err_cmap, dpi, _origin, title = args

    eg = np.abs(p - g)

    fig = plt.figure(figsize=(20, 6), constrained_layout=False)
    gs  = fig.add_gridspec(2, 3, height_ratios=[1, 0.03], hspace=0.35, wspace=0.35)
    axes = [fig.add_subplot(gs[0, k]) for k in range(3)]
    pbar_ax = fig.add_subplot(gs[1, :])

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

    progress = (frame_order + 1) / max(1, n_frames)
    pbar_ax.barh(0, progress,        height=1, color="steelblue", left=0.0)
    pbar_ax.barh(0, 1.0 - progress,  height=1, color="#333333",   left=progress)
    pbar_ax.set_xlim(0, 1)
    pbar_ax.set_axis_off()

    fig.suptitle(title, fontsize=13, y=0.98)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.08)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)

    return frame_order, buf.read()


class Animator:
    def __init__(
        self,
        logger      : Logger,
        *,
        cmap        : str = "jet",
        err_cmap    : str = "magma",
        dpi         : int = 110,
        fps         : int = 12,
        max_frames  : int = 150,
        num_workers : int | None = None,
    ) -> None:
        self.logger      = logger
        self.cmap        = cmap
        self.err_cmap    = err_cmap
        self.dpi         = dpi
        self.fps         = fps
        self.max_frames  = max_frames
        self.num_workers = num_workers

        self.logger.section("[Animator]")
        self.logger.subsection(f"colormap for GT and prediction : {cmap}")
        self.logger.subsection(f"colormap for error             : {err_cmap}")
        self.logger.subsection(f"figure DPI                     : {dpi}")
        self.logger.subsection(f"GIF frames per second          : {fps}")
        self.logger.subsection(f"max frames per GIF             : {max_frames}")
        self.logger.subsection(f"CPU workers for rendering      : {num_workers if num_workers is not None else 'auto'} \n")

    @staticmethod
    def _slice_elevation(pred_cube: np.ndarray, gt_cube: np.ndarray, i: int) -> tuple[np.ndarray, np.ndarray]:
        return pred_cube[i], gt_cube[i]

    @staticmethod
    def _slice_range(pred_cube: np.ndarray, gt_cube: np.ndarray, sort_idx: np.ndarray | None, i: int) -> tuple[np.ndarray, np.ndarray]:
        p, g = pred_cube[:, :, i], gt_cube[:, :, i]
        if sort_idx is not None:
            p, g = p[sort_idx], g[sort_idx]
        return p, g

    @staticmethod
    def _slice_azimuth(pred_cube: np.ndarray, gt_cube: np.ndarray, sort_idx: np.ndarray | None, i: int) -> tuple[np.ndarray, np.ndarray]:
        p, g = pred_cube[:, i, :], gt_cube[:, i, :]
        if sort_idx is not None:
            p, g = p[sort_idx], g[sort_idx]
        return p, g

    def _build_axis(self, axis: str, pred_cube: np.ndarray, gt_cube: np.ndarray, x_axis: np.ndarray, az_offset: int, rg_offset: int) -> dict:
        N_elev, az, rg = pred_cube.shape
        sort_idx = None

        if axis in ("range", "azimuth"):
            sort_idx = np.argsort(x_axis)
            x_axis   = x_axis[sort_idx]

        if axis == "elevation":
            return dict(
                n_total   = N_elev,
                get_slice = lambda i: self._slice_elevation(pred_cube, gt_cube, i),
                extent    = [rg_offset, rg_offset + rg, az_offset + az, az_offset],
                x_label   = "range index",
                y_label   = "azimuth index",
                title_fn  = lambda i: f"elevation = {x_axis[i]:.2f} m  (idx {i}/{N_elev - 1})",
                origin    = "upper",
            )

        if axis == "range":
            return dict(
                n_total   = rg,
                get_slice = lambda i: self._slice_range(pred_cube, gt_cube, sort_idx, i),
                extent    = [az_offset, az_offset + az, float(x_axis[0]), float(x_axis[-1])],
                x_label   = "azimuth index",
                y_label   = "elevation [m]",
                title_fn  = lambda i: f"range = {i + rg_offset}",
                origin    = "lower",
            )

        if axis == "azimuth":
            return dict(
                n_total   = az,
                get_slice = lambda i: self._slice_azimuth(pred_cube, gt_cube, sort_idx, i),
                extent    = [rg_offset, rg_offset + rg, float(x_axis[0]), float(x_axis[-1])],
                x_label   = "range index",
                y_label   = "elevation [m]",
                title_fn  = lambda i: f"azimuth = {i + az_offset}",
                origin    = "lower",
            )

        raise ValueError(f"axis must be elevation|range|azimuth, got {axis!r}")

    def _render(self, tasks: list[tuple[Any, ...]]) -> dict[int, bytes]:
        n_workers = self.num_workers if self.num_workers is not None else min(len(tasks), os.cpu_count() or 1)
        png_bytes: dict[int, bytes] = {}

        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as pool:
            futures = {pool.submit(_render_frame, t): t[0] for t in tasks}
            with tqdm(total=len(futures), desc="Rendering frames", unit="frame") as pbar:
                for fut in as_completed(futures):
                    order, data = fut.result()
                    png_bytes[order] = data
                    pbar.update(1)

        return png_bytes

    def walk_gif(
        self,
        pred_cube  : np.ndarray,
        gt_cube    : np.ndarray,
        axis       : str,
        out_path   : Path,
        *,
        x_axis     : np.ndarray,
        az_offset  : int,
        rg_offset  : int,
    ) -> Path:
        import matplotlib.pyplot as _plt
        _plt.rcParams.update(Ploter.SCIENTIFIC_RC)
        _plt.rcParams["figure.dpi"]  = self.dpi
        _plt.rcParams["savefig.dpi"] = self.dpi

        spec      = self._build_axis(axis, pred_cube, gt_cube, x_axis, az_offset, rg_offset)
        n_total   = spec["n_total"]
        get_slice = spec["get_slice"]

        frame_indices = (np.linspace(0, n_total - 1, self.max_frames).round().astype(int) if n_total > self.max_frames else np.arange(n_total))

        sample_idx  = frame_indices[:: max(1, len(frame_indices) // 16)]
        pred_sample = np.stack([get_slice(int(i))[0] for i in sample_idx])
        gt_sample   = np.stack([get_slice(int(i))[1] for i in sample_idx])
        vmin, vmax  = Ploter._shared_clim(pred_sample, gt_sample)
        emax_gt     = float(np.percentile(np.abs(pred_sample - gt_sample), 99.0))
       
        if emax_gt <= 0.0:
            emax_gt = 1.0

        tasks: list[tuple[Any, ...]] = []
        n_frames = len(frame_indices)
        for frame_order, fi in enumerate(frame_indices):
            i    = int(fi)
            p, g = get_slice(i)
            tasks.append((
                frame_order,
                n_frames,
                g.copy(), p.copy(),
                vmin, vmax,
                emax_gt,
                spec["extent"],
                spec["x_label"], spec["y_label"],
                self.cmap, self.err_cmap,
                self.dpi,
                spec["origin"],
                spec["title_fn"](i),
            ))

        png_bytes = self._render(tasks)
        frames    = [Image.open(BytesIO(png_bytes[k])).convert("P", dither=Image.Dither.NONE) for k in sorted(png_bytes)]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        duration_ms = int(round(1000.0 / max(1, self.fps)))

        frames[0].save(
            fp            = str(out_path),
            format        = "GIF",
            save_all      = True,
            append_images = frames[1:],
            loop          = 0,
            duration      = duration_ms,
            optimize      = False,
        )

        self.logger.subsection(f"GIF ({axis:<9}) : {out_path}")
       
        return out_path
