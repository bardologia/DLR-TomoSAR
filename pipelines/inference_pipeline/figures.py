from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from io                 import BytesIO
from pathlib            import Path
from typing             import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import numpy as np
from PIL     import Image
from tqdm    import tqdm

from configuration.inference_config       import InferenceConfig
from pipelines.inference_pipeline.loader  import InferenceMetadata
from pipelines.inference_pipeline.metrics import Metrics, Result
from pipelines.inference_pipeline.plots   import PlotTools, Ploter
from tools.logger                         import Logger


class Animator:

    @staticmethod
    def _init_worker() -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot
        import numpy
        import io

    @staticmethod
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

        panels = [
            (g,  "GT (Gaussian)", cmap,     vmin, vmax),
            (p,  "Prediction",    cmap,     vmin, vmax),
            (eg, "|Pred - GT|",   err_cmap, 0.0,  emax_gt),
        ]

        PlotTools._triple_panel(fig, axes, panels, x_label, "intensity", extent, origin=_origin)

        axes[0].set_ylabel(y_label)

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

        with ProcessPoolExecutor(max_workers=n_workers, initializer=Animator._init_worker) as pool:
            futures = {pool.submit(Animator._render_frame, t): t[0] for t in tasks}
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


class FigureComposer:
    def __init__(
        self,
        plotter : Ploter,
        meta    : InferenceMetadata,
        logger  : Logger,
        cfg     : InferenceConfig,
    ) -> None:

        self.plotter = plotter
        self.meta    = meta
        self.logger  = logger
        self.cfg     = cfg

    def compose(
        self,
        result         : Result,
        run,
        global_metrics : dict,
        x_axis_np      : np.ndarray,
        indices        : dict,
    ) -> Dict[str, List[Path]]:

        plotter = self.plotter
        meta    = self.meta
        logger  = self.logger
        cfg     = self.cfg

        slice_range_idx = indices["slice_range_idx"]
        slice_az_idx    = indices["slice_az_idx"]
        slice_elev_idx  = indices["slice_elev_idx"]
        all_range_idx   = indices["all_range_idx"]
        all_az_idx      = indices["all_az_idx"]
        all_elev_idx    = indices["all_elev_idx"]

        _N_elev, _az, _rg = result.pred_curves.shape
        figure_paths: Dict[str, List[Path]] = {}

        pixel_metrics_for_plot = {
            "mse" : result.pixel_mse,
            "mae" : result.pixel_mae,
            "r2"  : result.pixel_r2,
            "cos" : result.pixel_cosine,
        }

        selected = Metrics.select_pixels(
            result.pixel_mse,
            n_best   = cfg.n_best_profiles,
            n_worst  = cfg.n_worst_profiles,
            n_random = cfg.n_random_profiles,
            seed     = cfg.profile_seed,
        )

        for tag, pixels in (("best", selected["best"]), ("worst", selected["worst"]), ("random", selected["random"])):
            figure_paths[f"profiles_{tag}"] = plotter.plot_profiles(
                pred_curves   = result.pred_curves,
                gt_curves     = result.gt_curves,
                params_pred   = result.params_pred,
                x_axis        = x_axis_np,
                pixels        = pixels,
                tag           = tag,
                out_dir       = meta.figures_dir / "profiles",
                n_gaussians   = run.n_gaussians,
                pixel_metrics = pixel_metrics_for_plot,
                az_offset     = result.azimuth_offset,
                rg_offset     = result.range_offset,
            )

            logger.subsection(f"Profiles ({tag:<6}) : {len(figure_paths[f'profiles_{tag}'])} figures in {meta.figures_dir / 'profiles'}")

        for key, fname, data, title, label, extra in (
            ("pixel_mse_map",  "mse",        result.pixel_mse,                             "Per-pixel curve MSE (denorm)",          "MSE",          {"cmap": cfg.cmap_error, "log": True}),
            ("pixel_r2_map",   "r2",         result.pixel_r2,                              "Per-pixel R² (denorm)",                 "R²",           {"cmap": "RdYlGn", "q_low": 2.0, "q_high": 98.0}),
            ("pixel_peak_map", "peak_error", result.pixel_peak_err_idx.astype(np.float32), "Peak-location absolute error (denorm)", "|Δ peak idx|", {"cmap": cfg.cmap_error}),
        ):
            figure_paths[key] = [plotter.plot_pixel_metric_map(
                metric_map = data,
                title      = title,
                label      = label,
                out_path   = meta.figures_dir / "pixel_maps" / f"{fname}.png",
                az_offset  = result.azimuth_offset,
                rg_offset  = result.range_offset,
                **extra,
            )]

        logger.subsection(f"Pixel maps : mse, r2, peak written to {meta.figures_dir / 'pixel_maps'}")

        figure_paths["metric_histograms"] = plotter.plot_metric_histograms(
            {
                "pixel_mse"    : result.pixel_mse,
                "pixel_r2"     : result.pixel_r2,
                "pixel_cosine" : result.pixel_cosine,
            },
            meta.figures_dir / "histograms",
        )

        figure_paths["param_maps"] = plotter.plot_param_maps(
            params_pred = result.params_pred[: run.n_gaussians * 3],
            params_gt   = (result.params_gt[: run.n_gaussians * 3] if result.params_gt is not None else None),
            n_gaussians = run.n_gaussians,
            out_dir     = meta.figures_dir / "param_maps",
            az_offset   = result.azimuth_offset,
            rg_offset   = result.range_offset,
        )

        figure_paths["param_distributions"] = plotter.plot_param_distributions(
            params_pred = result.params_pred[: run.n_gaussians * 3],
            params_gt   = (result.params_gt[: run.n_gaussians * 3] if result.params_gt is not None else None),
            n_gaussians = run.n_gaussians,
            out_dir     = meta.figures_dir / "param_distributions",
        )

        figure_paths["param_scatter"] = plotter.plot_param_scatter(
            params_pred = result.params_pred[: run.n_gaussians * 3],
            params_gt   = result.params_gt  [: run.n_gaussians * 3],
            n_gaussians = run.n_gaussians,
            out_dir     = meta.figures_dir / "param_scatter",
        )

        figure_paths["param_error_maps"] = plotter.plot_param_error_maps(
            params_pred = result.params_pred[: run.n_gaussians * 3],
            params_gt   = result.params_gt  [: run.n_gaussians * 3],
            n_gaussians = run.n_gaussians,
            out_dir     = meta.figures_dir / "param_error_maps",
            az_offset   = result.azimuth_offset,
            rg_offset   = result.range_offset,
        )

        figure_paths["slot_mu_distributions"] = plotter.plot_slot_mu_distributions(
            global_metrics = global_metrics,
            n_gaussians    = run.n_gaussians,
            out_dir        = meta.figures_dir / "slots",
        )

        figure_paths["placeholder_detection"] = plotter.plot_placeholder_detection(
            global_metrics = global_metrics,
            n_gaussians    = run.n_gaussians,
            out_dir        = meta.figures_dir / "slots",
        )

        figure_paths["slot_ordering_summary"] = plotter.plot_slot_ordering_summary(
            global_metrics = global_metrics,
            n_gaussians    = run.n_gaussians,
            out_dir        = meta.figures_dir / "slots",
        )

        figure_paths["active_count_map"] = plotter.plot_active_count_map(
            params_pred = result.params_pred[: run.n_gaussians * 3],
            params_gt   = result.params_gt  [: run.n_gaussians * 3],
            n_gaussians = run.n_gaussians,
            out_dir     = meta.figures_dir / "slots",
            az_offset   = result.azimuth_offset,
            rg_offset   = result.range_offset,
        )

        logger.subsection(f"Param plots : maps, distributions, scatter, error maps, slots written to {meta.figures_dir}")

        figure_paths["slices_range"]   = []
        figure_paths["slices_azimuth"] = []
        figure_paths["slices_elev"]    = []

        for axis, indices_arr, stem_fn, metric_key, group in (
            ("range",   slice_range_idx, lambda i: f"range_{int(i) + result.range_offset}",     "ssim_gt_range",   "slices_range"),
            ("azimuth", slice_az_idx,    lambda i: f"azimuth_{int(i) + result.azimuth_offset}", "ssim_gt_azimuth", "slices_azimuth"),
        ):
            for s_idx, i in enumerate(indices_arr):
                figure_paths[group] += plotter.plot_tomogram_slice(
                    pred_cube  = result.pred_curves,
                    gt_cube    = result.gt_curves,
                    axis       = axis,
                    index      = int(i),
                    x_axis     = x_axis_np,
                    out_dir    = meta.figures_dir / "slices",
                    stem       = stem_fn(i),
                    az_offset  = result.azimuth_offset,
                    rg_offset  = result.range_offset,
                    ssim_value = global_metrics.get(f"{metric_key}_{s_idx}"),
                )

        for s_idx, i in enumerate(slice_elev_idx):
            figure_paths["slices_elev"] += plotter.plot_elevation_intensity_slice(
                pred_cube  = result.pred_curves,
                gt_cube    = result.gt_curves,
                elev_idx   = int(i),
                x_axis     = x_axis_np,
                out_dir    = meta.figures_dir / "slices",
                stem       = f"elev_idx_{int(i)}",
                az_offset  = result.azimuth_offset,
                rg_offset  = result.range_offset,
                ssim_value = global_metrics.get(f"ssim_gt_elev_{s_idx}"),
            )

        logger.subsection(f"Slices written : range={cfg.n_range_slices} azimuth={cfg.n_azimuth_slices} elev={cfg.n_elevation_slices} (gt, pred, error each)")

        for axis, n_slices, indices_arr, offset in (
            ("range",   _rg,     all_range_idx, result.range_offset),
            ("azimuth", _az,     all_az_idx,    result.azimuth_offset),
            ("elev",    _N_elev, all_elev_idx,  0),
        ):
            figure_paths[f"ssim_{axis}"] = [plotter.plot_ssim_curves(
                global_metrics = global_metrics,
                axis           = axis,
                out_path       = meta.figures_dir / "ssim" / f"{axis}.png",
                n_slices       = n_slices,
                slice_indices  = indices_arr,
                ax_offset      = offset,
            )]

        logger.subsection(f"SSIM plots : range, azimuth, elev written to {meta.figures_dir / 'ssim'}\n")

        figure_paths["elev_metric_curves"] = plotter.plot_elev_metric_curves(
            global_metrics = global_metrics,
            out_dir        = meta.figures_dir / "elev_metrics",
            n_elev         = _N_elev,
            x_axis         = x_axis_np,
        )

        logger.subsection(f"Elev metric curves (MAE, RMSE, R², CE) written to {meta.figures_dir / 'elev_metrics'}\n")

        return figure_paths

    def animate(self, result: Result, x_axis_np: np.ndarray) -> Dict[str, Path]:
        cfg    = self.cfg
        meta   = self.meta
        logger = self.logger

        animator = Animator(
            logger      = logger,
            cmap        = cfg.cmap_intensity,
            err_cmap    = cfg.cmap_error,
            dpi         = cfg.gif_dpi,
            fps         = cfg.gif_fps,
            max_frames  = cfg.gif_max_frames,
            num_workers = cfg.gif_workers,
        )

        gif_paths: Dict[str, Path] = {}
        for axis in cfg.gif_axes:
            logger.subsection(f"Generating walk GIF along {axis} axis")
            gif_paths[f"walk_{axis}"] = animator.walk_gif(
                pred_cube = result.pred_curves,
                gt_cube   = result.gt_curves,
                axis      = axis,
                out_path  = meta.animations_dir / f"walk_{axis}.gif",
                x_axis    = x_axis_np,
                az_offset = result.azimuth_offset,
                rg_offset = result.range_offset,
            )

        logger.subsection("")

        return gif_paths
