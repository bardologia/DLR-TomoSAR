from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses        import dataclass
from io                 import BytesIO
from pathlib            import Path
from typing             import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy             as np
from PIL                 import Image
from tqdm                import tqdm

from configuration.inference_config       import InferenceConfig
from pipelines.inference_pipeline.loader  import InferenceMetadata
from pipelines.inference_pipeline.metrics import Metrics, Result
from pipelines.inference_pipeline.plots   import PlotTools, Ploter
from pipelines.shared                     import ProfileNormalizer
from tools.logger                         import Logger


@dataclass
class FrameSpec:
    frame_order : int
    n_frames    : int
    gt          : np.ndarray
    pred        : np.ndarray
    vmin        : float
    vmax        : float
    emax_gt     : float
    extent      : list
    x_label     : str
    y_label     : str
    cmap        : str
    err_cmap    : str
    dpi         : int
    origin      : str
    title       : str
    full        : np.ndarray | None = None


class Animator:

    @staticmethod
    def _init_worker() -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot
        import numpy
        import io

    @staticmethod
    def _render_frame(spec: FrameSpec) -> tuple[int, bytes]:
        import matplotlib.pyplot as plt
        import numpy as np
        from io import BytesIO

        g, p = spec.gt, spec.pred
        eg   = np.abs(p - g)

        panels = []
        if spec.full is not None:
            panels.append((spec.full, "Full tomogram (raw)", spec.cmap, spec.vmin, spec.vmax))

        panels += [
            (g,  "GT (Gaussian)", spec.cmap,     spec.vmin, spec.vmax),
            (p,  "Prediction",    spec.cmap,     spec.vmin, spec.vmax),
            (eg, "|Pred - GT|",   spec.err_cmap, 0.0,       spec.emax_gt),
        ]

        n_col   = len(panels)
        fig     = plt.figure(figsize=(6.7 * n_col, 6), constrained_layout=False)
        gs      = fig.add_gridspec(2, n_col, height_ratios=[1, 0.03], hspace=0.35, wspace=0.35)
        axes    = [fig.add_subplot(gs[0, k]) for k in range(n_col)]
        pbar_ax = fig.add_subplot(gs[1, :])

        PlotTools._triple_panel(fig, axes, panels, spec.x_label, "intensity", spec.extent, origin=spec.origin)
        axes[0].set_ylabel(spec.y_label)

        progress = (spec.frame_order + 1) / max(1, spec.n_frames)
        pbar_ax.barh(0, progress,        height=1, color="steelblue", left=0.0)
        pbar_ax.barh(0, 1.0 - progress,  height=1, color="#333333",   left=progress)
        pbar_ax.set_xlim(0, 1)
        pbar_ax.set_axis_off()

        fig.suptitle(spec.title, fontsize=13, y=0.98)
        fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.08)

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=spec.dpi)
        plt.close(fig)
        buf.seek(0)

        return spec.frame_order, buf.read()

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
    def _slice_elevation(cubes: tuple, i: int) -> tuple:
        return tuple(cube[i] if cube is not None else None for cube in cubes)

    @staticmethod
    def _slice_range(cubes: tuple, sort_idx: np.ndarray | None, i: int) -> tuple:
        slices = tuple(cube[:, :, i] if cube is not None else None for cube in cubes)
        if sort_idx is not None:
            slices = tuple(s[sort_idx] if s is not None else None for s in slices)
        return slices

    @staticmethod
    def _slice_azimuth(cubes: tuple, sort_idx: np.ndarray | None, i: int) -> tuple:
        slices = tuple(cube[:, i, :] if cube is not None else None for cube in cubes)
        if sort_idx is not None:
            slices = tuple(s[sort_idx] if s is not None else None for s in slices)
        return slices

    def _build_axis(self, axis: str, cubes: tuple, x_axis: np.ndarray, az_offset: int, rg_offset: int) -> dict:
        N_elev, az, rg = cubes[0].shape
        sort_idx = None

        if axis in ("range", "azimuth"):
            sort_idx = np.argsort(x_axis)
            x_axis   = x_axis[sort_idx]

        if axis == "elevation":
            return dict(
                n_total   = N_elev,
                get_slice = lambda i: self._slice_elevation(cubes, i),
                extent    = [rg_offset, rg_offset + rg, az_offset + az, az_offset],
                x_label   = "range index",
                y_label   = "azimuth index",
                title_fn  = lambda i: f"elevation = {x_axis[i]:.2f} m  (idx {i}/{N_elev - 1})",
                origin    = "upper",
            )

        if axis == "range":
            return dict(
                n_total   = rg,
                get_slice = lambda i: self._slice_range(cubes, sort_idx, i),
                extent    = [az_offset, az_offset + az, float(x_axis[0]), float(x_axis[-1])],
                x_label   = "azimuth index",
                y_label   = "elevation [m]",
                title_fn  = lambda i: f"range = {i + rg_offset}",
                origin    = "lower",
            )

        if axis == "azimuth":
            return dict(
                n_total   = az,
                get_slice = lambda i: self._slice_azimuth(cubes, sort_idx, i),
                extent    = [rg_offset, rg_offset + rg, float(x_axis[0]), float(x_axis[-1])],
                x_label   = "range index",
                y_label   = "elevation [m]",
                title_fn  = lambda i: f"azimuth = {i + az_offset}",
                origin    = "lower",
            )

        raise ValueError(f"axis must be elevation|range|azimuth, got {axis!r}")

    def _render(self, tasks: list[FrameSpec]) -> dict[int, bytes]:
        n_workers = self.num_workers if self.num_workers is not None else min(len(tasks), os.cpu_count() or 1)
        png_bytes: dict[int, bytes] = {}

        with ProcessPoolExecutor(max_workers=n_workers, initializer=Animator._init_worker) as pool:
            futures = {pool.submit(Animator._render_frame, t): t.frame_order for t in tasks}
            with tqdm(total=len(futures), desc="Rendering frames", unit="frame") as pbar:
                for fut in as_completed(futures):
                    order, data = fut.result()
                    png_bytes[order] = data
                    pbar.update(1)

        return png_bytes

    def walk_gif(
        self,
        pred_cube    : np.ndarray,
        gt_cube      : np.ndarray,
        axis         : str,
        out_path     : Path,
        *,
        x_axis       : np.ndarray,
        az_offset    : int,
        rg_offset    : int,
        full_cube    : np.ndarray | None = None,
    ) -> Path:
        plt.rcParams.update(Ploter.SCIENTIFIC_RC)
        plt.rcParams["figure.dpi"]  = self.dpi
        plt.rcParams["savefig.dpi"] = self.dpi

        cubes     = (pred_cube, gt_cube) if full_cube is None else (pred_cube, gt_cube, full_cube)
        spec      = self._build_axis(axis, cubes, x_axis, az_offset, rg_offset)
        n_total   = spec["n_total"]
        get_slice = spec["get_slice"]

        frame_indices = (np.linspace(0, n_total - 1, self.max_frames).round().astype(int) if n_total > self.max_frames else np.arange(n_total))

        sample_idx  = frame_indices[:: max(1, len(frame_indices) // 16)]
        samples     = [get_slice(int(i)) for i in sample_idx]
        pred_sample = np.stack([s[0] for s in samples])
        gt_sample   = np.stack([s[1] for s in samples])
        vmin, vmax  = Ploter._shared_clim(pred_sample, gt_sample)
        emax_gt     = float(np.percentile(np.abs(pred_sample - gt_sample), 99.0))

        if emax_gt <= 0.0:
            emax_gt = 1.0

        tasks: list[FrameSpec] = []
        n_frames = len(frame_indices)
        for frame_order, fi in enumerate(frame_indices):
            i     = int(fi)
            slc   = get_slice(i)
            p, g  = slc[0], slc[1]
            f     = slc[2] if full_cube is not None else None
            tasks.append(FrameSpec(
                frame_order = frame_order,
                n_frames    = n_frames,
                gt          = g.copy(),
                pred        = p.copy(),
                full        = f.copy() if f is not None else None,
                vmin        = vmin,
                vmax        = vmax,
                emax_gt     = emax_gt,
                extent      = spec["extent"],
                x_label     = spec["x_label"],
                y_label     = spec["y_label"],
                cmap        = self.cmap,
                err_cmap    = self.err_cmap,
                dpi         = self.dpi,
                origin      = spec["origin"],
                title       = spec["title_fn"](i),
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

        slice_plotter = self.plotter.slice
        param_plotter = self.plotter.param
        slot_plotter  = self.plotter.slot
        track_plotter = self.plotter.track
        meta          = self.meta
        logger        = self.logger
        cfg           = self.cfg

        slice_range_idx = indices["slice_range_idx"]
        slice_az_idx    = indices["slice_az_idx"]
        slice_elev_idx  = indices["slice_elev_idx"]
        all_range_idx   = indices["all_range_idx"]
        all_az_idx      = indices["all_az_idx"]
        all_elev_idx    = indices["all_elev_idx"]

        _N_elev, _az, _rg = result.pred_curves.shape
        figure_paths: Dict[str, List[Path]] = {}

        if run.track_baselines is not None:
            figure_paths["track_geometry"] = [track_plotter.plot_track_geometry(
                baselines = run.track_baselines,
                out_path  = meta.figures_dir / "tracks" / "track_geometry.png",
            )]
            logger.subsection(f"Track geometry : {meta.figures_dir / 'tracks'}")

        if run.track_profiles is not None:
            figure_paths["track_profiles"] = track_plotter.plot_track_profiles(
                profiles      = run.track_profiles,
                out_dir       = meta.figures_dir / "tracks",
                split_azimuth = (run.split_region.azimuth_start, run.split_region.azimuth_end),
            )
            logger.subsection(f"Track profiles : {len(figure_paths['track_profiles'])} figures in {meta.figures_dir / 'tracks'}")

            figure_paths["track_flight_3d"] = [track_plotter.plot_track_flight_3d(
                profiles = run.track_profiles,
                out_path = meta.figures_dir / "tracks" / "flight_tracks_3d.png",
            )]
            logger.subsection(f"Flight tracks 3D : {meta.figures_dir / 'tracks' / 'flight_tracks_3d.png'}")

        if run.complex_inputs is not None and run.n_secondaries > 0:
            figure_paths["input_channels"] = slice_plotter.plot_input_channels(
                complex_inputs = run.complex_inputs,
                n_secondaries  = run.n_secondaries,
                labels         = run.secondary_labels,
                out_dir        = meta.figures_dir / "input_channels",
                az_offset      = result.azimuth_offset,
                rg_offset      = result.range_offset,
                primary_label  = run.track_baselines.reference if run.track_baselines is not None else "primary",
            )
            logger.subsection(f"Input channels : {len(figure_paths['input_channels'])} figures in {meta.figures_dir / 'input_channels'}")

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

        reduced_curves_for_plot = result.reduced.reduced_curves if result.reduced is not None else None

        for tag, pixels in (("best", selected["best"]), ("worst", selected["worst"]), ("random", selected["random"])):
            figure_paths[f"profiles_{tag}"] = slice_plotter.plot_profiles(
                pred_curves    = result.pred_curves,
                gt_curves      = result.gt_curves,
                params_pred    = result.params_pred,
                x_axis         = x_axis_np,
                pixels         = pixels,
                tag            = tag,
                out_dir        = meta.figures_dir / "profiles",
                n_gaussians    = run.n_gaussians,
                pixel_metrics  = pixel_metrics_for_plot,
                az_offset      = result.azimuth_offset,
                rg_offset      = result.range_offset,
                reduced_curves = reduced_curves_for_plot,
                full_curves    = run.full_curves,
            )

            logger.subsection(f"Profiles ({tag:<6}) : {len(figure_paths[f'profiles_{tag}'])} figures in {meta.figures_dir / 'profiles'}")

        pixel_map_specs = [
            ("pixel_mse_map",  "mse",        result.pixel_mse,                             "Per-pixel curve MSE (denorm)",          "MSE",          {"cmap": cfg.cmap_error, "log": True}),
            ("pixel_r2_map",   "r2",         result.pixel_r2,                              "Per-pixel R² (denorm)",                 "R²",           {"cmap": "RdYlGn", "q_low": 2.0, "q_high": 98.0}),
            ("pixel_peak_map", "peak_error", result.pixel_peak_err_idx.astype(np.float32), "Peak-location absolute error (denorm)", "|Δ peak idx|", {"cmap": cfg.cmap_error}),
        ]

        for key, fname, data, title, label, extra in pixel_map_specs:
            figure_paths[key] = [slice_plotter.plot_pixel_metric_map(
                metric_map = data,
                title      = title,
                label      = label,
                out_path   = meta.figures_dir / "pixel_maps" / f"{fname}.png",
                az_offset  = result.azimuth_offset,
                rg_offset  = result.range_offset,
                **extra,
            )]

        logger.subsection(f"Pixel maps : mse, r2, peak written to {meta.figures_dir / 'pixel_maps'}")

        histogram_arrays = {
            "pixel_mse"    : result.pixel_mse,
            "pixel_r2"     : result.pixel_r2,
            "pixel_cosine" : result.pixel_cosine,
        }

        figure_paths["metric_histograms"] = slice_plotter.plot_metric_histograms(
            histogram_arrays,
            meta.figures_dir / "histograms",
        )

        figure_paths["param_maps"] = param_plotter.plot_param_maps(
            params_pred = result.params_pred[: run.n_gaussians * 3],
            params_gt   = (result.params_gt[: run.n_gaussians * 3] if result.params_gt is not None else None),
            n_gaussians = run.n_gaussians,
            out_dir     = meta.figures_dir / "param_maps",
            az_offset   = result.azimuth_offset,
            rg_offset   = result.range_offset,
        )

        figure_paths["param_distributions"] = param_plotter.plot_param_distributions(
            params_pred = result.params_pred[: run.n_gaussians * 3],
            params_gt   = (result.params_gt[: run.n_gaussians * 3] if result.params_gt is not None else None),
            n_gaussians = run.n_gaussians,
            out_dir     = meta.figures_dir / "param_distributions",
        )

        figure_paths["param_scatter"] = param_plotter.plot_param_scatter(
            params_pred = result.params_pred[: run.n_gaussians * 3],
            params_gt   = result.params_gt  [: run.n_gaussians * 3],
            n_gaussians = run.n_gaussians,
            out_dir     = meta.figures_dir / "param_scatter",
        )

        figure_paths["param_error_maps"] = param_plotter.plot_param_error_maps(
            params_pred = result.params_pred[: run.n_gaussians * 3],
            params_gt   = result.params_gt  [: run.n_gaussians * 3],
            n_gaussians = run.n_gaussians,
            out_dir     = meta.figures_dir / "param_error_maps",
            az_offset   = result.azimuth_offset,
            rg_offset   = result.range_offset,
        )

        figure_paths["slot_mu_distributions"] = slot_plotter.plot_slot_mu_distributions(
            global_metrics = global_metrics,
            n_gaussians    = run.n_gaussians,
            out_dir        = meta.figures_dir / "slots",
        )

        figure_paths["placeholder_detection"] = slot_plotter.plot_placeholder_detection(
            global_metrics = global_metrics,
            n_gaussians    = run.n_gaussians,
            out_dir        = meta.figures_dir / "slots",
        )

        figure_paths["slot_ordering_summary"] = slot_plotter.plot_slot_ordering_summary(
            global_metrics = global_metrics,
            n_gaussians    = run.n_gaussians,
            out_dir        = meta.figures_dir / "slots",
        )

        figure_paths["active_count_map"] = slot_plotter.plot_active_count_map(
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
            for i in indices_arr:
                figure_paths[group] += slice_plotter.plot_tomogram_slice(
                    pred_cube    = result.pred_curves,
                    gt_cube      = result.gt_curves,
                    axis         = axis,
                    index        = int(i),
                    x_axis       = x_axis_np,
                    out_dir      = meta.figures_dir / "slices",
                    stem         = stem_fn(i),
                    az_offset    = result.azimuth_offset,
                    rg_offset    = result.range_offset,
                    ssim_value   = global_metrics[f"{metric_key}_{int(i)}"],
                    full_cube    = run.full_curves,
                )

        for i in slice_elev_idx:
            figure_paths["slices_elev"] += slice_plotter.plot_elevation_intensity_slice(
                pred_cube    = result.pred_curves,
                gt_cube      = result.gt_curves,
                elev_idx     = int(i),
                x_axis       = x_axis_np,
                out_dir      = meta.figures_dir / "slices",
                stem         = f"elev_idx_{int(i)}",
                az_offset    = result.azimuth_offset,
                rg_offset    = result.range_offset,
                ssim_value   = global_metrics[f"ssim_gt_elev_{int(i)}"],
                full_cube    = run.full_curves,
            )

        logger.subsection(f"Slices written : range={cfg.n_range_slices} azimuth={cfg.n_azimuth_slices} elev={cfg.n_elevation_slices} (gt, pred, error each)")

        for axis, n_slices, indices_arr, offset in (
            ("range",   _rg,     all_range_idx, result.range_offset),
            ("azimuth", _az,     all_az_idx,    result.azimuth_offset),
            ("elev",    _N_elev, all_elev_idx,  0),
        ):
            figure_paths[f"ssim_{axis}"] = [slice_plotter.plot_ssim_curves(
                global_metrics = global_metrics,
                axis           = axis,
                out_path       = meta.figures_dir / "ssim" / f"{axis}.png",
                n_slices       = n_slices,
                slice_indices  = indices_arr,
                ax_offset      = offset,
            )]

        logger.subsection(f"SSIM plots : range, azimuth, elev written to {meta.figures_dir / 'ssim'}\n")

        figure_paths["elev_metric_curves"] = slice_plotter.plot_elev_metric_curves(
            global_metrics = global_metrics,
            out_dir        = meta.figures_dir / "elev_metrics",
            n_elev         = _N_elev,
            x_axis         = x_axis_np,
        )

        logger.subsection(f"Elev metric curves (MAE, RMSE, R², CE) written to {meta.figures_dir / 'elev_metrics'}\n")

        if result.reduced is not None:
            self._compose_reduced(result, run, global_metrics, x_axis_np, indices, figure_paths)

        return figure_paths

    def _compose_reduced(
        self,
        result         : Result,
        run,
        global_metrics : dict,
        x_axis_np      : np.ndarray,
        indices        : dict,
        figure_paths   : Dict[str, List[Path]],
    ) -> None:

        slice_plotter = self.plotter.slice
        meta          = self.meta
        logger        = self.logger
        reduced       = result.reduced

        gt_n   = reduced.gt_norm
        red_n  = reduced.reduced_norm
        full_n = ProfileNormalizer.unit_area(run.full_curves) if run.full_curves is not None else None

        slice_range_idx = indices["slice_range_idx"]
        slice_az_idx    = indices["slice_az_idx"]
        slice_elev_idx  = indices["slice_elev_idx"]
        all_range_idx   = indices["all_range_idx"]
        all_az_idx      = indices["all_az_idx"]
        all_elev_idx    = indices["all_elev_idx"]

        _N_elev, _az, _rg = red_n.shape

        reduced_dir = meta.figures_dir / "reduced"

        figure_paths["improvement_map"] = [slice_plotter.plot_pixel_metric_map(
            metric_map = reduced.improvement,
            title      = "NN improvement over classical baseline (per-pixel ΔMSE, unit-area profiles)",
            label      = "MSE(reduced) − MSE(pred)",
            out_path   = reduced_dir / "improvement_map.png",
            az_offset  = result.azimuth_offset,
            rg_offset  = result.range_offset,
            cmap       = "RdBu_r",
            q_low      = 2.0,
            q_high     = 98.0,
        )]

        figure_paths["reduced_pixel_mse_map"] = [slice_plotter.plot_pixel_metric_map(
            metric_map = reduced.err_reduced,
            title      = "Per-pixel reduced-vs-GT MSE (unit-area profiles, log scale)",
            label      = "MSE",
            out_path   = reduced_dir / "reduced_mse_map.png",
            az_offset  = result.azimuth_offset,
            rg_offset  = result.range_offset,
            cmap       = self.cfg.cmap_error,
            log        = True,
        )]

        figure_paths["slices_range_reduced"]   = []
        figure_paths["slices_azimuth_reduced"] = []
        figure_paths["slices_elev_reduced"]    = []

        for axis, indices_arr, stem_fn, group in (
            ("range",   slice_range_idx, lambda i: f"reduced_range_{int(i) + result.range_offset}",     "slices_range_reduced"),
            ("azimuth", slice_az_idx,    lambda i: f"reduced_azimuth_{int(i) + result.azimuth_offset}", "slices_azimuth_reduced"),
        ):
            for i in indices_arr:
                figure_paths[group] += slice_plotter.plot_tomogram_slice(
                    pred_cube  = red_n,
                    gt_cube    = gt_n,
                    axis       = axis,
                    index      = int(i),
                    x_axis     = x_axis_np,
                    out_dir    = reduced_dir / "slices",
                    stem       = stem_fn(i),
                    az_offset  = result.azimuth_offset,
                    rg_offset  = result.range_offset,
                    ssim_value = global_metrics[f"ssim_red_{axis}_{int(i)}"],
                    ref_title  = "GT (unit-area)",
                    pred_title = "Reduced (Capon, unit-area)",
                    err_title  = "|Reduced − GT|",
                    full_cube  = full_n,
                    full_title = "Full tomogram (unit-area)",
                )

        for i in slice_elev_idx:
            figure_paths["slices_elev_reduced"] += slice_plotter.plot_elevation_intensity_slice(
                pred_cube  = red_n,
                gt_cube    = gt_n,
                elev_idx   = int(i),
                x_axis     = x_axis_np,
                out_dir    = reduced_dir / "slices",
                stem       = f"reduced_elev_idx_{int(i)}",
                az_offset  = result.azimuth_offset,
                rg_offset  = result.range_offset,
                ssim_value = global_metrics[f"ssim_red_elev_{int(i)}"],
                ref_title  = "GT (unit-area)",
                pred_title = "Reduced (Capon, unit-area)",
                err_title  = "|Reduced − GT|",
                full_cube  = full_n,
                full_title = "Full tomogram (unit-area)",
            )

        for axis, n_slices, indices_arr, offset in (
            ("range",   _rg,     all_range_idx, result.range_offset),
            ("azimuth", _az,     all_az_idx,    result.azimuth_offset),
            ("elev",    _N_elev, all_elev_idx,  0),
        ):
            figure_paths[f"ssim_{axis}_reduced"] = [slice_plotter.plot_ssim_curves(
                global_metrics = global_metrics,
                axis           = axis,
                out_path       = reduced_dir / "ssim" / f"{axis}.png",
                n_slices       = n_slices,
                slice_indices  = indices_arr,
                ax_offset      = offset,
                prefix         = "red",
                series_label   = "reduced × GT (unit-area)",
            )]

        figure_paths["elev_metric_curves_reduced"] = slice_plotter.plot_elev_metric_curves(
            global_metrics = global_metrics,
            out_dir        = reduced_dir / "elev_metrics",
            n_elev         = _N_elev,
            x_axis         = x_axis_np,
            suffix         = "red",
            series_label   = "reduced × GT (unit-area)",
        )

        logger.subsection(f"Reduced baseline figures written to {reduced_dir}")

    def animate(self, result: Result, run, x_axis_np: np.ndarray) -> Dict[str, Path]:
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
                pred_cube    = result.pred_curves,
                gt_cube      = result.gt_curves,
                axis         = axis,
                out_path     = meta.animations_dir / f"walk_{axis}.gif",
                x_axis       = x_axis_np,
                az_offset    = result.azimuth_offset,
                rg_offset    = result.range_offset,
                full_cube    = run.full_curves,
            )

        logger.subsection("")

        return gif_paths
