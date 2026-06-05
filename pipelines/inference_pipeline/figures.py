from __future__ import annotations

from pathlib import Path
from typing  import Dict, List

import numpy as np

from configuration.inference_config         import InferenceConfig
from pipelines.inference_pipeline.animation import Animator
from pipelines.inference_pipeline.metadata  import InferenceMetadata
from pipelines.inference_pipeline.metrics   import Metrics
from pipelines.inference_pipeline.plots     import Ploter
from pipelines.inference_pipeline.types     import Result
from tools.logger                           import Logger


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
