from __future__ import annotations

from pathlib  import Path
from typing   import Dict

import numpy as np

from pipelines.inference_pipeline.animation import Animator
from configuration.inference_config         import InferenceConfig
from pipelines.inference_pipeline.loader    import RunLoader
from pipelines.inference_pipeline.metadata  import InferenceMetadata
from pipelines.inference_pipeline.metrics   import Metrics
from pipelines.inference_pipeline.plots     import Ploter
from pipelines.inference_pipeline.predictor import Predictor
from pipelines.inference_pipeline.report    import Report, write_metrics_json
from tools.logger                           import Logger


class InferencePipeline:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config

    def _plot_figures(
        self,
        plotter         : Ploter,
        result,
        run,
        meta            : InferenceMetadata,
        global_metrics  : dict,
        x_axis_np       : np.ndarray,
        cfg             : InferenceConfig,
        logger          : Logger,
        slice_range_idx : np.ndarray,
        slice_az_idx    : np.ndarray,
        slice_elev_idx  : np.ndarray,
        all_range_idx   : np.ndarray,
        all_az_idx      : np.ndarray,
        all_elev_idx    : np.ndarray,
    ) -> Dict[str, Path]:

        _N_elev, _az, _rg = result.pred_curves.shape
        figure_paths: Dict[str, Path] = {}

        pixel_metrics_for_plot = {
            "mse" : result.pixel_mse,    # (denorm)
            "mae" : result.pixel_mae,    # (denorm)
            "r2"  : result.pixel_r2,     # (denorm)
            "cos" : result.pixel_cosine, # (denorm)
        }

        selected = Metrics.select_pixels(
            result.pixel_mse,
            n_best   = cfg.n_best_profiles,
            n_worst  = cfg.n_worst_profiles,
            n_random = cfg.n_random_profiles,
            seed     = cfg.profile_seed,
        )

        profile_titles = {
            "best"   : "Best-fit profile reconstructions (lowest per-pixel MSE)",
            "worst"  : "Worst-fit profile reconstructions (highest per-pixel MSE)",
            "random" : "Random profile reconstructions",
        }

        for tag, pixels in (("best", selected["best"]), ("worst", selected["worst"]), ("random", selected["random"])):
            figure_paths[f"profiles_{tag}"] = plotter.plot_profile_panel(
                pred_curves   = result.pred_curves,
                gt_curves     = result.gt_curves,
                params_pred   = result.params_pred,
                x_axis        = x_axis_np,
                pixels        = pixels,
                title         = profile_titles[tag],
                out_path      = meta.figure_path(f"profiles_{tag}"),
                n_gaussians   = run.n_gaussians,
                pixel_metrics = pixel_metrics_for_plot,
                az_offset     = result.azimuth_offset,
                rg_offset     = result.range_offset,
            )
           
            logger.subsection(f"Profiles ({tag:<6}) : {figure_paths[f'profiles_{tag}']}")

        for key, data, title, label, extra in (
            ("pixel_mse_map",  result.pixel_mse,                             "Per-pixel curve MSE (denorm)",          "MSE",          {"cmap": cfg.cmap_error, "log": True}),
            ("pixel_r2_map",   result.pixel_r2,                              "Per-pixel R² (denorm)",                 "R²",           {"cmap": "RdYlGn", "q_low": 2.0, "q_high": 98.0}),
            ("pixel_peak_map", result.pixel_peak_err_idx.astype(np.float32), "Peak-location absolute error (denorm)", "|Δ peak idx|", {"cmap": cfg.cmap_error}),
        ):
            figure_paths[key] = plotter.plot_pixel_metric_map(
                metric_map = data,
                title      = title,
                label      = label,
                out_path  = meta.figure_path(key),
                az_offset = result.azimuth_offset,
                rg_offset = result.range_offset,
                **extra,
            )
        
        logger.subsection(f"Pixel maps : mse, r2, peak written to {meta.figures_dir}")

        figure_paths["metric_histograms"] = plotter.plot_metric_histogram(
            {
                "pixel_mse (denorm)"    : result.pixel_mse,
                "pixel_r2 (denorm)"     : result.pixel_r2,
                "pixel_cosine (denorm)" : result.pixel_cosine,
            },
            meta.figure_path("metric_histograms"),
        )

        figure_paths["param_maps"] = plotter.plot_param_maps(
            params_pred = result.params_pred[: run.n_gaussians * 3],
            params_gt   = (result.params_gt[: run.n_gaussians * 3] if result.params_gt is not None else None),
            n_gaussians = run.n_gaussians,
            out_path    = meta.figure_path("param_maps"),
            az_offset   = result.azimuth_offset,
            rg_offset   = result.range_offset,
        )
        
        figure_paths["param_distributions"] = plotter.plot_param_distributions(
            params_pred = result.params_pred[: run.n_gaussians * 3],
            params_gt   = (result.params_gt[: run.n_gaussians * 3] if result.params_gt is not None else None),
            n_gaussians = run.n_gaussians,
            out_path    = meta.figure_path("param_distributions"),
        )
        
        figure_paths["param_scatter"] = plotter.plot_param_scatter(
            params_pred = result.params_pred[: run.n_gaussians * 3],
            params_gt   = result.params_gt  [: run.n_gaussians * 3],
            n_gaussians = run.n_gaussians,
            out_path    = meta.figure_path("param_scatter"),
        )
        
        figure_paths["param_error_maps"] = plotter.plot_param_error_maps(
            params_pred = result.params_pred[: run.n_gaussians * 3],
            params_gt   = result.params_gt  [: run.n_gaussians * 3],
            n_gaussians = run.n_gaussians,
            out_path    = meta.figure_path("param_error_maps"),
            az_offset   = result.azimuth_offset,
            rg_offset   = result.range_offset,
        )
        
        logger.subsection(f"Param plots : distributions, scatter, error maps written to {meta.figures_dir}")

        for axis, indices, tag_fn, metric_key in (
            ("range",   slice_range_idx, lambda i: f"slice_range_{int(i) + result.range_offset}",    "ssim_gt_range"),
            ("azimuth", slice_az_idx,    lambda i: f"slice_azimuth_{int(i) + result.azimuth_offset}", "ssim_gt_azimuth"),
        ):
            for s_idx, i in enumerate(indices):
                tag = tag_fn(i)
                figure_paths[tag] = plotter.plot_tomogram_slice(
                    pred_cube  = result.pred_curves,
                    gt_cube    = result.gt_curves,
                    axis       = axis,
                    index      = int(i),
                    x_axis     = x_axis_np,
                    out_path   = meta.figure_path(tag),
                    az_offset  = result.azimuth_offset,
                    rg_offset  = result.range_offset,
                    ssim_value = global_metrics.get(f"{metric_key}_{s_idx}"),
                )

        for s_idx, i in enumerate(slice_elev_idx):
            tag = f"slice_elev_idx_{int(i)}"
            figure_paths[tag] = plotter.plot_elevation_intensity_slice(
                pred_cube  = result.pred_curves,
                gt_cube    = result.gt_curves,
                elev_idx   = int(i),
                x_axis     = x_axis_np,
                out_path   = meta.figure_path(tag),
                az_offset  = result.azimuth_offset,
                rg_offset  = result.range_offset,
                ssim_value = global_metrics.get(f"ssim_gt_elev_{s_idx}"),
            )
        
        logger.subsection(f"Slices written : range={cfg.n_range_slices} azimuth={cfg.n_azimuth_slices} elev={cfg.n_elevation_slices}")

        for axis, n_slices, indices, offset in (
            ("range",   _rg,     all_range_idx, result.range_offset),
            ("azimuth", _az,     all_az_idx,    result.azimuth_offset),
            ("elev",    _N_elev, all_elev_idx,  0),
        ):
            figure_paths[f"ssim_{axis}"] = plotter.plot_ssim_curves(
                global_metrics = global_metrics,
                axis           = axis,
                out_path       = meta.figure_path(f"ssim_{axis}"),
                n_slices       = n_slices,
                slice_indices  = indices,
                ax_offset      = offset,
            )
       
        logger.subsection(f"SSIM plots : range, azimuth, elev written to {meta.figures_dir}\n")

        figure_paths["elev_metric_curves"] = plotter.plot_elev_metric_curves(
            global_metrics = global_metrics,
            out_path       = meta.figure_path("elev_metric_curves"),
            n_elev         = _N_elev,
            x_axis         = x_axis_np,
        )
        
        logger.subsection(f"Elev metric curves (MAE, RMSE, R², CE) written to {meta.figures_dir}\n")

        return figure_paths

    def _run_animations(
        self,
        result    ,
        meta      : InferenceMetadata,
        x_axis_np : np.ndarray,
        cfg       : InferenceConfig,
        logger    : Logger,
    ) -> Dict[str, Path]:

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

    def _build_report(
        self,
        meta           : InferenceMetadata,
        run,
        cfg            : InferenceConfig,
        x_axis_np      : np.ndarray,
        global_metrics : dict,
        figure_paths   : Dict[str, Path],
        gif_paths      : Dict[str, Path],
    ) -> Path:

        run_summary_payload = {
            "model_name"        : run.model_name,
            "in_channels"       : run.in_channels,
            "out_channels"      : run.out_channels,
            "n_gaussians"       : run.n_gaussians,
            "x_axis_length"     : run.x_axis_length,
            "x_axis_min"        : float(x_axis_np.min()),
            "x_axis_max"        : float(x_axis_np.max()),
            "split"             : run.split_name,
            "split_region"      : str(run.split_region.as_tuple()),
            "global_crop"       : str(run.global_crop.as_tuple()),
            "patches"           : run.grid.number_of_patches,
            "patch_size"        : str(run.grid.patch_size),
            "patch_stride"      : run.grid.stride,
            "preprocessing_dir" : str(run.dataset_config.preprocessing_run_directory),
            "input_config"      : run.dataset_config.input_config.as_dict(),
            "used_ema"          : run.used_ema,
        }
        
        inference_cfg_payload = {
            "stitch_window"      : cfg.stitch_window,
            "cube_dtype"         : cfg.cube_dtype,
            "save_cubes"         : cfg.save_cubes,
            "n_best_profiles"    : cfg.n_best_profiles,
            "n_worst_profiles"   : cfg.n_worst_profiles,
            "n_random_profiles"  : cfg.n_random_profiles,
            "n_range_slices"     : cfg.n_range_slices,
            "n_azimuth_slices"   : cfg.n_azimuth_slices,
            "n_elevation_slices" : cfg.n_elevation_slices,
            "gif_axes"           : cfg.gif_axes,
            "gif_fps"            : cfg.gif_fps,
            "gif_max_frames"     : cfg.gif_max_frames,
            "device"             : cfg.device,
            "batch_size"         : run.dataset_config.batch_size,
            "num_workers"        : cfg.num_workers,
        }

        return Report(
            output_dir       = meta.output_dir,
            run_summary      = run_summary_payload,
            inference_config = inference_cfg_payload,
            checkpoint_meta  = run.checkpoint_meta,
            global_metrics   = global_metrics,
            figure_paths     = figure_paths,
            gif_paths        = gif_paths,
        ).assemble()

    def _setup(self, cfg: InferenceConfig) -> tuple[InferenceMetadata, Logger, Ploter]:
        meta = InferenceMetadata(cfg)
        meta.create_dirs()
        np.random.seed(cfg.seed)

        logger = Logger(log_dir=str(meta.logs_dir), name="inference", level=cfg.log_level)
        logger.section("[Inference Pipeline]")
        logger.subsection(f"Run Directory : {cfg.run_directory}")
        logger.subsection(f"Output Dir    : {meta.output_dir}")
        logger.subsection(f"Split         : {cfg.split}")
        logger.subsection(f"Device        : {cfg.device}")
        logger.subsection(f"Use EMA       : {cfg.use_ema}\n")

        plotter = Ploter(
            cmap      = cfg.cmap_intensity,
            err_cmap  = cfg.cmap_error,
            normalize = cfg.normalize_intensity,
            fig_dpi   = cfg.fig_dpi,
            save_dpi  = cfg.save_dpi,
        )

        return meta, logger, plotter

    def _load_run(self, cfg: InferenceConfig, meta: InferenceMetadata, logger: Logger):
        loader = RunLoader(cfg.run_directory, logger=logger)
        return loader.load(
            split           = cfg.split,
            batch_size      = cfg.batch_size,
            num_workers     = cfg.num_workers,
            device          = cfg.device,
            use_ema         = cfg.use_ema,
            checkpoint_name = cfg.checkpoint_name,
        )

    def _predict(self, cfg: InferenceConfig, meta: InferenceMetadata, run, logger: Logger):
        predictor = Predictor(
            run         = run,
            logger      = logger,
            window_kind = cfg.stitch_window,
            cube_dtype  = cfg.cube_dtype,
            save_cubes  = cfg.save_cubes,
            meta        = meta,
            cpu_workers = cfg.cpu_workers,
        )
        return predictor.run_inference()

    def _compute_slice_indices(self, cfg: InferenceConfig, n_elev: int, n_az: int, n_rg: int) -> dict:
        def _equal_indices(n_total: int, n_slices: int) -> np.ndarray:
            n_slices = max(1, min(n_slices, n_total))
            return np.linspace(n_total * 0.1, n_total * 0.9, n_slices).round().astype(int)

        return {
            "slice_elev_idx"  : _equal_indices(n_elev, cfg.n_elevation_slices),
            "slice_range_idx" : _equal_indices(n_rg,   cfg.n_range_slices),
            "slice_az_idx"    : _equal_indices(n_az,   cfg.n_azimuth_slices),
            "all_elev_idx"    : np.arange(n_elev),
            "all_range_idx"   : np.arange(n_rg),
            "all_az_idx"      : np.arange(n_az),
        }

    def _evaluate_metrics(self, result, x_axis_np: np.ndarray, run, meta: InferenceMetadata, indices: dict) -> dict:
        global_metrics = Metrics(result, x_axis_np, run.n_gaussians).compute(
            elev_indices  = indices["all_elev_idx"],
            range_indices = indices["all_range_idx"],
            az_indices    = indices["all_az_idx"],
        )
        write_metrics_json(global_metrics, meta.metrics_path)
        return global_metrics

    def run(self) -> Path:
        cfg                    = self.config
        meta, logger, plotter  = self._setup(cfg)
        run                    = self._load_run(cfg, meta, logger)
        result                 = self._predict(cfg, meta, run, logger)

        x_axis_np         = np.asarray(run.x_axis, dtype=np.float64)
        _N_elev, _az, _rg = result.pred_curves.shape

        indices        = self._compute_slice_indices(cfg, _N_elev, _az, _rg)
        global_metrics = self._evaluate_metrics(result, x_axis_np, run, meta, indices)

        logger.section("[Inference: Plots]")
        figure_paths = self._plot_figures(
            plotter         = plotter,
            result          = result,
            run             = run,
            meta            = meta,
            global_metrics  = global_metrics,
            x_axis_np       = x_axis_np,
            cfg             = cfg,
            logger          = logger,
            slice_range_idx = indices["slice_range_idx"],
            slice_az_idx    = indices["slice_az_idx"],
            slice_elev_idx  = indices["slice_elev_idx"],
            all_range_idx   = indices["all_range_idx"],
            all_az_idx      = indices["all_az_idx"],
            all_elev_idx    = indices["all_elev_idx"],
        )

        logger.section("[Inference: Animations]")
        gif_paths = self._run_animations(
            result    = result,
            meta      = meta,
            x_axis_np = x_axis_np,
            cfg       = cfg,
            logger    = logger,
        )

        logger.section("[Inference: Report]")
        report_path = self._build_report(
            meta           = meta,
            run            = run,
            cfg            = cfg,
            x_axis_np      = x_axis_np,
            global_metrics = global_metrics,
            figure_paths   = figure_paths,
            gif_paths      = gif_paths,
        )

        logger.section("[Inference Pipeline Done]")
        logger.subsection(f"Report  : {report_path}")
        logger.subsection(f"Metrics : {meta.metrics_path}")
        logger.subsection(f"Cubes   : {result.cube_directory}\n")
        logger.close()

        return report_path
