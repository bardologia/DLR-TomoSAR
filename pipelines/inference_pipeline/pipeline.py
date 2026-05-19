from __future__ import annotations

from datetime import datetime
from pathlib  import Path
from typing   import Dict

import numpy as np

from pipelines.inference_pipeline.animation import make_walk_gif
from configuration.inference_config         import InferenceConfig
from pipelines.inference_pipeline.loader    import DirectoryLoader
from pipelines.inference_pipeline.metrics   import Metrics
from pipelines.inference_pipeline.plots     import Ploter
from pipelines.inference_pipeline.predictor import Predictor
from pipelines.inference_pipeline.report    import Report, write_metrics_json
from tools.logger                           import Logger


class InferencePipeline:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config

    def _resolve_output_dir(self) -> Path:
        base = self.config.run_directory / "inference"
        if self.config.output_subdir:
            return base / self.config.output_subdir
       
        return base / datetime.now().strftime("%Y%m%d_%H%M%S")

    def _plot_figures(
        self,
        plotter         : Ploter,
        result,
        run,
        figures_dir     : Path,
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
            "mse"     : result.pixel_mse,
            "mae"     : result.pixel_mae,
            "r2"      : result.pixel_r2,
            "cos"     : result.pixel_cosine,
            "mse_raw" : result.pixel_mse_raw,
            "r2_raw"  : result.pixel_r2_raw,
            "cos_raw" : result.pixel_cosine_raw,
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
                raw_curves    = result.raw_curves,
                params_pred   = result.params_pred,
                x_axis        = x_axis_np,
                pixels        = pixels,
                title         = profile_titles[tag],
                out_path      = figures_dir / f"profiles_{tag}.png",
                n_gaussians   = run.n_gaussians,
                pixel_metrics = pixel_metrics_for_plot,
                az_offset     = result.azimuth_offset,
                rg_offset     = result.range_offset,
            )
            logger.subsection(f"Profiles ({tag:<6}) : {figure_paths[f'profiles_{tag}']}")

        figure_paths["pixel_mse_map"] = plotter.plot_pixel_metric_map(
            result.pixel_mse, "Per-pixel curve MSE", "MSE",
            figures_dir / "pixel_mse_map.png",
            az_offset = result.azimuth_offset,
            rg_offset = result.range_offset,
            cmap      = cfg.cmap_error,
            log       = True,
        )
        
        figure_paths["pixel_r2_map"] = plotter.plot_pixel_metric_map(
            result.pixel_r2, "Per-pixel R²", "R²",
            figures_dir / "pixel_r2_map.png",
            az_offset = result.azimuth_offset,
            rg_offset = result.range_offset,
            cmap      = "RdYlGn",
            q_low     = 2.0,
            q_high    = 98.0,
        )
        
        figure_paths["pixel_peak_map"] = plotter.plot_pixel_metric_map(
            result.pixel_peak_err_idx.astype(np.float32),
            "Peak-location absolute error", "|Δ peak idx|",
            figures_dir / "pixel_peak_map.png",
            az_offset = result.azimuth_offset,
            rg_offset = result.range_offset,
            cmap      = cfg.cmap_error,
        )
        logger.subsection(f"Pixel maps      : mse, r2, peak written to {figures_dir}")

        figure_paths["metric_histograms"] = plotter.plot_metric_histogram(
            {
                "pixel_mse (gt)"    : result.pixel_mse,
                "pixel_r2 (gt)"     : result.pixel_r2,
                "pixel_cosine (gt)" : result.pixel_cosine,
                "pixel_mse (raw)"   : result.pixel_mse_raw,
                "pixel_r2 (raw)"    : result.pixel_r2_raw,
                "pixel_cosine (raw)": result.pixel_cosine_raw,
            },
            figures_dir / "metric_histograms.png",
        )

        figure_paths["param_maps"] = plotter.plot_param_maps(
            params_pred = result.params_pred[: run.n_gaussians * 3],
            params_gt   = (result.params_gt[: run.n_gaussians * 3] if result.params_gt is not None else None),
            n_gaussians = run.n_gaussians,
            out_path    = figures_dir / "param_maps.png",
            az_offset   = result.azimuth_offset,
            rg_offset   = result.range_offset,
        )
        
        figure_paths["param_distributions"] = plotter.plot_param_distributions(
            params_pred = result.params_pred[: run.n_gaussians * 3],
            params_gt   = (result.params_gt[: run.n_gaussians * 3] if result.params_gt is not None else None),
            n_gaussians = run.n_gaussians,
            out_path    = figures_dir / "param_distributions.png",
        )
        
        
        figure_paths["param_scatter"] = plotter.plot_param_scatter(
            params_pred = result.params_pred[: run.n_gaussians * 3],
            params_gt   = result.params_gt  [: run.n_gaussians * 3],
            n_gaussians = run.n_gaussians,
            out_path    = figures_dir / "param_scatter.png",
        )
        
        figure_paths["param_error_maps"] = plotter.plot_param_error_maps(
            params_pred = result.params_pred[: run.n_gaussians * 3],
            params_gt   = result.params_gt  [: run.n_gaussians * 3],
            n_gaussians = run.n_gaussians,
            out_path    = figures_dir / "param_error_maps.png",
            az_offset   = result.azimuth_offset,
            rg_offset   = result.range_offset,
        )
        
        logger.subsection(f"Param plots     : distributions, scatter, error maps written to {figures_dir}")

        for s_idx, i in enumerate(slice_range_idx):
            tag = f"slice_range_{int(i) + result.range_offset}"
            figure_paths[tag] = plotter.plot_tomogram_slice(
                result.pred_curves, result.gt_curves, result.raw_curves,
                axis       = "range",
                index      = int(i),
                x_axis     = x_axis_np,
                out_path   = figures_dir / f"{tag}.png",
                az_offset  = result.azimuth_offset,
                rg_offset  = result.range_offset,
                ssim_value = global_metrics.get(f"ssim_gt_range_{s_idx}"),
            )
        
        for s_idx, i in enumerate(slice_az_idx):
            tag = f"slice_azimuth_{int(i) + result.azimuth_offset}"
            figure_paths[tag] = plotter.plot_tomogram_slice(
                result.pred_curves, result.gt_curves, result.raw_curves,
                axis       = "azimuth",
                index      = int(i),
                x_axis     = x_axis_np,
                out_path   = figures_dir / f"{tag}.png",
                az_offset  = result.azimuth_offset,
                rg_offset  = result.range_offset,
                ssim_value = global_metrics.get(f"ssim_gt_azimuth_{s_idx}"),
            )
        
        for s_idx, i in enumerate(slice_elev_idx):
            tag = f"slice_elev_idx_{int(i)}"
            figure_paths[tag] = plotter.plot_elevation_intensity_slice(
                result.pred_curves, result.gt_curves, result.raw_curves,
                elev_idx   = int(i),
                x_axis     = x_axis_np,
                out_path   = figures_dir / f"{tag}.png",
                az_offset  = result.azimuth_offset,
                rg_offset  = result.range_offset,
                ssim_value = global_metrics.get(f"ssim_gt_elev_{s_idx}"),
            )
        
        logger.subsection(f"Slices written  : range={cfg.n_range_slices} azimuth={cfg.n_azimuth_slices} elev={cfg.n_elevation_slices}")

        figure_paths["ssim_range"] = plotter.plot_ssim_curves(
            global_metrics = global_metrics,
            axis           = "range",
            out_path       = figures_dir / "ssim_range.png",
            n_slices       = _rg,
            slice_indices  = all_range_idx,
            ax_offset      = result.range_offset,
        )
       
        figure_paths["ssim_azimuth"] = plotter.plot_ssim_curves(
            global_metrics = global_metrics,
            axis           = "azimuth",
            out_path       = figures_dir / "ssim_azimuth.png",
            n_slices       = _az,
            slice_indices  = all_az_idx,
            ax_offset      = result.azimuth_offset,
        )
        
        figure_paths["ssim_elev"] = plotter.plot_ssim_curves(
            global_metrics = global_metrics,
            axis           = "elev",
            out_path       = figures_dir / "ssim_elev.png",
            n_slices       = _N_elev,
            slice_indices  = all_elev_idx,
            ax_offset      = 0,
        )
       
        logger.subsection(f"SSIM plots      : range, azimuth, elev written to {figures_dir}\n")

        figure_paths["elev_metric_curves"] = plotter.plot_elev_metric_curves(
            global_metrics = global_metrics,
            out_path       = figures_dir / "elev_metric_curves.png",
            n_elev         = _N_elev,
            x_axis         = x_axis_np,
        )
        
        logger.subsection(f"Elev metric curves (MAE, RMSE, R², CE) written to {figures_dir}\n")

        return figure_paths

    def _run_animations(
        self,
        result    ,
        gif_dir   : Path,
        x_axis_np : np.ndarray,
        cfg       : InferenceConfig,
        logger    : Logger,
    ) -> Dict[str, Path]:

        gif_paths: Dict[str, Path] = {}
        for axis in cfg.gif_axes: 
            logger.subsection(f"Generating walk GIF along {axis} axis")
            gif_paths[f"walk_{axis}"] = make_walk_gif(
                pred_cube   = result.pred_curves,
                gt_cube     = result.gt_curves,
                raw_cube    = result.raw_curves,
                axis        = axis,
                out_path    = gif_dir / f"walk_{axis}.gif",
                x_axis      = x_axis_np,
                az_offset   = result.azimuth_offset,
                rg_offset   = result.range_offset,
                fps         = cfg.gif_fps,
                max_frames  = cfg.gif_max_frames,
                dpi         = cfg.gif_dpi,
                cmap        = cfg.cmap_intensity,
                err_cmap    = cfg.cmap_error,
                num_workers = cfg.gif_workers,
            )
            logger.subsection(f"GIF ({axis:<9}) : {gif_paths[f'walk_{axis}']} \n")
        
        logger.subsection("")

        return gif_paths

    def _build_report(
        self,
        output_dir     : Path,
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
            "has_noise_head"    : run.has_noise_head,
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
            output_dir       = output_dir,
            run_summary      = run_summary_payload,
            inference_config = inference_cfg_payload,
            checkpoint_meta  = run.checkpoint_meta,
            global_metrics   = global_metrics,
            figure_paths     = figure_paths,
            gif_paths        = gif_paths,
        ).assemble()

    def run(self) -> Path:
        cfg = self.config
        np.random.seed(cfg.seed)
        
        plotter = Ploter(
            cmap      = cfg.cmap_intensity,
            err_cmap  = cfg.cmap_error,
            normalize = cfg.normalize_intensity,
            fig_dpi   = cfg.fig_dpi,
            save_dpi  = cfg.save_dpi,
        )

        output_dir  = self._resolve_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        gif_dir     = output_dir / "animations"
        gif_dir.mkdir(parents=True, exist_ok=True)

        logger = Logger(log_dir=str(output_dir / "logs"), name="inference", level=cfg.log_level)

        logger.section("[Inference Pipeline]")
        logger.subsection(f"Run Directory : {cfg.run_directory}")
        logger.subsection(f"Output Dir    : {output_dir}")
        logger.subsection(f"Split         : {cfg.split}")
        logger.subsection(f"Device        : {cfg.device}")
        logger.subsection(f"Use EMA       : {cfg.use_ema}\n")

        loader = DirectoryLoader(cfg.run_directory, logger=logger)
        run    = loader.load(
            split           = cfg.split,
            batch_size      = cfg.batch_size,
            num_workers     = cfg.num_workers,
            device          = cfg.device,
            use_ema         = cfg.use_ema,
            checkpoint_name = cfg.checkpoint_name,
        )

        predictor = Predictor(
            run         = run,
            logger      = logger,
            window_kind = cfg.stitch_window,
            cube_dtype  = cfg.cube_dtype,
            save_cubes  = cfg.save_cubes,
            output_dir  = output_dir,
        )
        result = predictor.run_inference()

        x_axis_np         = np.asarray(run.x_axis, dtype=np.float64)
        _N_elev, _az, _rg = result.pred_curves.shape

        def _equal_indices(n_total: int, n_slices: int) -> np.ndarray:
            n_slices = max(1, min(n_slices, n_total))
            return np.linspace(n_total * 0.1, n_total * 0.9, n_slices).round().astype(int)

        slice_elev_idx  = _equal_indices(_N_elev, cfg.n_elevation_slices)
        slice_range_idx = _equal_indices(_rg,     cfg.n_range_slices)
        slice_az_idx    = _equal_indices(_az,     cfg.n_azimuth_slices)

        all_elev_idx  = np.arange(_N_elev)
        all_range_idx = np.arange(_rg)
        all_az_idx    = np.arange(_az)

        global_metrics = Metrics(result, x_axis_np, run.n_gaussians).compute(
            elev_indices  = all_elev_idx,
            range_indices = all_range_idx,
            az_indices    = all_az_idx,
        )
        write_metrics_json(global_metrics, output_dir / "metrics.json")

        logger.section("[Inference: Plots]")
        figure_paths = self._plot_figures(
            plotter         = plotter,
            result          = result,
            run             = run,
            figures_dir     = figures_dir,
            global_metrics  = global_metrics,
            x_axis_np       = x_axis_np,
            cfg             = cfg,
            logger          = logger,
            slice_range_idx = slice_range_idx,
            slice_az_idx    = slice_az_idx,
            slice_elev_idx  = slice_elev_idx,
            all_range_idx   = all_range_idx,
            all_az_idx      = all_az_idx,
            all_elev_idx    = all_elev_idx,
        )

        self.logger.section("[Inference: Animations]")
        gif_paths = self._run_animations(
            result    = result,
            gif_dir   = gif_dir,
            x_axis_np = x_axis_np,
            cfg       = cfg,
            logger    = logger,
        )

        self.logger.section("[Inference: Report]")
        report_path = self._build_report(
            output_dir     = output_dir,
            run            = run,
            cfg            = cfg,
            x_axis_np      = x_axis_np,
            global_metrics = global_metrics,
            figure_paths   = figure_paths,
            gif_paths      = gif_paths,
        )

        logger.section("[Inference Pipeline Done]")
        logger.subsection(f"Report  : {report_path}")
        logger.subsection(f"Metrics : {output_dir / 'metrics.json'}")
        logger.subsection(f"Cubes   : {result.cube_directory}\n")
        logger.close()
        
        return report_path
