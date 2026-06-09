from __future__ import annotations

from pathlib import Path
from typing  import Dict, List

import numpy as np

from configuration.inference_config         import InferenceConfig
from pipelines.inference_pipeline.baseline  import ClassicalBaseline
from pipelines.inference_pipeline.figures   import FigureComposer
from pipelines.inference_pipeline.loader    import InferenceMetadata, RunLoader
from pipelines.inference_pipeline.metrics   import Metrics
from pipelines.inference_pipeline.plots     import Ploter
from pipelines.inference_pipeline.predictor import Predictor
from pipelines.inference_pipeline.report    import Report, ReportPayloadBuilder
from tools.logger                           import Logger


class InferencePipeline:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config

    def _build_report(
        self,
        meta           : InferenceMetadata,
        run,
        cfg            : InferenceConfig,
        x_axis_np      : np.ndarray,
        global_metrics : dict,
        figure_paths   : Dict[str, List[Path]],
        gif_paths      : Dict[str, Path],
    ) -> Path:

        return Report(
            output_dir       = meta.output_dir,
            run_summary      = ReportPayloadBuilder.run_summary(run, x_axis_np),
            inference_config = ReportPayloadBuilder.inference_config(cfg, run),
            checkpoint_meta  = run.checkpoint_meta,
            global_metrics   = global_metrics,
            figure_paths     = figure_paths,
            gif_paths        = gif_paths,
            report_path      = meta.report_path,
        ).assemble()

    def _setup(self, cfg: InferenceConfig) -> tuple[InferenceMetadata, Logger, Ploter]:
        meta = InferenceMetadata(cfg)
        meta.create_dirs()
        np.random.seed(cfg.seed)

        logger = Logger(log_dir=str(meta.logs_dir), name="inference", level=cfg.log_level)
        logger.section("[Inference Pipeline]")
        logger.kv_table({
            "Run Directory": cfg.run_directory,
            "Output Dir":    meta.output_dir,
            "Split":         cfg.split,
            "Device":        cfg.device,
        })

        plotter = Ploter(
            cmap      = cfg.cmap_intensity,
            err_cmap  = cfg.cmap_error,
            normalize = cfg.normalize_intensity,
            fig_dpi   = cfg.fig_dpi,
            save_dpi  = cfg.save_dpi,
        )

        return meta, logger, plotter

    def _load_run(self, cfg: InferenceConfig, logger: Logger):
        loader = RunLoader(cfg.run_directory, logger=logger)
        return loader.load(
            split           = cfg.split,
            batch_size      = cfg.batch_size,
            num_workers     = cfg.num_workers,
            device          = cfg.device,
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

    def _attach_classical_baseline(self, cfg: InferenceConfig, run, result, logger: Logger) -> None:
        if not cfg.compare_classical:
            return

        baseline = ClassicalBaseline(
            run_directory     = cfg.run_directory,
            logger            = logger,
            preprocessing_dir = run.dataset_config.preprocessing_run_directory,
            window            = tuple(cfg.capon_window) if cfg.capon_window is not None else None,
            loading           = cfg.capon_loading,
            phase_sign        = cfg.capon_phase_sign,
        )

        reduced = baseline.compute(
            complex_inputs   = run.complex_inputs,
            n_secondaries    = run.n_secondaries,
            x_axis           = run.x_axis,
            secondary_labels = run.secondary_labels,
        )

        result.attach_reduced(reduced)

        if cfg.save_cubes:
            np.save(result.cube_directory / "reduced_curves.npy",      result.reduced_curves)
            np.save(result.cube_directory / "reduced_curves_norm.npy", result.reduced_norm_curves)

    @staticmethod
    def _equal_indices(n_total: int, n_slices: int) -> np.ndarray:
        n_slices = max(1, min(n_slices, n_total))
        return np.linspace(n_total * 0.1, n_total * 0.9, n_slices).round().astype(int)

    def _compute_slice_indices(self, cfg: InferenceConfig, n_elev: int, n_az: int, n_rg: int) -> dict:
        return {
            "slice_elev_idx"  : self._equal_indices(n_elev, cfg.n_elevation_slices),
            "slice_range_idx" : self._equal_indices(n_rg,   cfg.n_range_slices),
            "slice_az_idx"    : self._equal_indices(n_az,   cfg.n_azimuth_slices),
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

        if run.track_baselines is not None:
            global_metrics["tracks"] = run.track_baselines.to_payload()

        if run.track_profiles is not None:
            global_metrics["track_positions"] = run.track_profiles.position_summary()

        Metrics.write_json(global_metrics, meta.metrics_path)
        return global_metrics

    def run(self) -> Path:
        cfg                    = self.config
        meta, logger, plotter  = self._setup(cfg)
        run                    = self._load_run(cfg, logger)
        result                 = self._predict(cfg, meta, run, logger)

        self._attach_classical_baseline(cfg, run, result, logger)

        x_axis_np         = np.asarray(run.x_axis, dtype=np.float64)
        _N_elev, _az, _rg = result.pred_curves.shape

        indices        = self._compute_slice_indices(cfg, _N_elev, _az, _rg)
        global_metrics = self._evaluate_metrics(result, x_axis_np, run, meta, indices)

        composer = FigureComposer(plotter=plotter, meta=meta, logger=logger, cfg=cfg)

        logger.section("[Inference: Plots]")
        figure_paths = composer.compose(
            result         = result,
            run            = run,
            global_metrics = global_metrics,
            x_axis_np      = x_axis_np,
            indices        = indices,
        )

        logger.section("[Inference: Animations]")
        gif_paths = composer.animate(result, x_axis_np)

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
