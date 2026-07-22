from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path
from typing      import Dict, List, Type

import numpy as np

from configuration.inference                         import InferenceConfig
from pipelines.backbone.inference.data_consistency   import DataConsistencyEvaluator
from pipelines.backbone.inference.figures            import FigureComposer
from pipelines.backbone.inference.loader             import RunLoader
from pipelines.backbone.inference.run_metadata_paths import InferenceMetadata
from pipelines.backbone.inference.metrics            import Metrics
from pipelines.backbone.inference.plots              import Plotter
from pipelines.backbone.inference.predictor          import Predictor
from pipelines.backbone.inference.reduced            import ReducedTomogramSynthesizer
from pipelines.backbone.inference.report             import Report, ReportPayloadBuilder
from tools.monitoring.logger                         import Logger
from tools.reporting.plotting                        import PlotBase
from tools.runtime.completion                        import CompletionMarker


@dataclass(frozen=True)
class InferenceComponents:
    loader_cls              : Type[RunLoader] = RunLoader
    predictor_cls           : Type[Predictor] = Predictor
    param_space             : bool            = True
    embedding_evaluator_cls : Type | None     = None


class InferencePipeline:
    def __init__(self, config: InferenceConfig, components: InferenceComponents | None = None) -> None:
        self.config     = config
        self.components = components if components is not None else InferenceComponents()

    def _setup(self, cfg: InferenceConfig) -> tuple[InferenceMetadata, Logger, Plotter]:
        meta = InferenceMetadata(cfg)
        meta.create_dirs()
        np.random.seed(cfg.seed)

        evaluator = self.components.embedding_evaluator_cls

        logger = Logger(log_dir=str(meta.logs_dir), name="inference", level=cfg.log_level)
        logger.section("[Inference Pipeline]")
        logger.kv_table({
            "Run Directory"       : cfg.run_directory,
            "Output Dir"          : meta.output_dir,
            "Split"               : cfg.split,
            "Device"              : cfg.device,
            "Checkpoint"          : cfg.checkpoint_name,
            "Loader"              : self.components.loader_cls.__name__,
            "Predictor"           : self.components.predictor_cls.__name__,
            "Param Space"         : self.components.param_space,
            "Embedding Evaluator" : evaluator.__name__ if evaluator is not None else "none",
        })

        PlotBase.use_style(cfg.figure_style)

        plotter = Plotter(
            cmap      = cfg.cmap_intensity,
            err_cmap  = cfg.cmap_error,
            normalize = cfg.normalize_intensity,
            fig_dpi   = cfg.fig_dpi,
            save_dpi  = cfg.save_dpi,
        )

        return meta, logger, plotter

    def _load_run(self, cfg: InferenceConfig, logger: Logger):
        loader = self.components.loader_cls(cfg.run_directory, logger=logger)
        return loader.load(
            split           = cfg.split,
            batch_size      = cfg.batch_size,
            num_workers     = cfg.num_workers,
            device          = cfg.device,
            checkpoint_name = cfg.checkpoint_name,
        )

    def _predict(self, cfg: InferenceConfig, meta: InferenceMetadata, run, logger: Logger):
        predictor = self.components.predictor_cls(
            run         = run,
            logger      = logger,
            window_kind = cfg.stitch_window,
            cube_dtype  = cfg.cube_dtype,
            save_cubes  = cfg.save_cubes,
            meta        = meta,
            cpu_workers = cfg.cpu_workers,
        )
        return predictor.run_inference()

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
            param_space   = self.components.param_space,
        )

        global_metrics["split"]        = run.split_name
        global_metrics["split_region"] = list(run.split_region.as_tuple())

        if run.track_baselines is not None:
            global_metrics["tracks"] = run.track_baselines.to_payload()

        if run.track_profiles is not None:
            global_metrics["track_positions"] = run.track_profiles.position_summary()

        Metrics.write_json(global_metrics, meta.metrics_path)
        return global_metrics

    def _evaluate_embeddings(self, meta: InferenceMetadata, run, global_metrics: dict, logger: Logger) -> None:
        if self.components.embedding_evaluator_cls is None:
            return

        evaluator = self.components.embedding_evaluator_cls(run, logger)

        global_metrics.update(evaluator.run())
        Metrics.write_json(global_metrics, meta.metrics_path)

    def _synthesize_reduced(self, cfg: InferenceConfig, meta: InferenceMetadata, run, result, x_axis_np: np.ndarray, global_metrics: dict, indices: dict, logger: Logger) -> None:
        if not cfg.compute_reduced:
            logger.section("[Inference: Reduced Capon Baseline skipped]")
            logger.subsection("compute_reduced is disabled; the NN-vs-reduced-vs-GT comparison is absent from metrics, figures, and report.")
            global_metrics["reduced_status"] = "skipped: compute_reduced disabled"
            Metrics.write_json(global_metrics, meta.metrics_path)
            return

        if run.secondary_labels is None:
            logger.section("[Inference: Reduced Capon Baseline skipped]")
            logger.subsection("Run uses the full secondary stack; the reduced tomogram would equal the ground-truth tomogram, so no baseline comparison exists.")
            global_metrics["reduced_status"] = "skipped: full secondary stack"
            Metrics.write_json(global_metrics, meta.metrics_path)
            return

        synth     = ReducedTomogramSynthesizer(run, meta, cfg, logger)
        synthesis = synth.run(result.gt_curves)

        comparison = Metrics(result, x_axis_np, run.n_gaussians).reduced_comparison(
            synthesis.curves,
            elev_indices  = indices["all_elev_idx"],
            range_indices = indices["all_range_idx"],
            az_indices    = indices["all_az_idx"],
        )

        result.reduced = comparison
        global_metrics["reduced_status"] = "computed"
        global_metrics.update(synthesis.orientation)
        global_metrics.update(comparison.metrics)
        Metrics.write_json(global_metrics, meta.metrics_path)

        logger.subsection(f"Reduced baseline merged : relative MSE reduction = {comparison.metrics['relative_mse_reduction']:.4f}, NN beats reduced on {comparison.metrics['fraction_pred_beats_reduced'] * 100.0:.1f}% of pixels")

    def _evaluate_data_consistency(self, cfg: InferenceConfig, meta: InferenceMetadata, run, result, x_axis_np: np.ndarray, global_metrics: dict, logger: Logger) -> None:
        if not cfg.compute_data_consistency:
            logger.section("[Inference: Data Consistency skipped]")
            logger.subsection("compute_data_consistency is disabled; interferometric-consistency metrics are absent from metrics, figures, and report.")
            global_metrics["data_consistency_status"] = "skipped: compute_data_consistency disabled"
            Metrics.write_json(global_metrics, meta.metrics_path)
            return

        evaluator   = DataConsistencyEvaluator(run, cfg, logger)
        consistency = evaluator.run(result.pred_curves, result.gt_curves, x_axis_np)

        result.data_consistency = consistency
        global_metrics["data_consistency_status"] = "computed"
        global_metrics.update(consistency.metrics)
        Metrics.write_json(global_metrics, meta.metrics_path)

        if cfg.save_cubes:
            np.save(meta.cube_dir / "physics_coherence_error.npy",  consistency.coherence_error_map)
            np.save(meta.cube_dir / "physics_covariance_error.npy", consistency.covariance_error_map)
            np.save(meta.cube_dir / "physics_valid_mask.npy",       consistency.valid_mask)

    def _compose_figures(self, composer: FigureComposer, result, run, global_metrics: dict, x_axis_np: np.ndarray, indices: dict) -> Dict[str, List[Path]]:
        return composer.compose(
            result         = result,
            run            = run,
            global_metrics = global_metrics,
            x_axis_np      = x_axis_np,
            indices        = indices,
            param_space    = self.components.param_space,
        )

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

    def run(self) -> Path:
        cfg                    = self.config
        meta, logger, plotter  = self._setup(cfg)
        CompletionMarker.clear(meta.output_dir)
        run    = self._load_run(cfg, logger)
        result = self._predict(cfg, meta, run, logger)

        x_axis_np         = np.asarray(run.x_axis, dtype=np.float64)
        _N_elev, _az, _rg = result.pred_curves.shape

        logger.section("[Inference: Metrics]")
        indices        = self._compute_slice_indices(cfg, _N_elev, _az, _rg)
        global_metrics = self._evaluate_metrics(result, x_axis_np, run, meta, indices)
        logger.subsection(f"Global metrics written : {meta.metrics_path}")

        self._evaluate_embeddings(meta, run, global_metrics, logger)
        self._synthesize_reduced(cfg, meta, run, result, x_axis_np, global_metrics, indices, logger)
        self._evaluate_data_consistency(cfg, meta, run, result, x_axis_np, global_metrics, logger)

        figure_paths = {}
        gif_paths    = {}

        if cfg.save_plots or cfg.save_animations:
            composer = FigureComposer(plotter=plotter, meta=meta, logger=logger, cfg=cfg)

            if cfg.save_plots:
                logger.section("[Inference: Plots]")
                figure_paths = self._compose_figures(composer, result, run, global_metrics, x_axis_np, indices)
            else:
                logger.section("[Inference: Plots skipped]")

            if cfg.save_animations:
                logger.section("[Inference: Animations]")
                gif_paths = composer.animate(result, run, x_axis_np)
            else:
                logger.section("[Inference: Animations skipped]")
        else:
            logger.section("[Inference: Plots and animations skipped]")

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

        CompletionMarker.stamp(meta.output_dir, {
            "stage" : "inference",
            "split" : run.split_name,
        })

        logger.section("[Inference Pipeline Done]")
        logger.subsection(f"Report  : {report_path}")
        logger.subsection(f"Metrics : {meta.metrics_path}")
        logger.subsection(f"Cubes   : {result.cube_directory}\n")
        logger.close()

        return report_path
