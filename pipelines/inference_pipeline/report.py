from __future__ import annotations

from datetime import datetime
from pathlib  import Path
from typing   import Any, Dict, List, Optional, Tuple

import numpy as np

from pipelines.shared.reporting import ReportAssets
from pipelines.shared.scoring   import MetricOrientation, RelativeImprovement
from tools.markdown             import MarkdownTable, ScalarFormatter


class ReportPayloadBuilder:
    @staticmethod
    def run_summary(run, x_axis_np: np.ndarray) -> Dict[str, Any]:
        return {
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
            "secondary_labels"  : ", ".join(run.secondary_labels) if run.secondary_labels else "all passes",
        }

    @staticmethod
    def inference_config(cfg, run) -> Dict[str, Any]:
        return {
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


class Report:

    _DATASET_KEYS = frozenset({
        "gt_mean", "gt_std", "gt_max",
        "pred_mean", "pred_std", "pred_max",
        "n_pixels", "n_elevation",
        "x_axis_min", "x_axis_max", "x_axis_step",
    })

    def __init__(
        self,
        output_dir       : Path,
        run_summary      : Dict,
        inference_config : Dict,
        checkpoint_meta  : Dict,
        global_metrics   : Dict,
        figure_paths     : Dict[str, Path],
        gif_paths        : Dict[str, Path],
        report_path      : Path,
        extra_sections   : Optional[List[str]] = None,
    ) -> None:

        self.report_path      = Path(report_path)
        self.output_dir       = Path(output_dir)
        self.run_summary      = run_summary
        self.inference_config = inference_config
        self.checkpoint_meta  = checkpoint_meta
        self.global_metrics   = global_metrics
        self.figure_paths     = figure_paths
        self.gif_paths        = gif_paths
        self.extra_sections   = extra_sections or []
        self.assets           = ReportAssets(self.output_dir)

    @staticmethod
    def _fmt(v: Any) -> str:
        return ScalarFormatter.format_scalar(v, precision=6, adaptive=True)

    @staticmethod
    def _kv_table(rows: List[Tuple[str, Any]], header: Tuple[str, str] = ("Key", "Value")) -> str:
        table = MarkdownTable(header)
        for k, v in rows:
            table.add_row(f"`{k}`", Report._fmt(v))
        return "\n".join(table.render())

    @staticmethod
    def _dict_table(d: Dict[str, Any]) -> str:
        return Report._kv_table(sorted(d.items()))

    @staticmethod
    def _three_col_table(
        rows   : List[Tuple[str, Any, str]],
        header : Tuple[str, str, str] = ("Metric", "Pred vs GT", "Description"),
    ) -> str:
        table = MarkdownTable(header)
        for label, gt_val, desc in rows:
            table.add_row(label, Report._fmt(gt_val), desc)
        return "\n".join(table.render())

    @staticmethod
    def _padded_column(payload: Dict[str, Any], key: str, n_tracks: int) -> list:
        values = payload[key]
        return values if len(values) == n_tracks else [float("nan")] * n_tracks

    @staticmethod
    def _is_per_slice_ssim(k: str) -> bool:
        for prefix in (
            "ssim_gt_elev_", "ssim_gt_range_", "ssim_gt_azimuth_",
            "ssim_red_elev_", "ssim_red_range_", "ssim_red_azimuth_",
        ):
            if k.startswith(prefix) and k[len(prefix):].lstrip("-").isdigit():
                return True
        for prefix in (
            "elev_mae_gt_",  "elev_rmse_gt_",  "elev_r2_gt_",  "elev_ce_gt_",
            "elev_mae_red_", "elev_rmse_red_", "elev_r2_red_", "elev_ce_red_",
        ):
            if k.startswith(prefix) and k[len(prefix):].lstrip("-").isdigit():
                return True
        return False

    def _build_run_summary(self) -> List[str]:
        rs  = self.run_summary
        cfg = self.inference_config
        ck  = self.checkpoint_meta
        out = ["\n## 1. Run summary\n"]

        out.append("\n### 1.1 Model\n")
        out.append(self._kv_table([
            ("model_name",     rs["model_name"]),
            ("in_channels",    rs["in_channels"]),
            ("out_channels",   rs["out_channels"]),
            ("n_gaussians",    rs["n_gaussians"]),
        ]))
        out.append("")

        out.append("\n### 1.2 Dataset & patch grid\n")
        out.append(self._kv_table([
            ("split",             rs["split"]),
            ("split_region",      rs["split_region"]),
            ("global_crop",       rs["global_crop"]),
            ("secondary_labels",  rs["secondary_labels"]),
            ("patches",           rs["patches"]),
            ("patch_size",        rs["patch_size"]),
            ("patch_stride",      rs["patch_stride"]),
            ("preprocessing_dir", rs["preprocessing_dir"]),
            ("x_axis_length",     rs["x_axis_length"]),
            ("x_axis_min",        rs["x_axis_min"]),
            ("x_axis_max",        rs["x_axis_max"]),
        ]))
        out.append("")

        input_cfg = rs["input_config"]
        if input_cfg:
            out.append("\n### 1.3 Input configuration\n")
            out.append(self._kv_table(list(input_cfg.items())))
            out.append("")

        out.append("\n### 1.4 Checkpoint\n")
        out.append(self._kv_table([
            ("epoch",         ck["epoch"]),
            ("best_epoch",    ck["best_epoch"]),
            ("best_val_loss", ck["best_val_loss"]),
        ]))
        out.append("")

        out.append("\n### 1.5 Inference configuration\n")
        out.append(self._kv_table([
            ("device",             cfg["device"]),
            ("batch_size",         cfg["batch_size"]),
            ("num_workers",        cfg["num_workers"]),
            ("stitch_window",      cfg["stitch_window"]),
            ("cube_dtype",         cfg["cube_dtype"]),
            ("save_cubes",         cfg["save_cubes"]),
            ("n_best_profiles",    cfg["n_best_profiles"]),
            ("n_worst_profiles",   cfg["n_worst_profiles"]),
            ("n_random_profiles",  cfg["n_random_profiles"]),
            ("n_range_slices",     cfg["n_range_slices"]),
            ("n_azimuth_slices",   cfg["n_azimuth_slices"]),
            ("n_elevation_slices", cfg["n_elevation_slices"]),
            ("gif_axes",           cfg["gif_axes"]),
            ("gif_fps",            cfg["gif_fps"]),
            ("gif_max_frames",     cfg["gif_max_frames"]),
        ]))
        out.append("")

        return out

    def _build_headline_metrics(self) -> List[str]:
        gm  = self.global_metrics
        out = ["\n## 2. Headline metrics\n"]

        out.append("**Curve-level** (aggregated over the full stitched test cube)\n")
        out.append(self._three_col_table([
            ("MSE",     gm["curve_mse_gt"],   "Mean squared error over all (elev, az, rg)"),
            ("MAE",     gm["curve_mae_gt"],   "Mean absolute error"),
            ("RMSE",    gm["curve_rmse_gt"],  "Root mean squared error"),
            ("R\u00b2", gm["overall_r2_gt"],  "Global coefficient of determination"),
            ("PSNR dB", gm["psnr_db_gt"],     "Peak signal-to-noise ratio"),
        ]))
        out.append("")

        out.append("**Per-pixel** (computed per (az, rg) pixel over the elevation axis)\n")
        out.append(self._three_col_table([
            ("MSE mean",        gm["pixel_mse_gt_mean"],              "Mean of per-pixel MSE"),
            ("MSE median",      gm["pixel_mse_gt_median"],            "Median of per-pixel MSE"),
            ("MSE p95",         gm["pixel_mse_gt_p95"],               "95th percentile"),
            ("MAE mean",        gm["pixel_mae_gt_mean"],              "Mean of per-pixel MAE"),
            ("R\u00b2 mean",    gm["pixel_r2_gt_mean"],               "Mean of per-pixel R\u00b2"),
            ("R\u00b2 median",  gm["pixel_r2_gt_median"],             "Median of per-pixel R\u00b2"),
            ("R\u00b2 p5",      gm["pixel_r2_gt_p5"],                 "5th percentile (worst tail)"),
            ("Cosine mean",     gm["pixel_cosine_gt_mean"],           "Mean cosine similarity"),
            ("Cosine median",   gm["pixel_cosine_gt_median"],         "Median cosine similarity"),
            ("Peak err mean",   gm["pixel_peak_err_units_mean_gt"],   "|\u0394 peak| in elevation units"),
            ("Peak err median", gm["pixel_peak_err_units_median_gt"], "Median"),
            ("Peak err p95",    gm["pixel_peak_err_units_p95_gt"],    "95th percentile"),
        ]))
        out.append("")

        out.append("**SSIM** (mean over all slices of that axis)\n")
        out.append(self._three_col_table([
            ("SSIM elev mean",    gm["ssim_gt_elev_mean"],    "H\u00d7W intensity-at-elevation-bin planes"),
            ("SSIM range mean",   gm["ssim_gt_range_mean"],   "n_elev\u00d7H cross-sectional planes"),
            ("SSIM azimuth mean", gm["ssim_gt_azimuth_mean"], "n_elev\u00d7W cross-sectional planes"),
        ]))
        out.append("")

        out.append("**Per-elevation-bin** (mean over all elevation bins)\n")
        out.append(self._three_col_table([
            ("MAE mean",           gm["elev_mae_gt_mean"],   "Mean absolute error averaged over elevation"),
            ("RMSE mean",          gm["elev_rmse_gt_mean"],  "Root mean squared error averaged over elevation"),
            ("R\u00b2 mean",       gm["elev_r2_gt_mean"],    "Coefficient of determination averaged over elevation"),
            ("Cross-entropy mean", gm["elev_ce_gt_mean"],    "Cross-entropy (normalised profiles) averaged over elevation"),
        ]))
        out.append("")

        out += self._build_tracks_table()
        out += self._build_track_positions_table()
        out += self._build_baseline_comparison()

        return out

    def _build_tracks_table(self) -> List[str]:
        tracks = self.global_metrics.get("tracks")
        if not isinstance(tracks, dict):
            return []

        out = ["\n### 2.4 Tracks used in this run\n"]
        out.append(
            f"Baselines relative to the reference pass `{tracks['reference']}` over azimuth window "
            f"{tracks['azimuth_window']}; absolute values are the resa-frame windowed means.\n"
        )

        labels   = tracks["labels"]
        n_tracks = len(labels)

        table = MarkdownTable(("Pass", "Horizontal [m]", "Vertical [m]", "H std [m]", "V std [m]", "H absolute [m]", "V absolute [m]"))
        for row in zip(
            labels,
            self._padded_column(tracks, "horizontal",          n_tracks),
            self._padded_column(tracks, "vertical",            n_tracks),
            self._padded_column(tracks, "horizontal_std",      n_tracks),
            self._padded_column(tracks, "vertical_std",        n_tracks),
            self._padded_column(tracks, "horizontal_absolute", n_tracks),
            self._padded_column(tracks, "vertical_absolute",   n_tracks),
        ):
            table.add_row(row[0], *(self._fmt(value) for value in row[1:]))

        out.append("\n".join(table.render()))
        out.append("")

        return out

    def _build_track_positions_table(self) -> List[str]:
        positions = self.global_metrics.get("track_positions")
        if not isinstance(positions, dict):
            return []

        out = ["\n### 2.5 Track positions and temporal deviation\n"]
        out.append(
            f"Absolute mean positions in the resa track frame over {positions['n_samples']} azimuth samples "
            f"starting at index {positions['azimuth_start']}. Spans are peak-to-peak excursions per component; "
            f"the planar deviation is the distance of each sample from the per-pass mean position in the "
            f"horizontal-vertical plane, summarising how much the pass drifted over time.\n"
        )

        labels   = positions["labels"]
        n_tracks = len(labels)

        table = MarkdownTable(("Pass", "H mean [m]", "V mean [m]", "H span [m]", "V span [m]", "Planar dev RMS [m]", "Planar dev max [m]"))
        for row in zip(
            labels,
            self._padded_column(positions, "horizontal_mean", n_tracks),
            self._padded_column(positions, "vertical_mean",   n_tracks),
            self._padded_column(positions, "horizontal_span", n_tracks),
            self._padded_column(positions, "vertical_span",   n_tracks),
            self._padded_column(positions, "deviation_rms",   n_tracks),
            self._padded_column(positions, "deviation_max",   n_tracks),
        ):
            table.add_row(row[0], *(self._fmt(value) for value in row[1:]))

        out.append("\n".join(table.render()))
        out.append("")

        return out

    def _build_baseline_comparison(self) -> List[str]:
        gm = self.global_metrics
        if gm.get("curve_mse_red") is None:
            return []

        out = ["\n### 2.6 Classical Capon baseline (passes used in training)\n"]
        out.append(
            "The reduced tomogram is re-synthesised at inference time with classical Capon beamforming "
            "from exactly the passes this model was trained on, per-pixel max-normalised like the GT. "
            "Relative improvement is (baseline \u2212 model) / baseline for error metrics; for scores "
            "(R\u00b2, SSIM, cosine) the sign is flipped so positive always means the model is better.\n"
        )

        table = MarkdownTable(("Metric", "Pred vs GT", "Capon vs GT", "Model improvement"))

        rows = [
            ("Curve MSE",        "curve_mse_gt",             "curve_mse_red"),
            ("Curve MAE",        "curve_mae_gt",             "curve_mae_red"),
            ("Curve RMSE",       "curve_rmse_gt",            "curve_rmse_red"),
            ("Pixel MSE mean",   "pixel_mse_gt_mean",        "pixel_mse_red_mean"),
            ("Pixel MSE median", "pixel_mse_gt_median",      "pixel_mse_red_median"),
            ("Pixel MAE mean",   "pixel_mae_gt_mean",        "pixel_mae_red_mean"),
            ("Peak err mean",    "pixel_peak_idx_d_gt_mean", "pixel_peak_idx_d_red_mean"),
            ("Overall R\u00b2",       "overall_r2_gt",            "overall_r2_red"),
            ("Pixel R\u00b2 mean",    "pixel_r2_gt_mean",         "pixel_r2_red_mean"),
            ("Cosine mean",      "pixel_cosine_gt_mean",     "pixel_cosine_red_mean"),
            ("PSNR dB",          "psnr_db_gt",               "psnr_db_red"),
            ("SSIM elev mean",   "ssim_gt_elev_mean",        "ssim_red_elev_mean"),
            ("SSIM range mean",  "ssim_gt_range_mean",       "ssim_red_range_mean"),
            ("SSIM azimuth mean","ssim_gt_azimuth_mean",     "ssim_red_azimuth_mean"),
        ]

        for label, model_key, baseline_key in rows:
            model_val        = gm[model_key]
            baseline_val     = gm[baseline_key]
            higher_is_better = bool(MetricOrientation.higher_is_better(model_key))
            table.add_row(label, self._fmt(model_val), self._fmt(baseline_val), RelativeImprovement.percent(baseline_val, model_val, higher_is_better=higher_is_better))

        out.append("\n".join(table.render()))
        out.append("")

        out.append(self._kv_table([
            ("improvement_mse_rel",             gm["improvement_mse_rel"]),
            ("improvement_mae_rel",             gm["improvement_mae_rel"]),
            ("improvement_rmse_rel",            gm["improvement_rmse_rel"]),
            ("pixel_improvement_mean",          gm["pixel_improvement_mean"]),
            ("pixel_improvement_median",        gm["pixel_improvement_median"]),
            ("pixel_improvement_positive_frac", gm["pixel_improvement_positive_frac"]),
        ], header=("Improvement metric", "Value")))
        out.append("")

        return out

    def _build_full_metrics(self) -> List[str]:
        gm  = self.global_metrics
        out = ["\n## 3. Full metric tables\n"]

        groups: Dict[str, Dict] = {
            "3.1 Dataset statistics":             {},
            "3.2 Curve-level (GT)":               {},
            "3.3 Per-pixel (GT)":                 {},
            "3.4 SSIM summaries":                 {},
            "3.5 Capon baseline and improvement": {},
        }

        for k, v in sorted(gm.items()):
            if self._is_per_slice_ssim(k):
                continue
            if "_raw" in k or k in ("tracks", "track_positions"):
                continue
            if k in self._DATASET_KEYS:
                groups["3.1 Dataset statistics"][k] = v
            elif "_red" in k or k.startswith(("improvement_", "pixel_improvement_", "red_")):
                groups["3.5 Capon baseline and improvement"][k] = v
            elif k.startswith("pixel_"):
                groups["3.3 Per-pixel (GT)"][k] = v
            elif k.startswith("ssim_"):
                groups["3.4 SSIM summaries"][k] = v
            else:
                groups["3.2 Curve-level (GT)"][k] = v

        for title, d in groups.items():
            if not d:
                continue
            out.append(f"\n### {title}\n")
            out.append(self._dict_table(d))
            out.append("")

        return out

    def _section(self, out: List[str], groups) -> None:
        fp = self.figure_paths

        for key, title in groups:
            if fp.get(key):
                out.append(f"\n### {title}\n")
                out += self.assets.images(key, fp[key])

    def _numbered_section(self, out: List[str], prefix: str, groups) -> None:
        fp        = self.figure_paths
        available = [(key, title) for key, title in groups if fp.get(key)]

        for n, (key, title) in enumerate(available, start=1):
            out.append(f"\n### {prefix}{n} {title}\n")
            out += self.assets.images(key, fp[key])

    def _build_figures(self) -> List[str]:
        fp  = self.figure_paths
        gp  = self.gif_paths
        out = []

        track_groups = [(key, title) for key, title in (
            ("track_geometry",  "Baseline geometry of the passes used"),
            ("track_profiles",  "Per-azimuth baseline profiles"),
            ("track_flight_3d", "3D flight tracks with temporal deviation envelopes"),
            ("input_channels",  "Pass amplitudes and interferogram phases over the inference region"),
        ) if fp.get(key)]

        if track_groups:
            out.append("\n## 3b. Passes and interferograms used\n")
            out.append(
                "The model inputs for this run: the primary, the selected secondary passes and "
                "their interferograms against the primary.\n"
            )
            self._numbered_section(out, "3b.", track_groups)

        out.append("\n## 4. Profile reconstructions\n")
        out.append(
            "Each panel overlays the GT profile (black solid), "
            "prediction (red dashed), the classical Capon baseline when available (blue dotted) "
            "and individual Gaussian components. "
            "The shaded area shows the signed residual (pred \u2212 gt).\n"
        )
        self._section(out, (
            ("profiles_best",   "4.1 Best-fit profiles (lowest MSE)"),
            ("profiles_worst",  "4.2 Worst-fit profiles (highest MSE)"),
            ("profiles_random", "4.3 Random profiles"),
        ))

        out.append("\n## 5. Per-pixel metric maps\n")
        self._section(out, (
            ("pixel_mse_map",       "5.1 MSE map (log scale, pred vs GT)"),
            ("pixel_r2_map",        "5.2 R\u00b2 map (pred vs GT)"),
            ("pixel_peak_map",      "5.3 Peak-location error map (|\u0394 peak index|)"),
            ("pixel_mse_capon_map", "5.4 MSE map (log scale, Capon baseline vs GT)"),
            ("pixel_r2_capon_map",  "5.5 R\u00b2 map (Capon baseline vs GT)"),
            ("improvement_map",     "5.6 Relative MSE improvement over the Capon baseline"),
            ("metric_histograms",   "5.7 Metric distributions"),
        ))

        out.append("\n## 6. Gaussian parameter analysis\n")
        self._section(out, (
            ("param_maps",             "6.1 Parameter spatial maps (pred vs GT)"),
            ("param_distributions",    "6.2 Parameter distributions (GT vs Pred)"),
            ("param_scatter",          "6.3 Parameter scatter plots (GT vs Pred, with R²)"),
            ("param_error_maps",       "6.4 Parameter absolute-error maps |Pred − GT|"),
            ("slot_mu_distributions",  "6.5 Slot μ distributions (GT vs Pred per slot)"),
            ("placeholder_detection",  "6.6 Placeholder detection (precision / recall per slot)"),
            ("slot_ordering_summary",  "6.7 Slot ordering summary"),
            ("active_count_map",       "6.8 Active Gaussian count map"),
        ))

        slice_groups = [(key, title) for key, title in (
            ("slices_range",   "Range cuts"),
            ("slices_azimuth", "Azimuth cuts"),
            ("slices_elev",    "Elevation cuts"),
        ) if fp.get(key)]

        if slice_groups:
            out.append("\n## 7. Tomogram slices\n")
            out.append(
                "GT, prediction and (when available) the classical Capon baseline share a colour scale; "
                "error figures are clipped at p99 of that slice. "
                "SSIM (pred vs GT) is shown in the prediction title.\n"
            )
            self._numbered_section(out, "7.", slice_groups)

        out.append("\n## 8. SSIM curves\n")
        out.append("SSIM plotted for every slice along each axis \u2014 pred vs GT.\n")
        self._section(out, (
            ("ssim_range",   "8.1 SSIM along range axis"),
            ("ssim_azimuth", "8.2 SSIM along azimuth axis"),
            ("ssim_elev",    "8.3 SSIM along elevation axis"),
        ))

        if fp.get("elev_metric_curves"):
            out.append("\n### 8.4 Per-elevation-bin metrics (MAE, RMSE, R\u00b2, cross-entropy)\n")
            out.append(
                "Each panel shows a metric aggregated over all (az\u00d7rg) pixels for every "
                "elevation bin (pred vs GT). "
                "Dashed lines mark the mean over all bins.\n"
            )
            out += self.assets.images("elev_metric_curves", fp["elev_metric_curves"])

        if gp:
            out.append("\n## 9. Animations\n")
            out.append(
                "Each GIF walks through one axis of the stitched test cube. "
                "The colour scale is fixed across frames.\n"
            )
            for n, (name, path) in enumerate(sorted(gp.items()), start=1):
                out.append(f"\n### 9.{n} `{name}`\n")
                out += self.assets.image(name, path)

        return out

    def assemble(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.report_path

        lines: List[str] = []
        lines.append("# TomoSAR Inference Report")
        lines.append("")
        lines.append(f"_Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
        lines.append("")

        lines += self._build_run_summary()
        lines += self._build_headline_metrics()
        lines += self._build_full_metrics()
        lines += self._build_figures()

        if self.extra_sections:
            lines.append("\n## 10. Notes\n")
            for s in self.extra_sections:
                lines.append(s)
                lines.append("")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

