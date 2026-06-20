from __future__ import annotations

from datetime import datetime
from pathlib  import Path
from typing   import Any, Dict, List, Optional, Tuple

import numpy as np

from tools.reporting.reporting import ReportAssets
from tools.reporting.markdown  import MarkdownTable, ScalarFormatter


class ReportPayloadBuilder:
    @staticmethod
    def run_summary(run, x_axis_np: np.ndarray) -> Dict[str, Any]:
        return {
            "model_name"        : run.backbone_name,
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
            "save_plots"         : cfg.save_plots,
            "save_animations"    : cfg.save_animations,
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
    def _gt_pred_table(
        rows   : List[Tuple[str, Any, Any, str]],
        header : Tuple[str, str, str, str] = ("Metric", "GT", "Pred", "Description"),
    ) -> str:
        table = MarkdownTable(header)
        for label, gt_val, pred_val, desc in rows:
            table.add_row(label, Report._fmt(gt_val), Report._fmt(pred_val), desc)
        return "\n".join(table.render())

    @staticmethod
    def _padded_column(payload: Dict[str, Any], key: str, n_tracks: int) -> list:
        values = payload[key]
        return values if len(values) == n_tracks else [float("nan")] * n_tracks

    @staticmethod
    def _is_per_slice_ssim(k: str) -> bool:
        for prefix in (
            "ssim_gt_elev_",   "ssim_gt_range_",   "ssim_gt_azimuth_",
            "ssim_norm_elev_", "ssim_norm_range_", "ssim_norm_azimuth_",
            "ssim_red_elev_",  "ssim_red_range_",  "ssim_red_azimuth_",
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
            ("save_plots",         cfg["save_plots"]),
            ("save_animations",    cfg["save_animations"]),
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

        out.append("**SSIM** (denorm, mean over all slices of that axis)\n")
        out.append(self._three_col_table([
            ("SSIM elev mean",    gm["ssim_gt_elev_mean"],    "H\u00d7W intensity-at-elevation-bin planes"),
            ("SSIM range mean",   gm["ssim_gt_range_mean"],   "n_elev\u00d7H cross-sectional planes"),
            ("SSIM azimuth mean", gm["ssim_gt_azimuth_mean"], "n_elev\u00d7W cross-sectional planes"),
        ]))
        out.append("")

        out.append("**SSIM** (unit-area normalised pred vs GT, mean over all slices of that axis)\n")
        out.append(self._three_col_table([
            ("SSIM elev mean",    gm["ssim_norm_elev_mean"],    "H\u00d7W intensity-at-elevation-bin planes"),
            ("SSIM range mean",   gm["ssim_norm_range_mean"],   "n_elev\u00d7H cross-sectional planes"),
            ("SSIM azimuth mean", gm["ssim_norm_azimuth_mean"], "n_elev\u00d7W cross-sectional planes"),
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

        out += self._build_active_count_headline()
        out += self._build_matched_headline()
        out += self._build_tracks_table()
        out += self._build_track_positions_table()
        out += self._build_reduced_headline()

        return out

    def _build_matched_headline(self) -> List[str]:
        gm = self.global_metrics
        if "matched_mu_mae" not in gm:
            return []

        n_K = self.run_summary["n_gaussians"]
        out = ["\n### 2.5 Permutation-invariant matched Gaussian errors\n"]
        out.append(
            "Each pixel's predicted Gaussians are Hungarian-matched to its GT Gaussians on |Δμ| before "
            "scoring, so the errors are independent of slot ordering. A match counts as a detection hit when "
            f"|Δμ| ≤ {self._fmt(gm['matched_tol'])} elevation units.\n"
        )
        out.append(self._three_col_table([
            ("Matched μ MAE",   gm["matched_mu_mae"],    "Mean |Δμ| over matched pairs (all counts)"),
            ("Matched μ RMSE",  gm["matched_mu_rmse"],   "Root mean squared Δμ"),
            ("Matched σ MAE",   gm["matched_sig_mae"],   "Mean |Δσ| over matched pairs"),
            ("Detection recall",     gm["matched_recall"],    "Share of GT Gaussians matched within tolerance"),
            ("Detection precision",  gm["matched_precision"], "Share of predicted Gaussians matching a GT within tolerance"),
            ("Detection F1",         gm["matched_f1"],        "Harmonic mean of recall and precision"),
        ], header=("Metric", "Value", "Description")))
        out.append("")

        present = [k for k in range(1, n_K + 1) if f"matched_recall_gt{k}" in gm or f"matched_mu_mae_gt{k}" in gm]

        if present:
            out.append("**By GT count** (recall = recovery; precision = filler-free; μ MAE = placement)\n")
            table = MarkdownTable(("GT count", "Recall ↑", "Precision ↑", "Matched μ MAE ↓"))
            for k in present:
                table.add_row(
                    str(k),
                    self._fmt(gm.get(f"matched_recall_gt{k}",    float("nan"))),
                    self._fmt(gm.get(f"matched_precision_gt{k}", float("nan"))),
                    self._fmt(gm.get(f"matched_mu_mae_gt{k}",    float("nan"))),
                )
            out += table.render()
            out.append("")

        return out

    def _build_active_count_headline(self) -> List[str]:
        gm = self.global_metrics
        if "active_frac_gt" not in gm:
            return []

        n_K = self.run_summary["n_gaussians"]
        out = ["\n### 2.4 Slot occupancy (GT vs Pred)\n"]
        out.append(
            "Per-slot and overall fraction of pixels carrying an active Gaussian (amplitude ≥ 1e-3), "
            "ground truth versus prediction, over the full stitched test cube.\n"
        )

        rows = [
            ("Active fraction (all slots)", gm["active_frac_gt"],       gm["active_frac_pred"],       "Share of slot-pixels that are active"),
            ("Mean active count / pixel",   gm["active_count_gt_mean"], gm["active_count_pred_mean"], "Average number of active Gaussians per pixel"),
        ]
        for k in range(n_K):
            rows.append((f"Slot {k} active fraction", gm[f"slot_{k}_active_gt_frac"], gm[f"slot_{k}_active_pred_frac"], "Active-pixel fraction for this slot"))

        out.append(self._gt_pred_table(rows))
        out.append("")

        out.append("**Active-count agreement** (predicted vs GT number of active Gaussians per pixel)\n")
        agree_rows = [
            ("Exact count", gm["count_exact_frac"], "Pixels where pred count == GT count"),
            ("Undercount",  gm["count_under_frac"], "Pixels where pred count < GT count"),
            ("Overcount",   gm["count_over_frac"],  "Pixels where pred count > GT count"),
        ]
        for k in range(1, n_K + 1):
            key = f"count_acc_gt{k}"
            if key in gm:
                agree_rows.append((f"Accuracy | GT count = {k}", gm[key], f"Fraction correct among pixels whose GT has {k} active"))

        out.append(self._three_col_table(agree_rows, header=("Metric", "Fraction", "Description")))
        out.append("")

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

    def _build_reduced_headline(self) -> List[str]:
        gm = self.global_metrics
        if "improvement_pixel_mse_mean" not in gm:
            return []

        out = ["\n### 2.6 NN improvement over the classical baseline\n"]
        out.append(
            "The reduced tomogram is a classical Capon reconstruction from the primary plus the exact "
            "secondary subset this run was trained on, processed identically to the ground-truth full-stack "
            "tomogram (same pyrat `fusartomo`, height range, filter and stack). It is what tomography yields "
            "without the network. Reduced-vs-GT and improvement quantities are computed on per-elevation-profile "
            "unit-area-normalised cubes, so they measure elevation-distribution agreement independent of absolute "
            "Capon power calibration.\n"
        )
        out.append(self._three_col_table([
            ("Reduced MSE mean",       gm["reduced_pixel_mse_norm_mean"],  "Per-pixel reduced-vs-GT MSE (classical baseline)"),
            ("Pred MSE mean",          gm["pred_pixel_mse_norm_mean"],     "Per-pixel pred-vs-GT MSE (neural network)"),
            ("Improvement mean",       gm["improvement_pixel_mse_mean"],   "MSE(reduced) − MSE(pred), higher is better"),
            ("Improvement median",     gm["improvement_pixel_mse_median"], "Median per-pixel improvement"),
            ("Relative MSE reduction", gm["relative_mse_reduction"],       "1 − MSE(pred)/MSE(reduced)"),
            ("Pixels where NN wins",   gm["fraction_pred_beats_reduced"],  "Fraction with pred MSE < reduced MSE"),
        ], header=("Metric", "Value", "Description")))
        out.append("")

        out.append("**Reduced vs GT** (classical baseline, unit-area profiles)\n")
        out.append(self._three_col_table([
            ("Curve MSE",         gm["curve_mse_red"],         "Mean squared error over all (elev, az, rg)"),
            ("Overall R2",        gm["overall_r2_red"],        "Global coefficient of determination"),
            ("SSIM elev mean",    gm["ssim_red_elev_mean"],    "Intensity-at-elevation-bin planes"),
            ("SSIM range mean",   gm["ssim_red_range_mean"],   "Cross-sectional planes"),
            ("SSIM azimuth mean", gm["ssim_red_azimuth_mean"], "Cross-sectional planes"),
        ], header=("Metric", "Reduced vs GT", "Description")))
        out.append("")

        return out

    _IMPROVEMENT_KEYS = frozenset({
        "improvement_pixel_mse_mean", "improvement_pixel_mse_median",
        "fraction_pred_beats_reduced", "relative_mse_reduction",
        "pred_pixel_mse_norm_mean", "reduced_pixel_mse_norm_mean",
    })

    @classmethod
    def _is_improvement_key(cls, k: str) -> bool:
        return k in cls._IMPROVEMENT_KEYS

    @staticmethod
    def _is_reduced_key(k: str) -> bool:
        return ("_red" in k) or k.startswith("ssim_red") or ("predn" in k)

    @staticmethod
    def _is_occupancy_key(k: str) -> bool:
        return k.startswith("active_frac") or k.startswith("active_count") or k.startswith("count_") or (k.startswith("slot_") and "_active_" in k)

    def _build_full_metrics(self) -> List[str]:
        gm  = self.global_metrics
        out = ["\n## 3. Full metric tables\n"]

        groups: Dict[str, Dict] = {
            "3.1 Dataset statistics":             {},
            "3.2 Curve-level (GT)":               {},
            "3.3 Per-pixel (GT)":                 {},
            "3.4 SSIM summaries (denorm)":        {},
            "3.4b SSIM summaries (normalised)":   {},
            "3.5 Curve-level (reduced vs GT)":    {},
            "3.6 Per-pixel (reduced vs GT)":      {},
            "3.7 SSIM summaries (reduced vs GT)": {},
            "3.8 NN improvement over baseline":   {},
            "3.9 Slot occupancy & active count":  {},
            "3.10 Matched Gaussian errors (permutation-invariant)": {},
        }

        for k, v in sorted(gm.items()):
            if self._is_per_slice_ssim(k):
                continue
            if "_raw" in k or k in ("tracks", "track_positions", "split", "split_region"):
                continue
            if k in self._DATASET_KEYS:
                groups["3.1 Dataset statistics"][k] = v
            elif k.startswith("matched_"):
                groups["3.10 Matched Gaussian errors (permutation-invariant)"][k] = v
            elif self._is_occupancy_key(k):
                groups["3.9 Slot occupancy & active count"][k] = v
            elif self._is_improvement_key(k):
                groups["3.8 NN improvement over baseline"][k] = v
            elif self._is_reduced_key(k):
                if k.startswith("ssim_red"):
                    groups["3.7 SSIM summaries (reduced vs GT)"][k] = v
                elif k.startswith("pixel_"):
                    groups["3.6 Per-pixel (reduced vs GT)"][k] = v
                else:
                    groups["3.5 Curve-level (reduced vs GT)"][k] = v
            elif k.startswith("pixel_"):
                groups["3.3 Per-pixel (GT)"][k] = v
            elif k.startswith("ssim_norm"):
                groups["3.4b SSIM summaries (normalised)"][k] = v
            elif k.startswith("ssim_"):
                groups["3.4 SSIM summaries (denorm)"][k] = v
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
            "prediction (red dashed) and individual Gaussian components. "
            "The shaded area shows the signed residual (pred \u2212 gt). "
            "The raw full-stack tomogram that the GT Gaussians are fit to is drawn as a grey reference. "
            "When a reduced baseline is available, its classical-Capon profile is overlaid "
            "(green dotted, rescaled to the GT peak) so the network's gain in elevation shape is visible.\n"
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
            ("metric_histograms",   "5.4 Metric distributions"),
        ))

        out.append("\n## 6. Gaussian parameter analysis\n")
        out.append(
            "Caveat: figures 6.1, 6.2 and 6.5–6.7 compare predicted slot k against GT slot k, so they assume the "
            "canonical (μ-sorted) slot alignment and are only meaningful for sort-matched models — for a "
            "Hungarian-matched run they show the slot relabelling, not the error. The parameter scatter (6.3), "
            "parameter error maps (6.4) and active-count map (6.8) are permutation-invariant: the scatter and "
            "error maps Hungarian-match predicted Gaussians to GT Gaussians per pixel before scoring, so they are "
            "valid for both matching strategies. For aggregate ordering-independent accuracy see §2.5.\n"
        )
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
            ("slices_range",        "Range cuts"),
            ("slices_azimuth",      "Azimuth cuts"),
            ("slices_elev",         "Elevation cuts"),
            ("slices_range_norm",   "Range cuts (unit-area)"),
            ("slices_azimuth_norm", "Azimuth cuts (unit-area)"),
            ("slices_elev_norm",    "Elevation cuts (unit-area)"),
        ) if fp.get(key)]

        if slice_groups:
            out.append("\n## 7. Tomogram slices\n")
            out.append(
                "GT and prediction share a colour scale; "
                "error figures are clipped at p99 of that slice. "
                "SSIM (pred vs GT) is shown in the prediction title. "
                "The raw full-stack tomogram (the reference the GT Gaussians are fit to) is shown as an extra panel. "
                "Both denormalised slices and unit-area-normalised slices (each elevation profile normalised to unit area) are shown.\n"
            )
            self._numbered_section(out, "7.", slice_groups)

        out.append("\n## 8. SSIM curves\n")
        out.append("SSIM plotted for every slice along each axis \u2014 pred vs GT, both denormalised and unit-area-normalised.\n")
        self._section(out, (
            ("ssim_range",        "8.1 SSIM along range axis (denorm)"),
            ("ssim_azimuth",      "8.2 SSIM along azimuth axis (denorm)"),
            ("ssim_elev",         "8.3 SSIM along elevation axis (denorm)"),
            ("ssim_range_norm",   "8.4 SSIM along range axis (unit-area)"),
            ("ssim_azimuth_norm", "8.5 SSIM along azimuth axis (unit-area)"),
            ("ssim_elev_norm",    "8.6 SSIM along elevation axis (unit-area)"),
        ))

        if fp.get("elev_metric_curves"):
            out.append("\n### 8.7 Per-elevation-bin metrics (MAE, RMSE, R\u00b2, cross-entropy)\n")
            out.append(
                "Each panel shows a metric aggregated over all (az\u00d7rg) pixels for every "
                "elevation bin (pred vs GT). "
                "Dashed lines mark the mean over all bins.\n"
            )
            out += self.assets.images("elev_metric_curves", fp["elev_metric_curves"])

        out += self._build_reduced_figures()

        if gp:
            out.append("\n## 10. Animations\n")
            out.append(
                "Each GIF walks through one axis of the stitched test cube. "
                "The colour scale is fixed across frames.\n"
            )
            for n, (name, path) in enumerate(sorted(gp.items()), start=1):
                out.append(f"\n### 10.{n} `{name}`\n")
                out += self.assets.image(name, path)

        return out

    def _build_reduced_figures(self) -> List[str]:
        fp = self.figure_paths

        keys = (
            "improvement_map", "reduced_pixel_mse_map",
            "slices_range_reduced", "slices_azimuth_reduced", "slices_elev_reduced",
            "ssim_range_reduced", "ssim_azimuth_reduced", "ssim_elev_reduced",
            "elev_metric_curves_reduced",
        )
        if not any(fp.get(k) for k in keys):
            return []

        out = ["\n## 9. Classical baseline (reduced Capon) vs ground truth\n"]
        out.append(
            "The reduced tomogram is a classical Capon reconstruction from the primary plus the exact "
            "secondary subset this run was trained on — what tomography yields without the network. "
            "Reduced and GT cubes are unit-area-normalised per elevation profile before comparison, so "
            "panels and metrics measure elevation-distribution agreement. Read the network's gain by "
            "contrasting these errors with the pred-vs-GT figures in sections 5, 7 and 8.\n"
        )

        self._section(out, (
            ("improvement_map",       "9.1 Per-pixel NN improvement map (MSE reduced − MSE pred)"),
            ("reduced_pixel_mse_map", "9.2 Per-pixel reduced-vs-GT MSE map"),
        ))

        slice_groups = [(key, title) for key, title in (
            ("slices_range_reduced",   "Range cuts (GT, reduced, error)"),
            ("slices_azimuth_reduced", "Azimuth cuts (GT, reduced, error)"),
            ("slices_elev_reduced",    "Elevation cuts (GT, reduced, error)"),
        ) if fp.get(key)]

        if slice_groups:
            out.append("\n### 9.3 Reduced tomogram slices\n")
            self._numbered_section(out, "9.3.", slice_groups)

        self._section(out, (
            ("ssim_range_reduced",   "9.4 Reduced SSIM along range axis"),
            ("ssim_azimuth_reduced", "9.5 Reduced SSIM along azimuth axis"),
            ("ssim_elev_reduced",    "9.6 Reduced SSIM along elevation axis"),
        ))

        if fp.get("elev_metric_curves_reduced"):
            out.append("\n### 9.7 Reduced per-elevation-bin metrics (MAE, RMSE, R², cross-entropy)\n")
            out += self.assets.images("elev_metric_curves_reduced", fp["elev_metric_curves_reduced"])

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
            lines.append("\n## 11. Notes\n")
            for s in self.extra_sections:
                lines.append(s)
                lines.append("")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

