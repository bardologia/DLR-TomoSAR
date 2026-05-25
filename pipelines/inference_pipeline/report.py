from __future__ import annotations

import json
from datetime import datetime
from pathlib  import Path
from typing   import Any, Dict, List, Optional, Tuple


def write_metrics_json(metrics: Dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, default=str)
    return path


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
        extra_sections   : Optional[List[str]] = None,
    ) -> None:

        self.output_dir       = Path(output_dir)
        self.run_summary      = run_summary
        self.inference_config = inference_config
        self.checkpoint_meta  = checkpoint_meta
        self.global_metrics   = global_metrics
        self.figure_paths     = figure_paths
        self.gif_paths        = gif_paths
        self.extra_sections   = extra_sections or []

    @staticmethod
    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            if abs(v) >= 1e4 or (0 < abs(v) < 1e-3):
                return f"{v:.4e}"
            return f"{v:.6g}"
        if isinstance(v, (list, tuple)):
            return ", ".join(str(x) for x in v)
        return str(v)

    @staticmethod
    def _kv_table(rows: List[Tuple[str, Any]], header: Tuple[str, str] = ("Key", "Value")) -> str:
        out = [f"| {header[0]} | {header[1]} |", "| --- | --- |"]
        for k, v in rows:
            out.append(f"| `{k}` | {Report._fmt(v)} |")
        return "\n".join(out)

    @staticmethod
    def _dict_table(d: Dict[str, Any]) -> str:
        return Report._kv_table(sorted(d.items()))

    @staticmethod
    def _three_col_table(
        rows   : List[Tuple[str, Any, str]],
        header : Tuple[str, str, str] = ("Metric", "Pred vs GT", "Description"),
    ) -> str:
        out = [f"| {header[0]} | {header[1]} | {header[2]} |", "| --- | --- | --- |"]
        for label, gt_val, desc in rows:
            out.append(f"| {label} | {Report._fmt(gt_val)} | {desc} |")
        return "\n".join(out)

    def _rel(self, p: Path) -> str:
        try:
            return str(Path(p).resolve().relative_to(self.output_dir.resolve()))
        except ValueError:
            return str(p)

    def _img(self, key: str, path: Path) -> List[str]:
        return [f"![{key}]({self._rel(path)})", ""]

    @staticmethod
    def _is_per_slice_ssim(k: str) -> bool:
        for prefix in ("ssim_gt_elev_", "ssim_gt_range_", "ssim_gt_azimuth_"):
            if k.startswith(prefix) and k[len(prefix):].lstrip("-").isdigit():
                return True
        for prefix in (
            "elev_mae_gt_", "elev_rmse_gt_", "elev_r2_gt_", "elev_ce_gt_",
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
            ("model_name",     rs.get("model_name")),
            ("in_channels",    rs.get("in_channels")),
            ("out_channels",   rs.get("out_channels")),
            ("n_gaussians",    rs.get("n_gaussians")),
            ("has_noise_head", rs.get("has_noise_head")),
            ("used_ema",       rs.get("used_ema")),
        ]))
        out.append("")

        out.append("\n### 1.2 Dataset & patch grid\n")
        out.append(self._kv_table([
            ("split",             rs.get("split")),
            ("split_region",      rs.get("split_region")),
            ("global_crop",       rs.get("global_crop")),
            ("patches",           rs.get("patches")),
            ("patch_size",        rs.get("patch_size")),
            ("patch_stride",      rs.get("patch_stride")),
            ("preprocessing_dir", rs.get("preprocessing_dir")),
            ("x_axis_length",     rs.get("x_axis_length")),
            ("x_axis_min",        rs.get("x_axis_min")),
            ("x_axis_max",        rs.get("x_axis_max")),
        ]))
        out.append("")

        input_cfg = rs.get("input_config", {})
        if isinstance(input_cfg, dict) and input_cfg:
            out.append("\n### 1.3 Input configuration\n")
            out.append(self._kv_table(list(input_cfg.items())))
            out.append("")

        out.append("\n### 1.4 Checkpoint\n")
        out.append(self._kv_table([
            ("epoch",         ck.get("epoch")),
            ("best_epoch",    ck.get("best_epoch")),
            ("best_val_loss", ck.get("best_val_loss")),
        ]))
        out.append("")

        out.append("\n### 1.5 Inference configuration\n")
        out.append(self._kv_table([
            ("device",             cfg.get("device")),
            ("batch_size",         cfg.get("batch_size")),
            ("num_workers",        cfg.get("num_workers")),
            ("stitch_window",      cfg.get("stitch_window")),
            ("cube_dtype",         cfg.get("cube_dtype")),
            ("save_cubes",         cfg.get("save_cubes")),
            ("n_best_profiles",    cfg.get("n_best_profiles")),
            ("n_worst_profiles",   cfg.get("n_worst_profiles")),
            ("n_random_profiles",  cfg.get("n_random_profiles")),
            ("n_range_slices",     cfg.get("n_range_slices")),
            ("n_azimuth_slices",   cfg.get("n_azimuth_slices")),
            ("n_elevation_slices", cfg.get("n_elevation_slices")),
            ("gif_axes",           cfg.get("gif_axes")),
            ("gif_fps",            cfg.get("gif_fps")),
            ("gif_max_frames",     cfg.get("gif_max_frames")),
        ]))
        out.append("")

        return out

    def _build_headline_metrics(self) -> List[str]:
        gm  = self.global_metrics
        out = ["\n## 2. Headline metrics\n"]

        out.append("**Curve-level** (aggregated over the full stitched test cube)\n")
        out.append(self._three_col_table([
            ("MSE",     gm.get("curve_mse_gt"),   "Mean squared error over all (elev, az, rg)"),
            ("MAE",     gm.get("curve_mae_gt"),   "Mean absolute error"),
            ("RMSE",    gm.get("curve_rmse_gt"),  "Root mean squared error"),
            ("R\u00b2", gm.get("overall_r2_gt"),  "Global coefficient of determination"),
            ("PSNR dB", gm.get("psnr_db_gt"),     "Peak signal-to-noise ratio"),
        ]))
        out.append("")

        out.append("**Per-pixel** (computed per (az, rg) pixel over the elevation axis)\n")
        out.append(self._three_col_table([
            ("MSE mean",        gm.get("pixel_mse_gt_mean"),              "Mean of per-pixel MSE"),
            ("MSE median",      gm.get("pixel_mse_gt_median"),            "Median of per-pixel MSE"),
            ("MSE p95",         gm.get("pixel_mse_gt_p95"),               "95th percentile"),
            ("MAE mean",        gm.get("pixel_mae_gt_mean"),              "Mean of per-pixel MAE"),
            ("R\u00b2 mean",    gm.get("pixel_r2_gt_mean"),               "Mean of per-pixel R\u00b2"),
            ("R\u00b2 median",  gm.get("pixel_r2_gt_median"),             "Median of per-pixel R\u00b2"),
            ("R\u00b2 p5",      gm.get("pixel_r2_gt_p5"),                 "5th percentile (worst tail)"),
            ("Cosine mean",     gm.get("pixel_cosine_gt_mean"),           "Mean cosine similarity"),
            ("Cosine median",   gm.get("pixel_cosine_gt_median"),         "Median cosine similarity"),
            ("Peak err mean",   gm.get("pixel_peak_err_units_mean_gt"),   "|\u0394 peak| in elevation units"),
            ("Peak err median", gm.get("pixel_peak_err_units_median_gt"), "Median"),
            ("Peak err p95",    gm.get("pixel_peak_err_units_p95_gt"),    "95th percentile"),
        ]))
        out.append("")

        out.append("**SSIM** (mean over all slices of that axis)\n")
        out.append(self._three_col_table([
            ("SSIM elev mean",    gm.get("ssim_gt_elev_mean"),    "H\u00d7W intensity-at-elevation-bin planes"),
            ("SSIM range mean",   gm.get("ssim_gt_range_mean"),   "n_elev\u00d7H cross-sectional planes"),
            ("SSIM azimuth mean", gm.get("ssim_gt_azimuth_mean"), "n_elev\u00d7W cross-sectional planes"),
        ]))
        out.append("")

        out.append("**Per-elevation-bin** (mean over all elevation bins)\n")
        out.append(self._three_col_table([
            ("MAE mean",           gm.get("elev_mae_gt_mean"),   "Mean absolute error averaged over elevation"),
            ("RMSE mean",          gm.get("elev_rmse_gt_mean"),  "Root mean squared error averaged over elevation"),
            ("R\u00b2 mean",       gm.get("elev_r2_gt_mean"),    "Coefficient of determination averaged over elevation"),
            ("Cross-entropy mean", gm.get("elev_ce_gt_mean"),    "Cross-entropy (normalised profiles) averaged over elevation"),
        ]))
        out.append("")

        return out

    def _build_full_metrics(self) -> List[str]:
        gm  = self.global_metrics
        out = ["\n## 3. Full metric tables\n"]

        groups: Dict[str, Dict] = {
            "3.1 Dataset statistics": {},
            "3.2 Curve-level (GT)":   {},
            "3.3 Per-pixel (GT)":     {},
            "3.4 SSIM summaries":     {},
        }

        for k, v in sorted(gm.items()):
            if self._is_per_slice_ssim(k):
                continue
            if "_raw" in k:
                continue
            if k in self._DATASET_KEYS:
                groups["3.1 Dataset statistics"][k] = v
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

    def _build_figures(self) -> List[str]:
        fp  = self.figure_paths
        gp  = self.gif_paths
        out = []

        out.append("\n## 4. Profile reconstructions\n")
        out.append(
            "Each panel overlays the GT profile (black solid), "
            "prediction (red dashed) and individual Gaussian components. "
            "The shaded area shows the signed residual (pred \u2212 gt).\n"
        )
        for key, title in (
            ("profiles_best",   "4.1 Best-fit profiles (lowest MSE)"),
            ("profiles_worst",  "4.2 Worst-fit profiles (highest MSE)"),
            ("profiles_random", "4.3 Random profiles"),
        ):
            if key in fp:
                out.append(f"\n### {title}\n")
                out += self._img(key, fp[key])

        out.append("\n## 5. Per-pixel metric maps\n")
        for key, title in (
            ("pixel_mse_map",     "5.1 MSE map (log scale, pred vs GT)"),
            ("pixel_r2_map",      "5.2 R\u00b2 map (pred vs GT)"),
            ("pixel_peak_map",    "5.3 Peak-location error map (|\u0394 peak index|)"),
            ("metric_histograms", "5.4 Metric distributions"),
        ):
            if key in fp:
                out.append(f"\n### {title}\n")
                out += self._img(key, fp[key])

        out.append("\n## 6. Gaussian parameter analysis\n")
        for key, title in (
            ("param_maps",             "6.1 Parameter spatial maps (pred vs GT)"),
            ("param_distributions",    "6.2 Parameter distributions (GT vs Pred)"),
            ("param_scatter",          "6.3 Parameter scatter plots (GT vs Pred, with R²)"),
            ("param_error_maps",       "6.4 Parameter absolute-error maps |Pred − GT|"),
            ("slot_mu_distributions",  "6.5 Slot μ distributions (GT vs Pred per slot)"),
            ("placeholder_detection",  "6.6 Placeholder detection (precision / recall per slot)"),
            ("slot_ordering_summary",  "6.7 Slot ordering summary"),
            ("active_count_map",       "6.8 Active Gaussian count map"),
        ):
            if key in fp:
                out.append(f"\n### {title}\n")
                out += self._img(key, fp[key])

        slice_figs = {k: v for k, v in fp.items() if k.startswith("slice_")}
        if slice_figs:
            out.append("\n## 7. Tomogram slices\n")
            out.append(
                "GT and prediction share a colour scale; the error panel is clipped at p99 of that slice. "
                "SSIM (pred vs GT) is shown in the title.\n"
            )
            for n, k in enumerate(sorted(slice_figs), start=1):
                out.append(f"\n### 7.{n} `{k}`\n")
                out += self._img(k, slice_figs[k])

        out.append("\n## 8. SSIM curves\n")
        out.append("SSIM plotted for every slice along each axis \u2014 pred vs GT.\n")
        for key, title in (
            ("ssim_range",   "8.1 SSIM along range axis"),
            ("ssim_azimuth", "8.2 SSIM along azimuth axis"),
            ("ssim_elev",    "8.3 SSIM along elevation axis"),
        ):
            if key in fp:
                out.append(f"\n### {title}\n")
                out += self._img(key, fp[key])

        if "elev_metric_curves" in fp:
            out.append("\n### 8.4 Per-elevation-bin metrics (MAE, RMSE, R\u00b2, cross-entropy)\n")
            out.append(
                "Each panel shows a metric aggregated over all (az\u00d7rg) pixels for every "
                "elevation bin (pred vs GT). "
                "Dashed lines mark the mean over all bins.\n"
            )
            out += self._img("elev_metric_curves", fp["elev_metric_curves"])

        if gp:
            out.append("\n## 9. Animations\n")
            out.append(
                "Each GIF walks through one axis of the stitched test cube. "
                "The colour scale is fixed across frames.\n"
            )
            for n, (name, path) in enumerate(sorted(gp.items()), start=1):
                out.append(f"\n### 9.{n} `{name}`\n")
                out += self._img(name, path)

        return out

    def assemble(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "report.md"

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

