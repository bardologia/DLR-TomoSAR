from __future__ import annotations

import json
from datetime import datetime
from pathlib  import Path
from typing   import Any, Dict, List, Optional, Tuple


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if abs(v) >= 1e4 or (0 < abs(v) < 1e-3):
            return f"{v:.4e}"
        return f"{v:.6g}"
    return str(v)


def _table(d: Dict[str, object], cols: Tuple[str, str] = ("metric", "value")) -> str:
    lines = [f"| {cols[0]} | {cols[1]} |", "|---|---|"]
    for k in sorted(d.keys()):
        lines.append(f"| `{k}` | {_fmt(d[k])} |")
    return "\n".join(lines)


def write_metrics_json(metrics: Dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, default=str)
    return path


def assemble_report(
    output_dir       : Path,
    run_summary      : Dict,
    inference_config : Dict,
    checkpoint_meta  : Dict,
    global_metrics   : Dict,
    figure_paths     : Dict[str, Path],
    gif_paths        : Dict[str, Path],
    extra_sections   : Optional[List[str]] = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"

    def _rel(p: Path) -> str:
        try:
            return str(Path(p).resolve().relative_to(output_dir.resolve()))
        except ValueError:
            return str(p)

    lines: List[str] = []
    lines.append("# TomoSAR Inference Report")
    lines.append("")
    lines.append(f"_Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")

    lines.append("## 1. Run summary")
    lines.append("")
    lines.append(_table(run_summary))
    lines.append("")
    lines.append("### Checkpoint")
    lines.append("")
    lines.append(_table(checkpoint_meta))
    lines.append("")
    lines.append("### Inference configuration")
    lines.append("")
    lines.append(_table(inference_config))
    lines.append("")

    lines.append("## 2. Headline metrics")
    lines.append("")
    headline_keys = [
        "n_pixels", "n_elevation",
        "curve_mse", "curve_mae", "curve_rmse", "overall_r2", "psnr_db",
        "pixel_mse_mean", "pixel_mse_median", "pixel_mse_p95",
        "pixel_r2_mean", "pixel_r2_median", "pixel_r2_p5",
        "pixel_cosine_mean", "pixel_cosine_median",
        "pixel_peak_err_units_mean", "pixel_peak_err_units_median", "pixel_peak_err_units_p95",
        "ssim_elev_mean", "ssim_range_mean", "ssim_azimuth_mean",
    ]
    headline = {k: global_metrics[k] for k in headline_keys if k in global_metrics}
    lines.append(_table(headline))
    lines.append("")
    lines.append("### Definitions")
    lines.append("")
    lines.append("- **curve_mse / mae / rmse** : aggregated over every (elevation, az, rg) sample of the stitched test cube.")
    lines.append("- **overall_r²** : 1 − Σ(pred − gt)² / Σ(gt − mean(gt))²  with the sums spanning the entire test cube.")
    lines.append("- **psnr_db** : 10·log10(range(gt)² / curve_mse).")
    lines.append("- **pixel_mse / mae / r² / cosine** : computed *per pixel* (i.e. over the elevation axis only) and then aggregated over (az, rg).")
    lines.append("- **pixel_peak_err_units** : |argmax(pred) − argmax(gt)| converted to elevation units using the model's `x_axis_step`.")
    lines.append("- All cubes are reassembled from overlapping patches with a Hann blending window so seams do not bias the metrics.")
    lines.append("- **ssim_elev / range / azimuth** : Structural Similarity Index (SSIM) computed on each 2-D tomogram slice. `ssim_elev_<i>` is a H×W intensity-at-elevation-bin plane; `ssim_range_<i>` / `ssim_azimuth_<i>` are n_elev×H or n_elev×W cross-sectional planes. `_mean` averages over all sampled slices of that axis. `data_range` is set per-slice as `max(gt) − min(gt)`.")

    lines.append("")

    lines.append("## 3. Full metric table")
    lines.append("")
    lines.append(_table(global_metrics))
    lines.append("")

    section_titles = {
        "profiles_best"     : "4.1 Best-fit pixel profiles",
        "profiles_worst"    : "4.2 Worst-fit pixel profiles",
        "profiles_random"   : "4.3 Random pixel profiles",
        "pixel_mse_map"     : "5.1 Per-pixel MSE map",
        "pixel_r2_map"      : "5.2 Per-pixel R² map",
        "pixel_peak_map"    : "5.3 Peak-location error map",
        "metric_histograms" : "5.4 Metric distributions",
        "param_maps"        : "5.5 Gaussian parameter maps",
    }

    lines.append("## 4. Profile reconstructions")
    lines.append("")
    lines.append("Each panel overlays the ground-truth elevation profile (black solid), the model prediction (red dashed) and the individual Gaussian components used by the prediction. The shaded region shows the signed residual (pred − gt).")
    lines.append("")
    for key in ("profiles_best", "profiles_worst", "profiles_random"):
        if key in figure_paths:
            lines.append(f"### {section_titles[key]}")
            lines.append("")
            lines.append(f"![{key}]({_rel(figure_paths[key])})")
            lines.append("")

    lines.append("## 5. Spatial diagnostics")
    lines.append("")
    for key in ("pixel_mse_map", "pixel_r2_map", "pixel_peak_map", "metric_histograms", "param_maps"):
        if key in figure_paths:
            lines.append(f"### {section_titles[key]}")
            lines.append("")
            lines.append(f"![{key}]({_rel(figure_paths[key])})")
            lines.append("")

    slice_figs = {k: v for k, v in figure_paths.items() if k.startswith("slice_")}
    if slice_figs:
        lines.append("## 6. Tomogram slices")
        lines.append("")
        lines.append("Each figure shows three panels with a *shared intensity colour scale* across GT and prediction so any visual difference is genuine; the right-most panel uses an absolute-error colourmap clipped at the 99th percentile of the slice.")
        lines.append("")
        for k in sorted(slice_figs):
            lines.append(f"### 6.x `{k}`")
            lines.append("")
            lines.append(f"![{k}]({_rel(slice_figs[k])})")
            lines.append("")

    if gif_paths:
        lines.append("## 7. Animations")
        lines.append("")
        lines.append("These GIFs walk through one axis of the stitched test cube while keeping the colour scale fixed across frames, so changes reflect the real signal evolution and not per-frame contrast normalisation.")
        lines.append("")
        for name, path in gif_paths.items():
            lines.append(f"### 7.x `{name}`")
            lines.append("")
            lines.append(f"![{name}]({_rel(path)})")
            lines.append("")

    if extra_sections:
        lines.append("## 8. Notes")
        lines.append("")
        for s in extra_sections:
            lines.append(s)
            lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
