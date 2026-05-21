#!/usr/bin/env python3
"""
compare_trials.py
=================
Scans all experiment folders under logs/unet_trials/, picks the latest
inference run per trial, and writes three comparison reports:

    comparison_<timestamp>/
        training_comparison.md   – loss config, optimiser, best checkpoint
        test_results_comparison.md – metrics tables (means / medians only)
        gif_comparison.html      – side-by-side animated GIFs

Usage
-----
    python scripts/compare_trials.py
    python scripts/compare_trials.py --trials-dir logs/unet_trials
    python scripts/compare_trials.py --out-dir logs/comparisons
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parents[1]
TRIALS_DIR = REPO_ROOT / "logs" / "unet_trials"

# ---------------------------------------------------------------------------
# Metric keys to keep from metrics.json
# Only scalars: headline stats, means, medians, selected percentiles.
# The per-bin arrays (ssim_gt_elev_0 … ssim_gt_elev_149, elev_mae_gt_0 …)
# are intentionally excluded.
# ---------------------------------------------------------------------------
KEEP_KEYS: list[str] = [
    # ── dataset stats ────────────────────────────────────────────────────────
    "n_pixels", "n_elevation",
    "x_axis_min", "x_axis_max", "x_axis_step",
    "gt_mean", "gt_std", "gt_max",
    "pred_mean", "pred_std", "pred_max",
    # ── curve-level ──────────────────────────────────────────────────────────
    "curve_mse_gt",  "curve_mae_gt",  "curve_rmse_gt",
    "overall_r2_gt", "psnr_db_gt",
    # ── per-pixel MSE ────────────────────────────────────────────────────────
    "pixel_mse_gt_mean",   "pixel_mse_gt_median",
    "pixel_mse_gt_p5",     "pixel_mse_gt_p95",
    # ── per-pixel MAE ────────────────────────────────────────────────────────
    "pixel_mae_gt_mean",   "pixel_mae_gt_median",
    # ── per-pixel R² ────────────────────────────────────────────────────────
    "pixel_r2_gt_mean",    "pixel_r2_gt_median",
    "pixel_r2_gt_p5",      "pixel_r2_gt_p95",
    # ── cosine similarity ────────────────────────────────────────────────────
    "pixel_cosine_gt_mean",   "pixel_cosine_gt_median",
    "pixel_cosine_gt_p5",     "pixel_cosine_gt_p95",
    # ── peak location error ──────────────────────────────────────────────────
    "pixel_peak_idx_d_gt_mean",      "pixel_peak_idx_d_gt_median",
    "pixel_peak_idx_d_gt_p95",
    "pixel_peak_err_units_mean_gt",  "pixel_peak_err_units_median_gt",
    "pixel_peak_err_units_p95_gt",
    # ── SSIM summaries (mean over all slices per axis) ───────────────────────
    "ssim_gt_elev_mean",    "ssim_gt_range_mean",  "ssim_gt_azimuth_mean",
    # ── per-elevation-bin aggregates ─────────────────────────────────────────
    "elev_mae_gt_mean",  "elev_rmse_gt_mean",
    "elev_r2_gt_mean",   "elev_ce_gt_mean",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_inference(trial_dir: Path) -> Path | None:
    inf = trial_dir / "inference"
    if not inf.is_dir():
        return None
    dirs = sorted(
        [d for d in inf.iterdir() if d.is_dir() and re.match(r"\d{8}_\d{6}", d.name)]
    )
    return dirs[-1] if dirs else None


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _slim_metrics(raw: dict) -> dict:
    return {k: raw[k] for k in KEEP_KEYS if k in raw}


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _rel(target: Path, base: Path) -> str:
    """Relative POSIX path from *base directory* to *target*."""
    try:
        return target.relative_to(base).as_posix()
    except ValueError:
        # cross-drive or non-subpath → use absolute
        return target.as_posix()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_trials(trials_dir: Path) -> list[dict]:
    trials = []
    for td in sorted(trials_dir.iterdir()):
        if not td.is_dir() or td.name.startswith(".") or td.name.startswith("comparison"):
            continue

        inf_dir = _latest_inference(td)
        entry: dict = {
            "name":       td.name,
            "trial_dir":  td,
            "inf_dir":    inf_dir,
            "has_inf":    inf_dir is not None,
            "trainer_cfg": _load_json(td / "docs" / "trainer_config.json"),
            "run_summary": _load_json(td / "meta"  / "run_summary.json"),
            "metrics":    _slim_metrics(_load_json(inf_dir / "metrics.json")) if inf_dir else {},
        }
        trials.append(entry)
    return trials


# ---------------------------------------------------------------------------
# Report 1 – training_comparison.md
# ---------------------------------------------------------------------------

def _loss_row(cfg: dict) -> dict[str, str]:
    loss = cfg.get("loss", {})
    active = []
    for key, lbl in [
        ("use_mse_curve",          "MSE"),
        ("use_l1_curve",           "L1"),
        ("use_huber_curve",        "Huber"),
        ("use_charbonnier_curve",  "Charbonnier"),
        ("use_cosine_curve",       "Cosine"),
        ("use_spectral_coherence", "Spectral"),
        ("use_ssim_curve",         "SSIM"),
        ("use_param_l1",           "Param-L1"),
        ("use_param_huber",        "Param-Huber"),
        ("use_smoothness_tv",      "TV"),
    ]:
        if loss.get(key, False):
            w_key = key.replace("use_", "weight_").replace("_curve", "_curve").replace(
                "use_spectral_coherence", "weight_spectral_coh"
            ).replace("use_param_l1", "weight_param_l1").replace(
                "use_param_huber", "weight_param_huber"
            ).replace("use_smoothness_tv", "weight_smoothness_tv")
            w = loss.get(w_key, "?")
            active.append(f"{lbl}({w:g})" if isinstance(w, float) else f"{lbl}({w})")
    return ", ".join(active) if active else "—"


def write_training_comparison(trials: list[dict], out_dir: Path) -> Path:
    names = [t["name"] for t in trials]
    header  = "| Attribute | " + " | ".join(f"`{n}`" for n in names) + " |"
    divider = "| --- |" + " --- |" * len(names)

    def row(label: str, fn) -> str:
        cells = [_fmt(fn(t)) for t in trials]
        return "| " + label + " | " + " | ".join(cells) + " |"

    def cfg_val(t, *keys):
        d = t["trainer_cfg"]
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, "—")
            else:
                return "—"
        return d if d is not None else "—"

    def sum_val(t, *keys):
        d = t["run_summary"]
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, "—")
            else:
                return "—"
        return d if d is not None else "—"

    lines = [
        "# Training Comparison",
        f"\n_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
        "## Loss Configuration\n",
        header, divider,
        row("Active losses + weights", lambda t: _loss_row(t["trainer_cfg"])),
        row("Param match strategy",    lambda t: cfg_val(t, "loss", "param_match")),
        row("TV weight",               lambda t: cfg_val(t, "loss", "weight_smoothness_tv")),
        "",
        "## Optimiser & Scheduler\n",
        header, divider,
        row("Learning rate",  lambda t: cfg_val(t, "optimizer", "lr")),
        row("Scheduler type", lambda t: cfg_val(t, "scheduler", "type")),
        row("Max epochs",     lambda t: cfg_val(t, "scheduler", "epochs")),
        row("η_min",          lambda t: cfg_val(t, "scheduler", "eta_min")),
        row("Warmup",         lambda t: cfg_val(t, "warmup", "warmup_enabled")),
        row("Use EMA",        lambda t: cfg_val(t, "ema", "use_ema")),
        row("Use AMP",        lambda t: cfg_val(t, "training", "use_amp")),
        "",
        "## Best Checkpoint\n",
        header, divider,
        row("Best epoch",     lambda t: sum_val(t, "best_epoch")),
        row("Best val loss",  lambda t: sum_val(t, "best_val_loss")),
        row("Has inference",  lambda t: "✅" if t["has_inf"] else "⏳ pending"),
        row("Inference run",  lambda t: t["inf_dir"].name if t["inf_dir"] else "—"),
        "",
    ]

    out = out_dir / "training_comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Report 2 – test_results_comparison.md
# ---------------------------------------------------------------------------

METRIC_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Curve-level", [
        ("curve_mse_gt",  "MSE"),
        ("curve_mae_gt",  "MAE"),
        ("curve_rmse_gt", "RMSE"),
        ("overall_r2_gt", "R²"),
        ("psnr_db_gt",    "PSNR (dB)"),
    ]),
    ("SSIM (mean over slices)", [
        ("ssim_gt_elev_mean",    "SSIM elevation"),
        ("ssim_gt_range_mean",   "SSIM range"),
        ("ssim_gt_azimuth_mean", "SSIM azimuth"),
    ]),
    ("Per-pixel MSE", [
        ("pixel_mse_gt_mean",   "MSE mean"),
        ("pixel_mse_gt_median", "MSE median"),
        ("pixel_mse_gt_p5",     "MSE p5"),
        ("pixel_mse_gt_p95",    "MSE p95"),
    ]),
    ("Per-pixel R²", [
        ("pixel_r2_gt_mean",   "R² mean"),
        ("pixel_r2_gt_median", "R² median"),
        ("pixel_r2_gt_p5",     "R² p5"),
        ("pixel_r2_gt_p95",    "R² p95"),
    ]),
    ("Per-pixel Cosine Similarity", [
        ("pixel_cosine_gt_mean",   "Cosine mean"),
        ("pixel_cosine_gt_median", "Cosine median"),
        ("pixel_cosine_gt_p5",     "Cosine p5"),
        ("pixel_cosine_gt_p95",    "Cosine p95"),
    ]),
    ("Peak Location Error", [
        ("pixel_peak_err_units_mean_gt",   "Peak err mean (units)"),
        ("pixel_peak_err_units_median_gt", "Peak err median (units)"),
        ("pixel_peak_err_units_p95_gt",    "Peak err p95 (units)"),
        ("pixel_peak_idx_d_gt_mean",       "Peak Δidx mean"),
        ("pixel_peak_idx_d_gt_median",     "Peak Δidx median"),
        ("pixel_peak_idx_d_gt_p95",        "Peak Δidx p95"),
    ]),
    ("Per-elevation-bin aggregates", [
        ("elev_mae_gt_mean",  "MAE mean"),
        ("elev_rmse_gt_mean", "RMSE mean"),
        ("elev_r2_gt_mean",   "R² mean"),
        ("elev_ce_gt_mean",   "Cross-entropy mean"),
    ]),
]

FIGURE_SECTIONS: list[tuple[str, list[str]]] = [
    ("Profile reconstructions", [
        "profiles_best.png", "profiles_worst.png", "profiles_random.png",
    ]),
    ("Per-pixel metric maps", [
        "pixel_mse_map.png", "pixel_r2_map.png", "pixel_peak_map.png",
        "metric_histograms.png",
    ]),
    ("Gaussian parameter analysis", [
        "param_maps.png", "param_distributions.png",
        "param_scatter.png", "param_error_maps.png",
    ]),
    ("SSIM curves & elevation metrics", [
        "ssim_elev.png", "ssim_range.png", "ssim_azimuth.png",
        "elev_metric_curves.png",
    ]),
    ("Azimuth slices", [
        "slice_azimuth_12760.png", "slice_azimuth_13480.png",
        "slice_azimuth_14200.png", "slice_azimuth_14920.png",
        "slice_azimuth_15640.png",
    ]),
    ("Elevation slices", [
        "slice_elev_idx_15.png",  "slice_elev_idx_45.png",
        "slice_elev_idx_75.png",  "slice_elev_idx_105.png",
        "slice_elev_idx_135.png",
    ]),
    ("Range slices", [
        "slice_range_850.png",  "slice_range_1550.png", "slice_range_2250.png",
        "slice_range_2950.png", "slice_range_3650.png",
    ]),
]


def write_test_results_comparison(trials: list[dict], out_dir: Path) -> Path:
    inf_trials = [t for t in trials if t["has_inf"]]
    names = [t["name"] for t in inf_trials]

    header  = "| Metric | " + " | ".join(f"`{n}`" for n in names) + " |"
    divider = "| --- |" + " --- |" * len(names)

    def metric_row(label: str, key: str) -> str:
        cells = [_fmt(t["metrics"].get(key, "—")) for t in inf_trials]
        return "| " + label + " | " + " | ".join(cells) + " |"

    lines = [
        "# Test Results Comparison",
        f"\n_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
        "> Only trials that have at least one completed inference run are shown.\n",
    ]

    for section_title, metrics in METRIC_SECTIONS:
        lines += [f"## {section_title}\n", header, divider]
        for key, label in metrics:
            lines.append(metric_row(label, key))
        lines.append("")

    # ── Figure grids ─────────────────────────────────────────────────────────
    lines += ["---", "## Side-by-side Figures\n"]
    for section_title, figs in FIGURE_SECTIONS:
        lines.append(f"### {section_title}\n")
        for fig in figs:
            lines.append(f"**`{fig}`**\n")
            for t in inf_trials:
                fig_path = t["inf_dir"] / "figures" / fig
                if fig_path.exists():
                    rel = _rel(fig_path, out_dir)
                    lines.append(f"*{t['name']}*  \n![]({rel})\n")
                else:
                    lines.append(f"*{t['name']}* — _(not found)_\n")
            lines.append("")

    out = out_dir / "test_results_comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Report 3 – gif_comparison.html
# ---------------------------------------------------------------------------

GIF_AXES = ["walk_elevation", "walk_range", "walk_azimuth"]

_HTML_STYLE = """
<style>
  body  { font-family: sans-serif; background: #1a1a2e; color: #eee; margin: 0; padding: 16px; }
  h1    { text-align: center; margin-bottom: 8px; }
  p.ts  { text-align: center; color: #aaa; font-size: .85em; margin-top: 0; }
  h2    { border-bottom: 1px solid #444; padding-bottom: 4px; margin-top: 32px; }
  .grid { display: grid; gap: 12px; margin-top: 8px; }
  .cell { background: #16213e; border: 1px solid #333; border-radius: 6px;
          padding: 8px; text-align: center; }
  .cell img  { width: 100%; border-radius: 4px; }
  .cell span { display: block; font-size: .8em; color: #aaa; margin-top: 4px;
               word-break: break-all; }
</style>
"""


def write_gif_comparison(trials: list[dict], out_dir: Path) -> Path:
    inf_trials = [t for t in trials if t["has_inf"]]
    n = len(inf_trials)
    if n == 0:
        html = "<html><body><p>No inference results available yet.</p></body></html>"
        out  = out_dir / "gif_comparison.html"
        out.write_text(html, encoding="utf-8")
        return out

    cols_css = " ".join(["1fr"] * n)

    rows_html = ""
    for axis in GIF_AXES:
        fname = f"{axis}.gif"
        cells = ""
        for t in inf_trials:
            gif_path = t["inf_dir"] / "animations" / fname
            if gif_path.exists():
                rel  = _rel(gif_path, out_dir)
                cells += (
                    f'<div class="cell">'
                    f'<img src="{rel}" alt="{axis}" loading="lazy">'
                    f'<span>{t["name"]}</span>'
                    f'</div>\n'
                )
            else:
                cells += (
                    f'<div class="cell">'
                    f'<em style="color:#888">not found</em>'
                    f'<span>{t["name"]}</span>'
                    f'</div>\n'
                )
        rows_html += (
            f'<h2>{axis.replace("_", " ").title()}</h2>'
            f'<div class="grid" style="grid-template-columns:{cols_css};">'
            f'{cells}</div>\n'
        )

    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>GIF Comparison – {ts}</title>{_HTML_STYLE}</head>"
        f"<body><h1>GIF Comparison</h1><p class='ts'>Generated {ts}</p>"
        f"{rows_html}</body></html>"
    )

    out = out_dir / "gif_comparison.html"
    out.write_text(html, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TomoSAR unet trials.")
    parser.add_argument(
        "--trials-dir", type=Path, default=TRIALS_DIR,
        help="Root directory containing trial sub-folders."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Where to write comparison reports (default: <trials-dir>/comparison_<ts>)."
    )
    args = parser.parse_args()

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (args.trials_dir / f"comparison_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compare_trials] Scanning  → {args.trials_dir}")
    print(f"[compare_trials] Output    → {out_dir}\n")

    trials = collect_trials(args.trials_dir)

    # Summary
    for t in trials:
        status = f"✅ {t['inf_dir'].name}" if t["has_inf"] else "⏳ no inference yet"
        print(f"  {t['name']:<45} {status}")

    print()

    r1 = write_training_comparison(trials, out_dir)
    r2 = write_test_results_comparison(trials, out_dir)
    r3 = write_gif_comparison(trials, out_dir)

    print(f"\n[compare_trials] Reports written:")
    for r in (r1, r2, r3):
        print(f"  {r}")


if __name__ == "__main__":
    main()
