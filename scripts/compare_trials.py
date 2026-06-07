from __future__ import annotations

import argparse
import base64
import json
import os
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


def _parse_best_from_log(trial_dir: Path) -> dict:
    """Extract best epoch and best val loss from the metadata log file."""
    log_dir = trial_dir / "logs"
    logs = list(log_dir.glob("*.log")) if log_dir.is_dir() else []
    if not logs:
        return {}
    log_text = logs[0].read_text(encoding="utf-8", errors="ignore")
    # "Early stopping triggered at epoch NNN. Best epoch was NNN."
    es_match = re.search(r"Best epoch was (\d+)", log_text)
    # "best=0.0364 @ epoch 100" — take the last occurrence
    best_matches = re.findall(r"best=([\d.eE+\-]+)\s*@\s*epoch\s*(\d+)", log_text)
    result = {}
    if es_match:
        result["best_epoch"] = int(es_match.group(1))
    elif best_matches:
        result["best_epoch"] = int(best_matches[-1][1])
    if best_matches:
        result["best_val_loss"] = float(best_matches[-1][0])
    return result


def _rel(target: Path, base: Path) -> str:
    """Relative POSIX path from *base directory* to *target*."""
    return Path(os.path.relpath(target.resolve(), base.resolve())).as_posix()


_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}


def _img_src(path: Path, embed: bool, out_dir: Path) -> str:
    """Return either a base64 data URI (embed=True) or a relative path."""
    if embed and path.exists():
        mime = _MIME.get(path.suffix.lower(), "image/png")
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"
    return _rel(path, out_dir)


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
            "log_best":   _parse_best_from_log(td),
            "metrics":    _slim_metrics(_load_json(inf_dir / "metrics.json")) if inf_dir else {},
        }
        trials.append(entry)
    return trials


# ---------------------------------------------------------------------------
# Report 1 – training_comparison.md
# ---------------------------------------------------------------------------

def _loss_cfg(cfg: dict) -> dict:
    cur = cfg.get("curriculum", {})
    if cur.get("complete"):
        return cur["complete"]
    return cur.get("warmup", {})


def _loss_row(cfg: dict) -> dict[str, str]:
    loss = _loss_cfg(cfg)
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
            w_key = (
                key
                .replace("use_spectral_coherence", "weight_spectral_coh")
                .replace("use_param_l1",            "weight_param_l1")
                .replace("use_param_huber",          "weight_param_huber")
                .replace("use_smoothness_tv",        "weight_smoothness_tv")
                .replace("use_",                     "weight_")
            )
            w = loss.get(w_key, "?")
            active.append(f"{lbl}({w:g})" if isinstance(w, float) else f"{lbl}({w})")
    return ", ".join(active) if active else "—"


def write_training_comparison(trials: list[dict], out_dir: Path) -> Path:

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

    # Each column is one attribute; each row is one trial.
    LOSS_COLS: list[tuple[str, callable]] = [
        ("Active losses + weights", lambda t: _loss_row(t["trainer_cfg"])),
        ("Param match",             lambda t: _loss_cfg(t["trainer_cfg"]).get("param_match", "—")),
        ("TV weight",               lambda t: _loss_cfg(t["trainer_cfg"]).get("weight_smoothness_tv", "—")),
    ]
    OPT_COLS: list[tuple[str, callable]] = [
        ("LR",             lambda t: cfg_val(t, "optimizer", "lr")),
        ("Scheduler",      lambda t: cfg_val(t, "scheduler", "type")),
        ("Max epochs",     lambda t: cfg_val(t, "scheduler", "epochs")),
        ("η_min",          lambda t: cfg_val(t, "scheduler", "eta_min")),
        ("Warmup",         lambda t: cfg_val(t, "warmup", "warmup_enabled")),
        ("EMA",            lambda t: cfg_val(t, "ema", "use_ema")),
        ("AMP",            lambda t: cfg_val(t, "training", "use_amp")),
    ]
    CKPT_COLS: list[tuple[str, callable]] = [
        ("Best epoch",    lambda t: t["log_best"].get("best_epoch",    sum_val(t, "best_epoch"))),
        ("Best val loss", lambda t: t["log_best"].get("best_val_loss", sum_val(t, "best_val_loss"))),
        ("Has inf",       lambda t: "yes" if t["has_inf"] else "pending"),
        ("Inference run", lambda t: t["inf_dir"].name if t["inf_dir"] else "—"),
    ]

    def make_table(cols: list[tuple[str, callable]]) -> list[str]:
        col_labels = [c[0] for c in cols]
        header  = "| Trial | " + " | ".join(col_labels) + " |"
        divider = "| --- |" + " --- |" * len(cols)
        rows = []
        for t in trials:
            cells = [_fmt(fn(t)) for _, fn in cols]
            rows.append("| `" + t["name"] + "` | " + " | ".join(cells) + " |")
        return [header, divider, *rows, ""]

    lines = [
        "# Training Comparison",
        f"\n_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
        "## Loss Configuration\n",
        *make_table(LOSS_COLS),
        "## Optimiser & Scheduler\n",
        *make_table(OPT_COLS),
        "## Best Checkpoint\n",
        *make_table(CKPT_COLS),
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


def write_test_results_comparison(trials: list[dict], out_dir: Path, embed: bool = False) -> list[Path]:
    inf_trials = [t for t in trials if t["has_inf"]]

    def make_metrics_table(metrics: list[tuple[str, str]]) -> list[str]:
        """One row per trial, one column per metric."""
        col_labels = [label for _, label in metrics]
        header  = "| Trial | " + " | ".join(col_labels) + " |"
        divider = "| --- |" + " --- |" * len(metrics)
        rows = []
        for t in inf_trials:
            cells = [_fmt(t["metrics"].get(key, "—")) for key, _ in metrics]
            rows.append("| `" + t["name"] + "` | " + " | ".join(cells) + " |")
        return [header, divider, *rows, ""]

    # ── File 1: numerical metrics only ───────────────────────────────────────
    lines = [
        "# Test Results – Numerical Metrics",
        f"\n_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
        "> Only trials that have at least one completed inference run are shown.\n",
    ]
    for section_title, metrics in METRIC_SECTIONS:
        lines += [f"## {section_title}\n", *make_metrics_table(metrics)]

    metrics_out = out_dir / "test_results_metrics.md"
    metrics_out.write_text("\n".join(lines), encoding="utf-8")
    written = [metrics_out]

    # ── One file per figure section ───────────────────────────────────────────
    ts_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for section_title, figs in FIGURE_SECTIONS:
        slug = re.sub(r"[^a-z0-9]+", "_", section_title.lower()).strip("_")
        lines = [
            f"# Figures – {section_title}",
            f"\n_Generated {ts_str}_\n",
            "> Only trials that have at least one completed inference run are shown.\n",
        ]
        for fig in figs:
            lines.append(f"## `{fig}`\n")
            for t in inf_trials:
                fig_path = t["inf_dir"] / "figures" / fig
                if fig_path.exists():
                    src = _img_src(fig_path, embed, out_dir)
                    lines.append(f"*{t['name']}*  \n![]({src})\n")
                else:
                    lines.append(f"*{t['name']}* — _(not found)_\n")
            lines.append("")

        fig_out = out_dir / f"figures_{slug}.md"
        fig_out.write_text("\n".join(lines), encoding="utf-8")
        written.append(fig_out)

    return written


# ---------------------------------------------------------------------------
# Report 3 – gif_comparison.html
# ---------------------------------------------------------------------------

GIF_AXES = ["walk_elevation", "walk_range", "walk_azimuth"]


def write_gif_comparison(trials: list[dict], out_dir: Path, embed: bool = False) -> Path:
    inf_trials = [t for t in trials if t["has_inf"]]
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# GIF Comparison",
        f"\n_Generated {ts_str}_\n",
        "> Only trials that have at least one completed inference run are shown.\n",
    ]

    if not inf_trials:
        lines.append("_No inference results available yet._\n")
    else:
        for axis in GIF_AXES:
            fname = f"{axis}.gif"
            lines.append(f"## {axis.replace('_', ' ').title()}\n")
            for t in inf_trials:
                gif_path = t["inf_dir"] / "animations" / fname
                if gif_path.exists():
                    src = _img_src(gif_path, embed, out_dir)
                    lines.append(f"*{t['name']}*  \n![]({src})\n")
                else:
                    lines.append(f"*{t['name']}* — _(not found)_\n")
            lines.append("")

    out = out_dir / "gif_comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
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
    parser.add_argument(
        "--embed", action="store_true",
        help="Base64-embed all images/GIFs into the output files (portable, no external deps)."
    )
    args = parser.parse_args()

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (args.trials_dir / f"comparison_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compare_trials] Scanning  → {args.trials_dir}")
    print(f"[compare_trials] Output    → {out_dir}")
    if args.embed:
        print("[compare_trials] Mode      → embedded (portable, base64 images)\n")
    else:
        print("[compare_trials] Mode      → linked (relative paths)\n")

    trials = collect_trials(args.trials_dir)

    # Summary
    for t in trials:
        status = f"{t['inf_dir'].name}" if t["has_inf"] else "no inference yet"
        print(f"  {t['name']:<45} {status}")

    print()

    r1 = write_training_comparison(trials, out_dir)
    r2 = write_test_results_comparison(trials, out_dir, embed=args.embed)
    r3 = write_gif_comparison(trials, out_dir, embed=args.embed)

    print(f"\n[compare_trials] Reports written:")
    for r in [r1, *r2, r3]:
        print(f"  {r}")


if __name__ == "__main__":
    main()
