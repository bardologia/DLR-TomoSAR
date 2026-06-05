from __future__ import annotations

import base64
import gc
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from tools.logger import Logger

_TOTAL_PARAMS_PATTERN = re.compile(r"\*\*Total Parameters:\*\*\s*`([\d,]+)`")
_CHECKPOINT_KEYS      = ("best_val_loss", "best_epoch", "epoch", "global_step")

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}

_PER_BIN_PATTERN  = re.compile(r"_\d+$")
_NEUTRAL_PATTERN  = re.compile(r"^(n_pixels|n_elevation|x_axis_|gt_|pred_|slot_\d+_mu_)|_n_valid$|placeholder_(gt|pred)_rate$")
_HIGHER_TOKENS    = ("r2", "ssim", "psnr", "cosine", "precision", "recall", "f1", "consensus", "ordering")

_METRIC_SECTIONS = [
    ("Dataset Statistics",           re.compile(r"^(n_pixels|n_elevation|x_axis_|gt_|pred_)")),
    ("Curve-Level",                  re.compile(r"^(curve_|overall_r2|psnr_)")),
    ("SSIM",                         re.compile(r"^ssim_")),
    ("Per-Pixel MSE and MAE",        re.compile(r"^pixel_(mse|mae)_")),
    ("Per-Pixel R² and Cosine",      re.compile(r"^pixel_(r2|cosine)_")),
    ("Peak Location Error",          re.compile(r"^pixel_peak_")),
    ("Per-Elevation-Bin Aggregates", re.compile(r"^elev_")),
    ("Gaussian Parameter Errors",    re.compile(r"^gauss_")),
    ("Slot Statistics",              re.compile(r"^slot_")),
    ("Placeholder Detection",        re.compile(r"^placeholder_")),
    ("Permutation and Ordering",     re.compile(r"^(permutation_|mu_ordering)")),
]

_FIGURE_GROUPS = [
    ("Profile reconstructions",     re.compile(r"^profiles_")),
    ("Per-pixel metric maps",       re.compile(r"^(pixel_|metric_histograms)")),
    ("Gaussian parameter analysis", re.compile(r"^param_")),
    ("Slot diagnostics",            re.compile(r"^(slot_|placeholder_|active_count)")),
    ("SSIM and elevation curves",   re.compile(r"^(ssim_|elev_metric)")),
    ("Azimuth slices",              re.compile(r"^slice_azimuth_")),
    ("Elevation slices",            re.compile(r"^slice_elev_")),
    ("Range slices",                re.compile(r"^slice_range_")),
]

_HEADLINE_METRICS = [
    ("curve_rmse_gt",                "RMSE"),
    ("curve_mae_gt",                 "MAE"),
    ("overall_r2_gt",                "R²"),
    ("psnr_db_gt",                   "PSNR"),
    ("pixel_r2_gt_mean",             "Pixel R²"),
    ("pixel_cosine_gt_mean",         "Cosine"),
    ("ssim_gt_elev_mean",            "SSIM elev"),
    ("pixel_peak_err_units_mean_gt", "Peak err"),
]


@dataclass
class TrialRecord:
    name            : str
    run_dir         : Path
    parameters      : int | None  = None
    size_match      : dict        = field(default_factory=dict)
    trainer_config  : dict        = field(default_factory=dict)
    run_summary     : dict        = field(default_factory=dict)
    checkpoint      : dict        = field(default_factory=dict)
    overfit         : dict        = field(default_factory=dict)
    training_result : dict        = field(default_factory=dict)
    inference_dir   : Path | None = None
    metrics         : dict        = field(default_factory=dict)
    figures         : list[Path]  = field(default_factory=list)
    animations      : list[Path]  = field(default_factory=list)
    report_path     : Path | None = None

    @property
    def has_inference(self) -> bool:
        return self.inference_dir is not None


class TrialCollector:
    def __init__(self, run_dir: Path, logger: Logger) -> None:
        self.run_dir      = run_dir
        self.training_dir = run_dir / "training"
        self.pipeline_dir = run_dir / "pipeline"
        self.logger       = logger

    def collect(self) -> list[TrialRecord]:
        size_match       = self._load_json(self.pipeline_dir / "size_match.json") or {}
        overfit_results  = {r.get("model"): r for r in self._load_json(self.pipeline_dir / "overfit_results.json") or []}
        training_results = {r.get("name"):  r for r in self._load_json(self.pipeline_dir / "training_results.json") or []}

        if not self.training_dir.is_dir():
            self.logger.error(f"No training directory found at: {self.training_dir}")
            return []

        records = []
        for trial_dir in sorted(d for d in self.training_dir.iterdir() if d.is_dir()):
            record = TrialRecord(name=trial_dir.name, run_dir=trial_dir)

            record.size_match      = size_match.get(trial_dir.name, {})
            record.trainer_config  = self._load_json(trial_dir / "docs" / "trainer_config.json") or {}
            record.run_summary     = self._load_json(trial_dir / "meta" / "run_summary.json") or {}
            record.overfit         = overfit_results.get(trial_dir.name, {})
            record.training_result = training_results.get(trial_dir.name, {})
            record.parameters      = self._parse_parameters(trial_dir, record.size_match)
            record.checkpoint      = self._read_checkpoint(trial_dir)

            self._attach_inference(record)

            status = f"inference {record.inference_dir.name}" if record.has_inference else "no inference"
            self.logger.info(f"{record.name:<22} {status}")

            records.append(record)

        return records

    def _load_json(self, path: Path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _parse_parameters(self, trial_dir: Path, size_match: dict) -> int | None:
        summary_path = trial_dir / "docs" / "model_summary.md"

        if summary_path.exists():
            match = _TOTAL_PARAMS_PATTERN.search(summary_path.read_text(encoding="utf-8", errors="ignore"))
            if match:
                return int(match.group(1).replace(",", ""))

        return size_match.get("parameters")

    def _read_checkpoint(self, trial_dir: Path) -> dict:
        import torch

        checkpoint_path = next(trial_dir.rglob("best_model.pt"), None)
        if checkpoint_path is None:
            return {}

        try:
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception:
            return {}

        info = {key: checkpoint.get(key) for key in _CHECKPOINT_KEYS}
        info["n_train_epochs"] = len(checkpoint.get("train_losses") or [])
        info["n_val_epochs"]   = len(checkpoint.get("val_losses") or [])

        del checkpoint
        gc.collect()

        return {key: value for key, value in info.items() if value is not None}

    def _attach_inference(self, record: TrialRecord) -> None:
        inference_root = record.run_dir / "inference"
        if not inference_root.is_dir():
            return

        candidates = sorted(d for d in inference_root.iterdir() if d.is_dir() and (d / "metrics.json").exists())
        if not candidates:
            return

        inference_dir = candidates[-1]

        record.inference_dir = inference_dir
        record.metrics       = self._load_json(inference_dir / "metrics.json") or {}
        record.figures       = sorted((inference_dir / "figures").glob("*.png")) if (inference_dir / "figures").is_dir() else []
        record.animations    = sorted((inference_dir / "animations").glob("*.gif")) if (inference_dir / "animations").is_dir() else []

        report_path = inference_dir / "report.md"
        if report_path.exists():
            record.report_path = report_path


class ComparisonReport:
    def __init__(self, records: list[TrialRecord], out_dir: Path, reference_model: str, embed_images: bool, logger: Logger) -> None:
        self.records         = records
        self.out_dir         = out_dir
        self.reference_model = reference_model
        self.embed_images    = embed_images
        self.logger          = logger
        self.timestamp       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def write_all(self) -> list[Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        written = [self._write_overview(), self._write_metrics()]
        written.extend(self._write_figures())
        written.append(self._write_gifs())
        written.append(self._write_summary_json())

        return written

    def _header(self, title: str) -> list[str]:
        return [f"# {title}", f"\n_Generated {self.timestamp}_\n"]

    def _fmt(self, value) -> str:
        if isinstance(value, float):
            return f"{value:.5g}"
        if value is None:
            return "—"
        return str(value)

    def _natural_key(self, name: str) -> list:
        return [int(token) if token.isdigit() else token for token in re.split(r"(\d+)", name)]

    def _rel(self, target: Path) -> str:
        return Path(os.path.relpath(target.resolve(), self.out_dir.resolve())).as_posix()

    def _img_src(self, path: Path) -> str:
        if self.embed_images and path.exists():
            mime = _MIME.get(path.suffix.lower(), "image/png")
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{data}"
        return self._rel(path)

    def _direction(self, key: str) -> str | None:
        if _NEUTRAL_PATTERN.search(key):
            return None
        if any(token in key for token in _HIGHER_TOKENS):
            return "higher"
        return "lower"

    def _write_overview(self) -> Path:
        lines = self._header("Benchmark Overview")

        lines += ["## Model Capacity\n"]
        lines += [f"Reference model: `{self.reference_model}`.\n"]
        lines += ["| Model | Parameters | Δ vs reference | Width scale | Scaled attributes |", "| --- | --- | --- | --- | --- |"]
        for r in self.records:
            parameters = f"{r.parameters:,}" if r.parameters is not None else "—"
            deviation  = f"{r.size_match['deviation_pct']:+.3f} %" if "deviation_pct" in r.size_match else "—"
            scale      = f"{r.size_match['scale']:.4f}" if "scale" in r.size_match else "—"
            overrides  = ", ".join(f"`{k}={v}`" for k, v in r.size_match.get("overrides", {}).items()) or "_(defaults)_"
            lines.append(f"| `{r.name}` | {parameters} | {deviation} | {scale} | {overrides} |")
        lines.append("")

        lines += ["## Overfit Gate\n"]
        lines += ["| Model | Status | Final loss | Converged |", "| --- | --- | --- | --- |"]
        for r in self.records:
            final_loss = f"{r.overfit['final_loss']:.4e}" if r.overfit.get("final_loss") is not None else "—"
            converged  = {True: "yes", False: "no"}.get(r.overfit.get("converged"), "—")
            lines.append(f"| `{r.name}` | {r.overfit.get('status', '—')} | {final_loss} | {converged} |")
        lines.append("")

        lines += ["## Training\n"]
        lines += ["| Model | Status | Best epoch | Best val loss | Epochs run | Duration |", "| --- | --- | --- | --- | --- | --- |"]
        for r in self.records:
            duration_s = r.training_result.get("duration_s")
            duration   = f"{duration_s / 60:.1f} min" if duration_s is not None else "—"
            lines.append(
                f"| `{r.name}` | {r.training_result.get('status', '—')} | {self._fmt(r.checkpoint.get('best_epoch'))} "
                f"| {self._fmt(r.checkpoint.get('best_val_loss'))} | {self._fmt(r.checkpoint.get('n_train_epochs'))} | {duration} |"
            )
        lines.append("")

        lines += ["## Inference\n"]
        lines += ["| Model | Inference run | Figures | Animations | Report |", "| --- | --- | --- | --- | --- |"]
        for r in self.records:
            inference  = f"`{r.inference_dir.name}`" if r.has_inference else "pending"
            report_md  = f"[report.md]({self._rel(r.report_path)})" if r.report_path else "—"
            lines.append(f"| `{r.name}` | {inference} | {len(r.figures)} | {len(r.animations)} | {report_md} |")
        lines.append("")

        lines += self._leaderboard()

        out = self.out_dir / "benchmark_overview.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def _leaderboard(self) -> list[str]:
        scored = [r for r in self.records if r.metrics]
        if not scored:
            return ["## Leaderboard\n", "_No inference metrics available yet._\n"]

        ranks : dict[str, dict[str, int]] = {r.name: {} for r in scored}

        for key, _ in _HEADLINE_METRICS:
            valued  = [(r.name, r.metrics[key]) for r in scored if isinstance(r.metrics.get(key), (int, float))]
            reverse = self._direction(key) == "higher"
            ordered = sorted(valued, key=lambda item: item[1], reverse=reverse)

            for position, (name, _) in enumerate(ordered, start=1):
                ranks[name][key] = position

        worst = len(scored) + 1
        mean_ranks = {
            name: sum(ranks[name].get(key, worst) for key, _ in _HEADLINE_METRICS) / len(_HEADLINE_METRICS)
            for name in ranks
        }

        lines = ["## Leaderboard\n", "Mean rank across the headline metrics (1 = best); missing metrics rank last.\n"]
        lines += ["| Rank | Model | Mean rank | " + " | ".join(label for _, label in _HEADLINE_METRICS) + " |"]
        lines += ["| --- | --- | --- |" + " --- |" * len(_HEADLINE_METRICS)]

        for position, name in enumerate(sorted(mean_ranks, key=mean_ranks.get), start=1):
            cells = " | ".join(str(ranks[name].get(key, "—")) for key, _ in _HEADLINE_METRICS)
            lines.append(f"| {position} | `{name}` | {mean_ranks[name]:.2f} | {cells} |")

        lines.append("")
        return lines

    def _write_metrics(self) -> Path:
        scored = [r for r in self.records if r.metrics]

        lines = self._header("Test Metrics Comparison")
        lines += ["> Best value per metric in **bold** (↓ lower is better, ↑ higher is better). Per-bin array metrics are excluded.\n"]

        if not scored:
            lines += ["_No inference metrics available yet._\n"]
        else:
            all_keys = sorted({key for r in scored for key, value in r.metrics.items() if isinstance(value, (int, float)) and not _PER_BIN_PATTERN.search(key)})
            claimed  : set[str] = set()

            for title, pattern in _METRIC_SECTIONS:
                keys = [key for key in all_keys if key not in claimed and pattern.search(key)]
                if not keys:
                    continue
                claimed.update(keys)
                lines += [f"## {title}\n", *self._metric_table(keys, scored)]

            leftover = [key for key in all_keys if key not in claimed]
            if leftover:
                lines += ["## Other Metrics\n", *self._metric_table(leftover, scored)]

        out = self.out_dir / "metrics_comparison.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def _metric_table(self, keys: list[str], scored: list[TrialRecord]) -> list[str]:
        header  = "| Metric | " + " | ".join(f"`{r.name}`" for r in scored) + " |"
        divider = "| --- |" + " --- |" * len(scored)
        rows    = []

        for key in keys:
            direction = self._direction(key)
            arrow     = {"higher": " ↑", "lower": " ↓", None: ""}[direction]
            values    = [r.metrics.get(key) for r in scored]
            numeric   = [v for v in values if isinstance(v, (int, float))]

            best = None
            if direction is not None and numeric:
                best = max(numeric) if direction == "higher" else min(numeric)

            cells = []
            for value in values:
                cell = self._fmt(value)
                if best is not None and isinstance(value, (int, float)) and value == best:
                    cell = f"**{cell}**"
                cells.append(cell)

            rows.append(f"| `{key}`{arrow} | " + " | ".join(cells) + " |")

        return [header, divider, *rows, ""]

    def _write_figures(self) -> list[Path]:
        scored = [r for r in self.records if r.has_inference]

        names : set[str] = {path.name for r in scored for path in r.figures}
        claimed : set[str] = set()
        written : list[Path] = []

        groups = list(_FIGURE_GROUPS) + [("Other figures", re.compile(r".*"))]

        for title, pattern in groups:
            group_names = sorted((n for n in names if n not in claimed and pattern.search(n)), key=self._natural_key)
            if not group_names:
                continue
            claimed.update(group_names)

            slug  = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
            lines = self._header(f"Figures – {title}")
            lines += ["> Only trials with at least one completed inference run are shown.\n"]

            for figure_name in group_names:
                lines.append(f"## `{figure_name}`\n")
                for r in scored:
                    figure_path = r.inference_dir / "figures" / figure_name
                    if figure_path.exists():
                        lines.append(f"*{r.name}*  \n![]({self._img_src(figure_path)})\n")
                    else:
                        lines.append(f"*{r.name}* — _(not found)_\n")
                lines.append("")

            out = self.out_dir / f"figures_{slug}.md"
            out.write_text("\n".join(lines), encoding="utf-8")
            written.append(out)

        return written

    def _write_gifs(self) -> Path:
        scored = [r for r in self.records if r.has_inference]
        names  = sorted({path.name for r in scored for path in r.animations}, key=self._natural_key)

        lines = self._header("Animation Comparison")
        lines += ["> Only trials with at least one completed inference run are shown.\n"]

        if not names:
            lines.append("_No animations available yet._\n")

        for gif_name in names:
            lines.append(f"## `{gif_name}`\n")
            for r in scored:
                gif_path = r.inference_dir / "animations" / gif_name
                if gif_path.exists():
                    lines.append(f"*{r.name}*  \n![]({self._img_src(gif_path)})\n")
                else:
                    lines.append(f"*{r.name}* — _(not found)_\n")
            lines.append("")

        out = self.out_dir / "gif_comparison.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def _write_summary_json(self) -> Path:
        payload = []
        for r in self.records:
            payload.append({
                "name"            : r.name,
                "run_dir"         : str(r.run_dir),
                "parameters"      : r.parameters,
                "size_match"      : r.size_match,
                "overfit"         : r.overfit,
                "training_result" : r.training_result,
                "checkpoint"      : r.checkpoint,
                "run_summary"     : r.run_summary,
                "trainer_config"  : r.trainer_config,
                "inference_dir"   : str(r.inference_dir) if r.inference_dir else None,
                "metrics"         : r.metrics,
                "figures"         : [path.name for path in r.figures],
                "animations"      : [path.name for path in r.animations],
                "report_path"     : str(r.report_path) if r.report_path else None,
            })

        out = self.out_dir / "comparison_summary.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        return out
