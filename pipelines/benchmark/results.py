from __future__ import annotations

import gc
import re
from dataclasses import dataclass, field
from pathlib     import Path

from tools.data.io             import FileIO
from tools.reporting.reporting import MetricSectionGrouper, ReportAssets
from tools.metrics.scoring     import FiniteScalar, MetricOrientation
from tools.monitoring.logger   import Logger
from tools.reporting.markdown  import MarkdownTable, ScalarFormatter

_TOTAL_PARAMS_PATTERN = re.compile(r"\*\*Total Parameters:\*\*\s*`([\d,]+)`")
_CHECKPOINT_KEYS      = ("best_val_loss", "best_epoch", "epoch", "global_step")


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

    def _optional_json(self, path: Path) -> dict:
        if not path.exists():
            return {}
        return FileIO.load_json(path)

    def _parse_parameters(self, trial_dir: Path, size_match: dict) -> int | None:
        summary_path = trial_dir / "docs" / "model_doc.md"

        if summary_path.exists():
            match = _TOTAL_PARAMS_PATTERN.search(summary_path.read_text(encoding="utf-8", errors="ignore"))
            if match:
                return int(match.group(1).replace(",", ""))

        return size_match["parameters"] if "parameters" in size_match else None

    def _read_checkpoint(self, trial_dir: Path) -> dict:
        import torch

        checkpoint_path = next(trial_dir.rglob("best_model.pt"), None)
        if checkpoint_path is None:
            return {}

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

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
        record.metrics       = FileIO.load_json(inference_dir / "metrics.json")
        record.figures       = sorted((inference_dir / "figures").glob("*.png")) if (inference_dir / "figures").is_dir() else []
        record.animations    = sorted((inference_dir / "animations").glob("*.gif")) if (inference_dir / "animations").is_dir() else []

        report_path = inference_dir / "report.md"
        if report_path.exists():
            record.report_path = report_path

    def _aggregate_sources(self) -> tuple[dict, dict, dict]:
        size_match       = self._optional_json(self.pipeline_dir / "size_match.json")
        overfit_results  = {r["model"]: r for r in FileIO.load_json(self.pipeline_dir / "overfit_results.json")}
        training_results = {r["name"]:  r for r in FileIO.load_json(self.pipeline_dir / "training_results.json")}

        return size_match, overfit_results, training_results

    def collect(self) -> list[TrialRecord]:
        size_match, overfit_results, training_results = self._aggregate_sources()

        if not self.training_dir.is_dir():
            self.logger.error(f"No training directory found at: {self.training_dir}")
            return []

        records = []
        for trial_dir in sorted(d for d in self.training_dir.iterdir() if d.is_dir()):
            record = TrialRecord(name=trial_dir.name, run_dir=trial_dir)

            record.size_match      = size_match[trial_dir.name] if trial_dir.name in size_match else {}
            record.trainer_config  = self._optional_json(trial_dir / "docs" / "trainer_config.json")
            record.run_summary     = self._optional_json(trial_dir / "meta" / "run_summary.json")
            record.overfit         = overfit_results[trial_dir.name] if trial_dir.name in overfit_results else {}
            record.training_result = training_results[trial_dir.name] if trial_dir.name in training_results else {}
            record.parameters      = self._parse_parameters(trial_dir, record.size_match)
            record.checkpoint      = self._read_checkpoint(trial_dir)

            self._attach_inference(record)

            status = f"inference {record.inference_dir.name}" if record.has_inference else "no inference"
            self.logger.info(f"{record.name:<22} {status}")

            records.append(record)

        return records


class ComparisonReport:

    FIGURE_GROUPS = [
        ("Profile reconstructions",     re.compile(r"^profiles_")),
        ("Per-pixel metric maps",       re.compile(r"^(pixel_|metric_histograms)")),
        ("Gaussian parameter analysis", re.compile(r"^param_")),
        ("Active Gaussian count",       re.compile(r"^active_count")),
        ("SSIM and elevation curves",   re.compile(r"^(ssim_|elev_metric)")),
        ("Azimuth slices",              re.compile(r"^slice_azimuth_")),
        ("Elevation slices",            re.compile(r"^slice_elev_")),
        ("Range slices",                re.compile(r"^slice_range_")),
    ]

    HEADLINE_METRICS = [
        ("curve_rmse_gt",                "RMSE"),
        ("curve_mae_gt",                 "MAE"),
        ("overall_r2_gt",                "R²"),
        ("psnr_db_gt",                   "PSNR"),
        ("pixel_r2_gt_mean",             "Pixel R²"),
        ("pixel_cosine_gt_mean",         "Cosine"),
        ("ssim_gt_elev_mean",            "SSIM elev"),
        ("pixel_peak_err_units_mean_gt", "Peak err"),
    ]

    def __init__(self, records: list[TrialRecord], out_dir: Path, reference_model: str, embed_images: bool, logger: Logger, rank_models: bool = True) -> None:
        self.records         = records
        self.out_dir         = out_dir
        self.reference_model = reference_model
        self.embed_images    = embed_images
        self.logger          = logger
        self.rank_models     = rank_models
        self.assets          = ReportAssets(base=out_dir, embed_images=embed_images)
        self.timestamp       = self.assets.timestamp

    def _write_overview(self) -> Path:
        if self.rank_models:
            lines = self.assets.header("Benchmark Overview")
        else:
            lines  = self.assets.header("Cross-Validation Fold Overview")
            lines += [f"Folds of the single model `{self.reference_model}`; each fold tests a different disjoint azimuth region, so fold-to-fold differences reflect data heterogeneity, not model quality. No ranking is implied.\n"]

        if self.rank_models:
            lines += ["## Model Capacity\n"]
            lines += [f"Reference model: `{self.reference_model}`.\n"]
            lines += self._capacity_table()
            lines.append("")

        lines += ["## Overfit Gate\n"]
        lines += self._overfit_table()
        lines.append("")

        lines += ["## Training\n"]
        lines += self._training_table()
        lines.append("")

        lines += ["## Inference\n"]
        lines += self._inference_table()
        lines.append("")

        if self.rank_models:
            lines += self._leaderboard()

        out = self.out_dir / "benchmark_overview.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def _capacity_table(self) -> list[str]:
        table = MarkdownTable(["Model", "Parameters", "Δ vs reference", "Width scale", "Scaled attributes"])

        for r in self.records:
            parameters = f"{r.parameters:,}" if r.parameters is not None else "—"
            deviation  = f"{r.size_match['deviation_pct']:+.3f} %" if "deviation_pct" in r.size_match else "—"
            scale      = f"{r.size_match['scale']:.4f}" if "scale" in r.size_match else "—"
            overrides  = ", ".join(f"`{k}={v}`" for k, v in r.size_match["overrides"].items()) if "overrides" in r.size_match and r.size_match["overrides"] else "_(defaults)_"

            table.add_row(f"`{r.name}`", parameters, deviation, scale, overrides)

        return table.render()

    def _overfit_table(self) -> list[str]:
        table = MarkdownTable(["Model", "Status", "Final loss", "Converged"])

        for r in self.records:
            final_loss = f"{r.overfit['final_loss']:.4e}" if r.overfit.get("final_loss") is not None else "—"
            converged  = {True: "yes", False: "no"}.get(r.overfit.get("converged"), "—")

            table.add_row(f"`{r.name}`", r.overfit.get("status", "—"), final_loss, converged)

        return table.render()

    def _training_table(self) -> list[str]:
        table = MarkdownTable(["Model", "Status", "Best epoch", "Best val loss", "Epochs run", "Duration"])

        for r in self.records:
            duration_s = r.training_result.get("duration_s")
            duration   = f"{duration_s / 60:.1f} min" if duration_s is not None else "—"

            table.add_row(
                f"`{r.name}`",
                r.training_result.get("status", "—"),
                ScalarFormatter.format_scalar(r.checkpoint.get("best_epoch")),
                ScalarFormatter.format_scalar(r.checkpoint.get("best_val_loss")),
                ScalarFormatter.format_scalar(r.checkpoint.get("n_train_epochs")),
                duration,
            )

        return table.render()

    def _inference_table(self) -> list[str]:
        table = MarkdownTable(["Model", "Inference run", "Figures", "Animations", "Report"])

        for r in self.records:
            inference = f"`{r.inference_dir.name}`" if r.has_inference else "pending"
            report_md = f"[report.md]({self.assets.rel(r.report_path)})" if r.report_path else "—"

            table.add_row(f"`{r.name}`", inference, len(r.figures), len(r.animations), report_md)

        return table.render()

    def _leaderboard(self) -> list[str]:
        scored = [r for r in self.records if r.metrics]
        if not scored:
            return ["## Leaderboard\n", "_No inference metrics available yet._\n"]

        ranks : dict[str, dict[str, int]] = {r.name: {} for r in scored}

        for key, _ in self.HEADLINE_METRICS:
            valued  = [(r.name, value) for r in scored if (value := FiniteScalar.coerce(r.metrics.get(key))) is not None]
            reverse = MetricOrientation.direction(key) == "higher"
            ordered = sorted(valued, key=lambda item: item[1], reverse=reverse)

            for position, (name, _) in enumerate(ordered, start=1):
                ranks[name][key] = position

        worst = len(scored) + 1
        mean_ranks = {
            name: sum(ranks[name].get(key, worst) for key, _ in self.HEADLINE_METRICS) / len(self.HEADLINE_METRICS)
            for name in ranks
        }

        lines = ["## Leaderboard\n", "Mean rank across the headline metrics (1 = best); missing metrics rank last.\n"]

        table = MarkdownTable(["Rank", "Model", "Mean rank", *[label for _, label in self.HEADLINE_METRICS]])

        for position, name in enumerate(sorted(mean_ranks, key=mean_ranks.get), start=1):
            cells = [str(ranks[name].get(key, "—")) for key, _ in self.HEADLINE_METRICS]
            table.add_row(position, f"`{name}`", f"{mean_ranks[name]:.2f}", *cells)

        lines += table.render()
        lines.append("")

        return lines

    def _write_metrics(self) -> Path:
        scored = [r for r in self.records if r.metrics]

        if self.rank_models:
            lines  = self.assets.header("Test Metrics Comparison")
            lines += ["> Best value per metric in **bold** (↓ lower is better, ↑ higher is better). Per-bin array metrics are excluded.\n"]
        else:
            lines  = self.assets.header("Per-Fold Test Metrics")
            lines += ["> Folds of one model across disjoint azimuth regions; no value is highlighted as best (↓ lower is better, ↑ higher is better). Per-bin array metrics are excluded.\n"]

        if not scored:
            lines += ["_No inference metrics available yet._\n"]
        else:
            all_keys = MetricSectionGrouper.scalar_keys(scored)

            for title, keys in MetricSectionGrouper().group(all_keys):
                lines += [f"## {title}\n", *self._metric_table(keys, scored)]

        out = self.out_dir / "metrics_comparison.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def _metric_table(self, keys: list[str], scored: list[TrialRecord]) -> list[str]:
        table = MarkdownTable(["Metric", *[f"`{r.name}`" for r in scored]])

        for key in keys:
            direction = MetricOrientation.direction(key)
            arrow     = {"higher": " ↑", "lower": " ↓", None: ""}[direction]
            values    = [r.metrics.get(key) for r in scored]
            numeric   = [v for v in (FiniteScalar.coerce(value) for value in values) if v is not None]

            best = None
            if self.rank_models and direction is not None and numeric:
                best = max(numeric) if direction == "higher" else min(numeric)

            cells = []
            for value in values:
                cell   = ScalarFormatter.format_scalar(value)
                finite = FiniteScalar.coerce(value)
                if best is not None and finite is not None and finite == best:
                    cell = f"**{cell}**"
                cells.append(cell)

            table.add_row(f"`{key}`{arrow}", *cells)

        return [*table.render(), ""]

    def _write_media(self, groups: list[tuple[str, re.Pattern]] | None, subdir: str, title: str) -> list[Path]:
        scored = [r for r in self.records if r.has_inference]

        if groups is None:
            return [self._write_media_file(self._media_names(scored, subdir), subdir, title, "gif_comparison.md")]

        names   : set[str]   = {name for r in scored for name in self._record_media(r, subdir)}
        claimed : set[str]   = set()
        written : list[Path] = []

        for group_title, pattern in groups + [("Other figures", re.compile(r".*"))]:
            group_names = sorted((n for n in names if n not in claimed and pattern.search(n)), key=ReportAssets.natural_key)
            if not group_names:
                continue
            claimed.update(group_names)

            slug = re.sub(r"[^a-z0-9]+", "_", group_title.lower()).strip("_")
            out  = self._write_media_file(group_names, subdir, f"{title} – {group_title}", f"figures_{slug}.md")
            written.append(out)

        return written

    def _media_names(self, scored: list[TrialRecord], subdir: str) -> list[str]:
        names = {name for r in scored for name in self._record_media(r, subdir)}
        return sorted(names, key=ReportAssets.natural_key)

    def _record_media(self, record: TrialRecord, subdir: str) -> list[str]:
        media = record.figures if subdir == "figures" else record.animations
        return [path.name for path in media]

    def _write_media_file(self, names: list[str], subdir: str, title: str, filename: str) -> Path:
        scored = [r for r in self.records if r.has_inference]

        lines = self.assets.header(title)
        lines += ["> Only trials with at least one completed inference run are shown.\n"]

        if not names:
            lines.append("_No media available yet._\n")

        for media_name in names:
            lines.append(f"## `{media_name}`\n")
            for r in scored:
                media_path = r.inference_dir / subdir / media_name
                if media_path.exists():
                    lines.append(f"*{r.name}*  \n![]({self.assets.src(media_path)})\n")
                else:
                    lines.append(f"*{r.name}* — _(not found)_\n")
            lines.append("")

        out = self.out_dir / filename
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
        return FileIO.save_json(payload, out, indent=2)

    def write_all(self) -> list[Path]:
        FileIO.ensure_dir(self.out_dir)

        written = [self._write_overview(), self._write_metrics()]
        written.extend(self._write_media(self.FIGURE_GROUPS, "figures", "Figures"))
        written.extend(self._write_media(None, "animations", "Animation Comparison"))
        written.append(self._write_summary_json())

        return written
