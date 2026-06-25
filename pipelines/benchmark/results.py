from __future__ import annotations

import math
import re
from dataclasses import replace
from pathlib     import Path

import numpy as np

from pipelines.shared.comparison_report import ComparisonReportBase
from pipelines.shared.seed_sweep        import SeedSet
from pipelines.shared.trial_collection  import SeedAggregation, TrialCollector, TrialRecord
from tools.data.io                      import FileIO
from tools.reporting.reporting          import MetricSectionGrouper, ReportAssets
from tools.metrics.scoring              import FiniteScalar, MetricOrientation
from tools.monitoring.logger            import Logger
from tools.reporting.markdown           import MarkdownTable, ScalarFormatter


class BenchmarkSeedCollector(TrialCollector):
    CHECKPOINT_KEYS = ("best_val_loss", "best_epoch", "n_train_epochs")

    def __init__(self, run_dir: Path, logger: Logger) -> None:
        super().__init__(run_dir=run_dir, logger=logger)
        self.seed_dispersion = {}

    def _group_by_model(self, records: list[TrialRecord]) -> list[tuple[str, list[TrialRecord]]]:
        groups: dict[str, list[TrialRecord]] = {}

        for record in records:
            groups.setdefault(SeedSet.base(record.name), []).append(record)

        return list(groups.items())

    def _aggregate_group(self, model_name: str, runs: list[TrialRecord]) -> tuple[TrialRecord, dict]:
        representative = next((run for run in runs if run.has_inference), runs[0])

        metric_keys             = sorted({key for run in runs for key in run.metrics})
        metric_means, metric_std = SeedAggregation.aggregate([run.metrics for run in runs], metric_keys)
        ckpt_means, ckpt_std     = SeedAggregation.aggregate([run.checkpoint for run in runs], list(self.CHECKPOINT_KEYS))

        overfit_losses = [run.overfit.get("final_loss") for run in runs]
        overfit_losses = [value for value in overfit_losses if isinstance(value, (int, float))]
        durations      = [run.training_result.get("duration_s") for run in runs]
        durations      = [value for value in durations if isinstance(value, (int, float))]

        record                 = replace(representative, name=model_name, metrics=metric_means)
        record.checkpoint      = {**representative.checkpoint, **ckpt_means}
        record.training_result = {
            "status"     : "DONE" if all(run.training_result.get("status") == "DONE" for run in runs) else "PARTIAL",
            "duration_s" : float(np.mean(durations)) if durations else None,
        }
        record.overfit = {
            **representative.overfit,
            "status"     : "PASS" if all(run.overfit.get("status") == "PASS" for run in runs) else "FAIL",
            "final_loss" : float(np.mean(overfit_losses)) if overfit_losses else None,
            "converged"  : all(run.overfit.get("converged") for run in runs) if all("converged" in run.overfit for run in runs) else None,
        }

        dispersion = {
            "n_seeds"           : len(runs),
            "best_val_loss_std" : ckpt_std.get("best_val_loss"),
            "metrics"           : metric_std,
        }

        return record, dispersion

    def collect(self) -> list[TrialRecord]:
        self.seed_dispersion = {}
        aggregated           = []

        for model_name, runs in self._group_by_model(super().collect()):
            if len(runs) == 1:
                aggregated.append(runs[0])
                continue

            record, dispersion = self._aggregate_group(model_name, runs)
            aggregated.append(record)
            self.seed_dispersion[model_name] = dispersion

        return aggregated


class ComparisonReport(ComparisonReportBase):

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

    def __init__(self, records: list[TrialRecord], out_dir: Path, reference_model: str, embed_images: bool, logger: Logger, rank_models: bool = True, seed_dispersion: dict | None = None) -> None:
        self.records         = records
        self.out_dir         = out_dir
        self.reference_model = reference_model
        self.embed_images    = embed_images
        self.logger          = logger
        self.rank_models     = rank_models
        self.seed_dispersion = seed_dispersion or {}
        self.has_seed_sweep  = any(entry["n_seeds"] > 1 for entry in self.seed_dispersion.values())
        self.assets          = ReportAssets(base=out_dir, embed_images=embed_images)
        self.timestamp       = self.assets.timestamp

    def _seed_annotated(self, cell: str, std: float | None) -> str:
        if std is None or not math.isfinite(std):
            return cell

        return f"{cell} ± {ScalarFormatter.format_scalar(std)}"

    def _metric_seed_std(self, name: str, key: str) -> float | None:
        entry = self.seed_dispersion.get(name)
        return entry["metrics"].get(key) if entry else None

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

            best_val_loss_std = self.seed_dispersion.get(r.name, {}).get("best_val_loss_std")

            table.add_row(
                f"`{r.name}`",
                r.training_result.get("status", "—"),
                ScalarFormatter.format_scalar(r.checkpoint.get("best_epoch")),
                self._seed_annotated(ScalarFormatter.format_scalar(r.checkpoint.get("best_val_loss")), best_val_loss_std),
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

        ranks, mean_ranks = self._rank_metrics(self.HEADLINE_METRICS, scored)

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
            if self.has_seed_sweep:
                lines += ["> Each model was trained under multiple seeds; cells show the seed mean ± seed standard deviation.\n"]
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
            for r, value in zip(scored, values):
                cell   = ScalarFormatter.format_scalar(value)
                finite = FiniteScalar.coerce(value)
                if best is not None and finite is not None and finite == best:
                    cell = f"**{cell}**"
                cell = self._seed_annotated(cell, self._metric_seed_std(r.name, key))
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
            entry = {
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
            }

            if r.name in self.seed_dispersion:
                entry["seed_dispersion"] = self.seed_dispersion[r.name]

            payload.append(entry)

        out = self.out_dir / "comparison_summary.json"
        return FileIO.save_json(payload, out, indent=2)

    def write_all(self) -> list[Path]:
        FileIO.ensure_dir(self.out_dir)

        written = [self._write_overview(), self._write_metrics()]
        written.extend(self._write_media(self.FIGURE_GROUPS, "figures", "Figures"))
        written.extend(self._write_media(None, "animations", "Animation Comparison"))
        written.append(self._write_summary_json())

        return written
