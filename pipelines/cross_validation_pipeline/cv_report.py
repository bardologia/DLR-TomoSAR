from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from pipelines.benchmark_pipeline.results import _METRIC_SECTIONS, _PER_BIN_PATTERN, ComparisonReport
from pipelines.benchmark_pipeline.results import TrialRecord
from pipelines.cross_validation_pipeline.folds import FoldPlanner
from tools.logger import Logger
from tools.markdown import MarkdownTable


class CrossValidationReport:
    def __init__(
        self,
        base_records     : list[TrialRecord],
        records_by_split : dict[str, list[TrialRecord]],
        planner          : FoldPlanner,
        out_dir          : Path,
        model_name       : str,
        embed_images     : bool,
        logger           : Logger,
    ) -> None:
        self.base_records     = base_records
        self.records_by_split = records_by_split
        self.planner          = planner
        self.out_dir          = out_dir
        self.model_name       = model_name
        self.embed_images     = embed_images
        self.logger           = logger
        self.timestamp        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def write_all(self) -> list[Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        written = [self._write_aggregate()]

        for split, records in self.records_by_split.items():
            report = ComparisonReport(
                records         = records,
                out_dir         = self.out_dir / split,
                reference_model = self.model_name,
                embed_images    = self.embed_images,
                logger          = self.logger,
            )
            written.extend(report.write_all())

        written.append(self._write_summary_json())

        return written

    def _mean_std(self, values: list[float]) -> tuple[float, float]:
        n    = len(values)
        mean = sum(values) / n

        if n < 2:
            return mean, 0.0

        variance = sum((value - mean) ** 2 for value in values) / (n - 1)
        return mean, math.sqrt(variance)

    def _fmt(self, value) -> str:
        if isinstance(value, float):
            return f"{value:.5g}"
        if value is None:
            return "—"
        return str(value)

    def _scalar_keys(self, records: list[TrialRecord]) -> list[str]:
        return sorted({
            key
            for record in records
            for key, value in record.metrics.items()
            if isinstance(value, (int, float)) and not _PER_BIN_PATTERN.search(key)
        })

    def _aggregate_table(self, keys: list[str], records: list[TrialRecord]) -> list[str]:
        fold_labels = [f"`{record.name}`" for record in records]
        table       = MarkdownTable(["Metric", "Mean", "Std", *fold_labels])

        for key in keys:
            values  = [record.metrics.get(key) for record in records]
            numeric = [value for value in values if isinstance(value, (int, float))]

            if not numeric:
                continue

            mean, std = self._mean_std([float(value) for value in numeric])
            table.add_row(f"`{key}`", self._fmt(mean), self._fmt(std), *[self._fmt(value) for value in values])

        return [*table.render(), ""]

    def _write_aggregate(self) -> Path:
        lines = [
            "# Cross-Validation Aggregate Report",
            f"\n_Generated {self.timestamp}_\n",
            f"Model `{self.model_name}`, {self.planner.n_folds} folds. All aggregates are reported as mean and sample standard deviation across folds.\n",
        ]

        lines += ["## Fold Plan\n"]

        plan_table = MarkdownTable(["Fold", "Test azimuth", "Val azimuth", "Train azimuth regions"])
        for plan in self.planner.plans():
            test_region   = plan.split_regions.regions("test")[0]
            val_region    = plan.split_regions.regions("val")[0]
            train_regions = plan.split_regions.regions("train")
            train_text    = ", ".join(f"[{r.azimuth_start}, {r.azimuth_end})" for r in train_regions)
            plan_table.add_row(plan.fold_index, f"[{test_region.azimuth_start}, {test_region.azimuth_end})", f"[{val_region.azimuth_start}, {val_region.azimuth_end})", train_text)

        lines += [*plan_table.render(), ""]

        lines += self._training_aggregate()

        for split, records in self.records_by_split.items():
            scored = [record for record in records if record.metrics]

            lines += [f"## Split `{split}` — Metric Aggregates\n"]

            if not scored:
                lines += ["_No inference metrics available for this split._\n"]
                continue

            if len(scored) < len(records):
                missing = [record.name for record in records if not record.metrics]
                lines += [f"_Missing folds (no metrics): {', '.join(missing)}._\n"]

            all_keys = self._scalar_keys(scored)
            claimed  : set[str] = set()

            for title, pattern in _METRIC_SECTIONS:
                keys = [key for key in all_keys if key not in claimed and pattern.search(key)]
                if not keys:
                    continue
                claimed.update(keys)
                lines += [f"### {title}\n", *self._aggregate_table(keys, scored)]

            leftover = [key for key in all_keys if key not in claimed]
            if leftover:
                lines += ["### Other Metrics\n", *self._aggregate_table(leftover, scored)]

        out = self.out_dir / "cv_aggregate_report.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def _training_aggregate(self) -> list[str]:
        lines = ["## Training Across Folds\n"]
        table = MarkdownTable(["Fold", "Best epoch", "Best val loss", "Epochs run", "Duration"])

        best_losses = []
        best_epochs = []
        durations   = []

        for record in self.base_records:
            best_val_loss = record.checkpoint.get("best_val_loss")
            best_epoch    = record.checkpoint.get("best_epoch")
            duration_s    = record.training_result.get("duration_s")

            if isinstance(best_val_loss, (int, float)):
                best_losses.append(float(best_val_loss))
            if isinstance(best_epoch, (int, float)):
                best_epochs.append(float(best_epoch))
            if isinstance(duration_s, (int, float)):
                durations.append(float(duration_s))

            duration = f"{duration_s / 60:.1f} min" if duration_s is not None else "—"
            table.add_row(f"`{record.name}`", self._fmt(best_epoch), self._fmt(best_val_loss), self._fmt(record.checkpoint.get("n_train_epochs")), duration)

        lines += [*table.render(), ""]

        summary = MarkdownTable(["Quantity", "Mean", "Std"])
        if best_losses:
            mean, std = self._mean_std(best_losses)
            summary.add_row("Best val loss", self._fmt(mean), self._fmt(std))
        if best_epochs:
            mean, std = self._mean_std(best_epochs)
            summary.add_row("Best epoch", self._fmt(mean), self._fmt(std))
        if durations:
            mean, std = self._mean_std(durations)
            summary.add_row("Duration (min)", self._fmt(mean / 60), self._fmt(std / 60))

        if not summary.is_empty():
            lines += [*summary.render(), ""]

        return lines

    def _write_summary_json(self) -> Path:
        payload : dict = {
            "model"   : self.model_name,
            "n_folds" : self.planner.n_folds,
            "folds"   : [record.name for record in self.base_records],
            "splits"  : {},
        }

        for split, records in self.records_by_split.items():
            scored = [record for record in records if record.metrics]
            keys   = self._scalar_keys(scored)

            split_payload = {}
            for key in keys:
                values = [float(record.metrics[key]) for record in scored if isinstance(record.metrics.get(key), (int, float))]
                if not values:
                    continue
                mean, std = self._mean_std(values)
                split_payload[key] = {
                    "mean"     : mean,
                    "std"      : std,
                    "per_fold" : {record.name: record.metrics.get(key) for record in scored},
                }

            payload["splits"][split] = split_payload

        best_losses = [record.checkpoint.get("best_val_loss") for record in self.base_records]
        best_losses = [float(value) for value in best_losses if isinstance(value, (int, float))]
        if best_losses:
            mean, std = self._mean_std(best_losses)
            payload["best_val_loss"] = {"mean": mean, "std": std}

        out = self.out_dir / "cv_summary.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        return out
