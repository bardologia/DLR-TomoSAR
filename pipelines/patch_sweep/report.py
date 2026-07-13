from __future__ import annotations

import gc
import re
from dataclasses import dataclass, field
from pathlib     import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pipelines.patch_sweep.planner                import PatchSweepPlanner, SweepUnit
from tools.data.io                 import FileIO
from tools.metrics.scoring         import SeedAggregation
from tools.monitoring.logger       import Logger
from tools.reporting.markdown      import MarkdownDoc, MarkdownTable, ScalarFormatter
from tools.reporting.plotting      import PlotBase


@dataclass
class SweepRecord:
    unit              : SweepUnit
    status            : str
    duration_s        : float | None
    test_loss         : float | None
    best_val_loss     : float | None
    best_epoch        : float | None
    n_seeds           : int          = 1
    test_loss_std     : float | None = None
    best_val_loss_std : float | None = None
    seed_runs         : list[dict]   = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.test_loss is not None


class SweepCollector:

    SEED_DIR_PATTERN = re.compile(r"seed\d+")

    def __init__(self, run_dir: Path, planner: PatchSweepPlanner, logger: Logger) -> None:
        self.run_dir      = run_dir
        self.training_dir = run_dir / "training"
        self.planner      = planner
        self.logger       = logger

    def _training_results(self) -> dict:
        path = self.run_dir / "pipeline" / "training_results.json"
        if not path.exists():
            raise FileNotFoundError(f"The sweep report needs {path}; run the training stage first")

        return {result["name"]: result for result in FileIO.load_json(path)}

    def _test_loss(self, unit_dir: Path) -> float | None:
        path = unit_dir / "meta" / "test_metrics.json"
        if not path.exists():
            return None

        return float(FileIO.load_json(path)["avg_loss"])

    def _checkpoint_fields(self, unit_dir: Path) -> tuple[float | None, int | None]:
        path = next(unit_dir.rglob("best_model.pt"), None)
        if path is None:
            return None, None

        checkpoint    = torch.load(path, map_location="cpu", weights_only=False)
        best_val_loss = checkpoint.get("best_val_loss")
        best_epoch    = checkpoint.get("best_epoch")

        del checkpoint
        gc.collect()

        return (float(best_val_loss) if best_val_loss is not None else None,
                int(best_epoch)      if best_epoch    is not None else None)

    def _seed_runs(self, unit_dir: Path) -> list[tuple[str | None, Path]]:
        if unit_dir.is_dir():
            seed_dirs = sorted(d for d in unit_dir.iterdir() if d.is_dir() and self.SEED_DIR_PATTERN.fullmatch(d.name))
            if seed_dirs:
                return [(seed_dir.name, seed_dir) for seed_dir in seed_dirs]

        return [(None, unit_dir)]

    def _collect_seed_run(self, unit: SweepUnit, seed_name: str | None, run_dir: Path, results: dict) -> dict:
        name   = unit.name if seed_name is None else f"{unit.name}/{seed_name}"
        result = results.get(name, {})

        test_loss                 = self._test_loss(run_dir)
        best_val_loss, best_epoch = self._checkpoint_fields(run_dir)

        return {
            "name"          : name,
            "status"        : result.get("status", "MISSING"),
            "duration_s"    : result.get("duration_s"),
            "test_loss"     : test_loss,
            "best_val_loss" : best_val_loss,
            "best_epoch"    : best_epoch,
        }

    @staticmethod
    def _status(runs: list[dict]) -> str:
        statuses = [run["status"] for run in runs]

        if all(status == "DONE" for status in statuses):
            return "DONE"
        if any(status == "DONE" for status in statuses):
            return "PARTIAL"
        return statuses[0] if len(set(statuses)) == 1 else "MISSING"

    def _aggregate(self, unit: SweepUnit, runs: list[dict]) -> SweepRecord:
        means, stds = SeedAggregation.aggregate(runs, ["test_loss", "best_val_loss", "best_epoch", "duration_s"])

        return SweepRecord(
            unit              = unit,
            status            = self._status(runs),
            duration_s        = means.get("duration_s"),
            test_loss         = means.get("test_loss"),
            best_val_loss     = means.get("best_val_loss"),
            best_epoch        = means.get("best_epoch"),
            n_seeds           = len(runs),
            test_loss_std     = stds.get("test_loss"),
            best_val_loss_std = stds.get("best_val_loss"),
            seed_runs         = runs,
        )

    def collect(self) -> list[SweepRecord]:
        results = self._training_results()

        records = []
        for unit in self.planner.units():
            unit_dir = self.training_dir / unit.name
            runs     = [self._collect_seed_run(unit, seed_name, run_dir, results) for seed_name, run_dir in self._seed_runs(unit_dir)]
            record   = self._aggregate(unit, runs)

            if record.test_loss is None:
                self.logger.warning(f"{unit.name}: no test metrics, excluded from the ranking")

            records.append(record)

        return records


class SweepPlots(PlotBase):
    save_dpi = 300

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir

        self._apply_style()

    def _band(self, ax, records: list[SweepRecord], value_key: str, std_key: str, color: str) -> None:
        with_std = [record for record in records if getattr(record, std_key) is not None]
        if len(with_std) < 2:
            return

        sizes  = np.array([record.unit.patch_size for record in with_std], dtype=np.float64)
        means  = np.array([getattr(record, value_key) for record in with_std], dtype=np.float64)
        stds   = np.array([getattr(record, std_key)   for record in with_std], dtype=np.float64)

        ax.fill_between(sizes, means - stds, means + stds, color=color, alpha=0.15, linewidth=0)

    def curve(self, track_count: int, records: list[SweepRecord], predicted: float) -> Path:
        complete = [record for record in records if record.complete]

        fig, ax = plt.subplots(figsize=(5.2, 3.6))

        ax.plot([r.unit.patch_size for r in complete], [r.test_loss for r in complete], marker="o", color="#B03052", label="test loss")
        self._band(ax, complete, "test_loss", "test_loss_std", "#B03052")

        with_val = [record for record in records if record.best_val_loss is not None]
        ax.plot([r.unit.patch_size for r in with_val], [r.best_val_loss for r in with_val], marker="s", color="#787880", label="best validation loss")
        self._band(ax, with_val, "best_val_loss", "best_val_loss_std", "#787880")

        ax.axvline(predicted, color="#1E6E46", linestyle="--", linewidth=1.0, label=rf"predicted $W^{{*}} = {predicted:.1f}$ px")

        ax.set_xlabel("patch size (px)")
        ax.set_ylabel("held-out loss")
        ax.set_title(f"{track_count} tracks")
        ax.legend(frameon=False)

        return self._save(fig, self.out_dir / "curves" / f"n{track_count:02d}.png")

    def best_versus_tracks(self, best: dict[int, int], planner: PatchSweepPlanner) -> Path:
        counts = sorted(best)

        fig, ax = plt.subplots(figsize=(5.2, 3.6))

        n_grid = np.linspace(min(counts), max(counts), 200)
        ax.plot(n_grid, [planner.predicted_optimum(n) for n in n_grid], color="#1E6E46", linestyle="--", linewidth=1.0, label=r"predicted $W^{*} = w\sqrt{N/n}$")

        ax.plot(counts, [best[n] for n in counts], marker="o", linestyle="none", color="#B03052", label="observed best patch")

        ax.set_xlabel("tracks used $n$")
        ax.set_ylabel("patch size (px)")
        ax.set_title("Best patch size versus track count")
        ax.legend(frameon=False)

        return self._save(fig, self.out_dir / "best_patch_vs_tracks.png")


class PatchSweepReport:
    def __init__(self, records: list[SweepRecord], planner: PatchSweepPlanner, out_dir: Path, logger: Logger) -> None:
        self.records = records
        self.planner = planner
        self.out_dir = out_dir
        self.logger  = logger
        self.plots   = SweepPlots(out_dir)

    def _by_track_count(self) -> dict[int, list[SweepRecord]]:
        groups: dict[int, list[SweepRecord]] = {}
        for record in self.records:
            groups.setdefault(record.unit.track_count, []).append(record)

        return {count: sorted(group, key=lambda r: r.unit.patch_size) for count, group in sorted(groups.items())}

    def _best(self, group: list[SweepRecord]) -> SweepRecord | None:
        complete = [record for record in group if record.complete]
        if not complete:
            return None

        return min(complete, key=lambda record: record.test_loss)

    def _summary_payload(self, groups: dict[int, list[SweepRecord]]) -> dict:
        per_count = {}
        for count, group in groups.items():
            best = self._best(group)

            per_count[str(count)] = {
                "secondary_labels" : list(group[0].unit.secondary_labels),
                "predicted_optimum": self.planner.predicted_optimum(count),
                "best_patch_size"  : best.unit.patch_size if best else None,
                "best_test_loss"   : best.test_loss       if best else None,
                "units"            : [{
                    "name"              : record.unit.name,
                    "patch_size"        : record.unit.patch_size,
                    "patch_stride"      : record.unit.patch_stride,
                    "batch_size"        : record.unit.batch_size,
                    "status"            : record.status,
                    "n_seeds"           : record.n_seeds,
                    "test_loss"         : record.test_loss,
                    "test_loss_std"     : record.test_loss_std,
                    "best_val_loss"     : record.best_val_loss,
                    "best_val_loss_std" : record.best_val_loss_std,
                    "best_epoch"        : record.best_epoch,
                    "seed_runs"         : record.seed_runs,
                } for record in group],
            }

        return {
            "boxcar_window" : self.planner.config.boxcar_window,
            "total_tracks"  : self.planner.total_tracks,
            "patch_step"    : self.planner.patch_step(),
            "track_counts"  : per_count,
        }

    def _write_json(self, groups: dict[int, list[SweepRecord]]) -> Path:
        path = self.out_dir / "patch_sweep.json"
        FileIO.save_json(self._summary_payload(groups), path, indent=2)

        return path

    def _write_curves(self, groups: dict[int, list[SweepRecord]]) -> list[Path]:
        return [self.plots.curve(count, group, self.planner.predicted_optimum(count)) for count, group in groups.items()]

    def _write_best_plot(self, groups: dict[int, list[SweepRecord]]) -> Path | None:
        best = {count: self._best(group) for count, group in groups.items()}
        best = {count: record.unit.patch_size for count, record in best.items() if record is not None}

        if len(best) < 2:
            self.logger.warning("Fewer than two track counts have a ranked best patch; the best-versus-tracks figure is skipped")
            return None

        return self.plots.best_versus_tracks(best, self.planner)

    def _summary_table(self, groups: dict[int, list[SweepRecord]]) -> MarkdownTable:
        table = MarkdownTable(["Tracks n", "Predicted W* (px)", "Best patch (px)", "Ratio best/W*", "Test loss at best"], align=["right"] * 5)

        for count, group in groups.items():
            best      = self._best(group)
            predicted = self.planner.predicted_optimum(count)

            if best is None:
                table.add_row(count, f"{predicted:.1f}", None, None, None)
                continue

            table.add_row(
                count,
                f"{predicted:.1f}",
                best.unit.patch_size,
                f"{best.unit.patch_size / predicted:.2f}",
                self._with_std(best.test_loss, best.test_loss_std),
            )

        return table

    @staticmethod
    def _with_std(value: float | None, std: float | None) -> str | None:
        if value is None:
            return None

        rendered = ScalarFormatter.format_scalar(value)
        return rendered if std is None else f"{rendered} ± {ScalarFormatter.format_scalar(std)}"

    def _group_table(self, group: list[SweepRecord], best: SweepRecord | None) -> MarkdownTable:
        table = MarkdownTable(["Patch (px)", "Stride", "Batch", "Seeds", "Test loss", "Best val loss", "Best epoch", "Status"], align=["right"] * 8)

        for record in group:
            patch = f"**{record.unit.patch_size}**" if best is not None and record is best else str(record.unit.patch_size)
            table.add_row(
                patch,
                record.unit.patch_stride,
                record.unit.batch_size,
                record.n_seeds,
                self._with_std(record.test_loss, record.test_loss_std),
                self._with_std(record.best_val_loss, record.best_val_loss_std),
                ScalarFormatter.format_scalar(record.best_epoch),
                record.status,
            )

        return table

    def _write_markdown(self, groups: dict[int, list[SweepRecord]], curve_paths: list[Path], best_plot: Path | None) -> Path:
        config = self.planner.config
        doc    = MarkdownDoc("Patch-size sweep")

        doc.paragraph("Best patch size per track count, against the sample-budget prediction W* = w sqrt(N/n): the ground truth estimates each label from a w x w boxcar covariance over N tracks, so a network reading n tracks must repay the missing samples with receptive-field area.")
        doc.kv_table({
            "Backbone"         : f"{config.backbone_name}-{config.backbone_head}",
            "Dataset"          : config.paths.dataset_path,
            "Boxcar window w"  : config.boxcar_window,
            "Total tracks N"   : self.planner.total_tracks,
            "Patch step"       : self.planner.patch_step(),
            "Seeds"            : config.seeds or [config.seed],
            "Ranking metric"   : "seed-mean test avg_loss at the restored best-validation checkpoint",
        })

        doc.heading("Best patch size per track count", level=2)
        doc.table(self._summary_table(groups))

        if best_plot is not None:
            doc.image("best patch size versus track count", best_plot.relative_to(self.out_dir))

        for (count, group), curve_path in zip(groups.items(), curve_paths):
            best = self._best(group)

            doc.heading(f"n = {count} tracks", level=2)
            doc.bold_kv("Secondaries", ", ".join(group[0].unit.secondary_labels))
            doc.blank()
            doc.table(self._group_table(group, best))
            doc.image(f"loss versus patch size at n = {count}", curve_path.relative_to(self.out_dir))

        path = self.out_dir / "report.md"
        doc.save(path)

        return path

    def write_all(self) -> list[Path]:
        groups = self._by_track_count()

        json_path   = self._write_json(groups)
        curve_paths = self._write_curves(groups)
        best_plot   = self._write_best_plot(groups)
        md_path     = self._write_markdown(groups, curve_paths, best_plot)

        written = [json_path, *curve_paths, md_path]
        if best_plot is not None:
            written.append(best_plot)

        return written
