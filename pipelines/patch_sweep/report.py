from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib     import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pipelines.patch_sweep.planner import PatchSweepPlanner, SweepUnit
from tools.data.io                 import FileIO
from tools.monitoring.logger       import Logger
from tools.reporting.markdown      import MarkdownDoc, MarkdownTable, ScalarFormatter
from tools.reporting.plotting      import PlotBase


@dataclass
class SweepRecord:
    unit          : SweepUnit
    status        : str
    duration_s    : float | None
    test_loss     : float | None
    best_val_loss : float | None
    best_epoch    : int | None

    @property
    def complete(self) -> bool:
        return self.test_loss is not None


class SweepCollector:
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

    def collect(self) -> list[SweepRecord]:
        results = self._training_results()

        records = []
        for unit in self.planner.units():
            unit_dir = self.training_dir / unit.name
            result   = results.get(unit.name, {})

            test_loss                 = self._test_loss(unit_dir)
            best_val_loss, best_epoch = self._checkpoint_fields(unit_dir)

            if test_loss is None:
                self.logger.warning(f"{unit.name}: no test metrics, excluded from the ranking")

            records.append(SweepRecord(
                unit          = unit,
                status        = result.get("status", "MISSING"),
                duration_s    = result.get("duration_s"),
                test_loss     = test_loss,
                best_val_loss = best_val_loss,
                best_epoch    = best_epoch,
            ))

        return records


class SweepPlots(PlotBase):
    save_dpi = 300

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir

        self._apply_style()

    def curve(self, track_count: int, records: list[SweepRecord], predicted: float) -> Path:
        complete = [record for record in records if record.complete]

        fig, ax = plt.subplots(figsize=(5.2, 3.6))

        ax.plot([r.unit.patch_size for r in complete], [r.test_loss for r in complete], marker="o", color="#B03052", label="test loss")

        with_val = [record for record in records if record.best_val_loss is not None]
        ax.plot([r.unit.patch_size for r in with_val], [r.best_val_loss for r in with_val], marker="s", color="#787880", label="best validation loss")

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
                    "name"          : record.unit.name,
                    "patch_size"    : record.unit.patch_size,
                    "patch_stride"  : record.unit.patch_stride,
                    "batch_size"    : record.unit.batch_size,
                    "status"        : record.status,
                    "test_loss"     : record.test_loss,
                    "best_val_loss" : record.best_val_loss,
                    "best_epoch"    : record.best_epoch,
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
                ScalarFormatter.format_scalar(best.test_loss),
            )

        return table

    def _group_table(self, group: list[SweepRecord], best: SweepRecord | None) -> MarkdownTable:
        table = MarkdownTable(["Patch (px)", "Stride", "Batch", "Test loss", "Best val loss", "Best epoch", "Status"], align=["right"] * 7)

        for record in group:
            patch = f"**{record.unit.patch_size}**" if best is not None and record is best else str(record.unit.patch_size)
            table.add_row(
                patch,
                record.unit.patch_stride,
                record.unit.batch_size,
                ScalarFormatter.format_scalar(record.test_loss),
                ScalarFormatter.format_scalar(record.best_val_loss),
                record.best_epoch,
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
            "Ranking metric"   : "test avg_loss at the restored best-validation checkpoint",
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
