from __future__ import annotations

import re
from pathlib import Path

from pipelines.comparison.trial_collector import TrialRecord
from pipelines.shared.comparison_report    import ComparisonReportBase
from tools.metrics.scoring                 import FiniteScalar, MetricOrientation
from tools.reporting.markdown              import MarkdownTable, ScalarFormatter
from tools.reporting.reporting             import MetricSectionGrouper, ReportAssets
from tools.monitoring.logger               import Logger


class TrialComparisonReport(ComparisonReportBase):

    FIGURE_SUBDIRS = [
        ("tracks",              "Passes and interferograms"),
        ("input_channels",      "Input channels"),
        ("profiles",            "Profile reconstructions"),
        ("pixel_maps",          "Per-pixel metric maps"),
        ("histograms",          "Metric distributions"),
        ("param_distributions", "Parameter distributions"),
        ("param_scatter",       "Parameter scatter plots"),
        ("param_error_maps",    "Parameter error maps"),
        ("slots",               "Slot diagnostics"),
        ("ssim",                "SSIM curves (denorm)"),
        ("ssim_norm",           "SSIM curves (unit-area)"),
        ("elev_metrics",        "Per-elevation-bin metrics"),
        ("slices",              "Tomogram slices"),
        ("slices_norm",         "Tomogram slices (unit-area)"),
        ("reduced",             "Classical baseline"),
    ]

    HEADLINE_METRICS = [
        ("curve_rmse_gt",                "RMSE"),
        ("curve_mae_gt",                 "MAE"),
        ("overall_r2_gt",                "R²"),
        ("psnr_db_gt",                   "PSNR"),
        ("pixel_r2_gt_mean",             "Pixel R²"),
        ("pixel_cosine_gt_mean",         "Cosine"),
        ("ssim_gt_elev_mean",            "SSIM elev"),
        ("ssim_norm_elev_mean",          "SSIM norm"),
        ("pixel_peak_err_units_mean_gt", "Peak err"),
        ("relative_mse_reduction",       "vs Capon"),
    ]

    COUNT_METRICS = [
        ("count_exact_frac", "Exact"),
        ("count_under_frac", "Under"),
        ("count_over_frac",  "Over"),
    ]

    PRECISION_BUCKET_PATTERN = re.compile(r"^matched_precision_gt(\d+)$")

    def __init__(
        self,
        records        : list[TrialRecord],
        out_dir        : Path,
        compare_images : bool,
        compare_gifs   : bool,
        embed_images   : bool,
        logger         : Logger,
    ) -> None:
        self.records        = records
        self.out_dir        = Path(out_dir)
        self.compare_images = compare_images
        self.compare_gifs   = compare_gifs
        self.logger         = logger
        self.assets         = ReportAssets(base=self.out_dir, embed_images=embed_images)

    def _write_overview(self) -> Path:
        lines  = self.assets.header("Trial Comparison Overview")
        lines += ["## Runs\n"]

        table = MarkdownTable(("Run", "Inference run", "Checkpoint", "Best val loss", "Epochs", "Report"))
        for r in self.records:
            inference  = f"`{r.inference_dir.name}`" if r.has_inference else "_(no inference)_"
            epoch      = ScalarFormatter.format_scalar(r.checkpoint.get("epoch"))
            best_epoch = ScalarFormatter.format_scalar(r.checkpoint.get("best_epoch"))
            best_loss  = ScalarFormatter.format_scalar(r.checkpoint.get("best_val_loss"))
            n_epochs   = ScalarFormatter.format_scalar(r.checkpoint.get("n_train_epochs"))
            report_lnk = f"[report.md]({self.assets.rel(r.report_path)})" if r.report_path else "—"

            table.add_row(f"`{r.name}`", inference, f"{epoch} (best {best_epoch})", best_loss, n_epochs, report_lnk)

        lines += table.render()
        lines.append("")
        lines += self._leaderboard()

        out = self.out_dir / "overview.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def _leaderboard(self) -> list[str]:
        scored = [r for r in self.records if r.metrics]
        if not scored:
            return ["## Leaderboard\n", "_No inference metrics available._\n"]

        headline_intro  = "Mean rank across the headline metrics (1 = best); missing metrics rank last."
        precision_intro = (
            "Mean rank of matched (permutation-invariant) precision, per GT scatterer count `k` — the share of "
            "predicted scatterers that hit a true scatterer where the pixel truly holds `k` (1 = best; missing "
            "buckets rank last). Precision rewards under-prediction, so read it alongside recall, not on its own."
        )
        count_intro = (
            "Mean rank of count calibration (1 = best): `Exact` is the pixel share where the predicted active "
            "scatterer count equals GT (higher better), `Under` / `Over` are the shares where the model predicts "
            "too few / too many (lower better). Permutation-invariant."
        )

        lines  = self._rank_section("Leaderboard",            headline_intro,  self.HEADLINE_METRICS,            scored)
        lines += self._rank_section("Per-Gaussian Precision",  precision_intro, self._precision_metrics(scored),  scored)
        lines += self._rank_section("Count Calibration",       count_intro,     self.COUNT_METRICS,               scored)
        return lines

    def _precision_metrics(self, scored: list[TrialRecord]) -> list[tuple[str, str]]:
        buckets: set[int] = set()
        for r in scored:
            for key in r.metrics:
                match = self.PRECISION_BUCKET_PATTERN.match(key)
                if match:
                    buckets.add(int(match.group(1)))

        metrics = [("matched_precision", "Overall")]
        metrics += [(f"matched_precision_gt{k}", f"k={k}") for k in sorted(buckets)]
        return metrics

    def _rank_section(self, title: str, intro: str, metrics: list[tuple[str, str]], scored: list[TrialRecord]) -> list[str]:
        if not metrics:
            return [f"## {title}\n", "_No applicable metrics available._\n"]

        ranks, mean_ranks = self._rank_metrics(metrics, scored)

        lines = [f"## {title}\n", f"{intro}\n"]

        table = MarkdownTable(["Rank", "Run", "Mean rank", *[label for _, label in metrics]])

        for position, name in enumerate(sorted(mean_ranks, key=mean_ranks.get), start=1):
            cells = [str(ranks[name].get(key, "—")) for key, _ in metrics]
            table.add_row(position, f"`{name}`", f"{mean_ranks[name]:.2f}", *cells)

        lines += table.render()
        lines.append("")
        return lines

    def _write_metrics(self) -> Path:
        scored = [r for r in self.records if r.metrics]

        lines  = self.assets.header("Metrics Comparison")
        lines += ["> Best value per metric in **bold** (↓ lower is better, ↑ higher is better). Per-bin array metrics are excluded.\n"]
        lines += [
            "> **Reading guide.** Gaussian accuracy is reported by the *Matched Gaussian "
            "(Permutation-Invariant)* section, which Hungarian-matches predicted Gaussians to GT before "
            "scoring. `count_acc_gt{k}` is exact-count agreement (calibration), not scatterer recovery — use "
            "`matched_recall_gt{k}`. Precision and F1 reward under-prediction (a model that collapses slots has "
            "few false positives), so do not rank trials on them; read them alongside recall.\n"
        ]

        if not scored:
            lines += ["_No inference metrics available._\n"]
        else:
            all_keys = MetricSectionGrouper.scalar_keys(scored)
            for title, keys in MetricSectionGrouper().group(all_keys):
                lines += [f"## {title}\n", *self._metric_table(keys, scored)]

        out = self.out_dir / "metrics_comparison.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def _metric_table(self, keys: list[str], scored: list[TrialRecord]) -> list[str]:
        headers = ["Run"]
        best    : dict[str, float | None] = {}

        for key in keys:
            direction = MetricOrientation.direction(key)
            arrow     = {"higher": " ↑", "lower": " ↓", None: ""}[direction]
            headers.append(f"`{key}`{arrow}")

            numeric = [v for v in (FiniteScalar.coerce(r.metrics.get(key)) for r in scored) if v is not None]
            best[key] = (max(numeric) if direction == "higher" else min(numeric)) if direction is not None and numeric else None

        table = MarkdownTable(headers)

        for r in scored:
            cells = [f"`{r.name}`"]
            for key in keys:
                value  = r.metrics.get(key)
                cell   = ScalarFormatter.format_scalar(value)
                finite = FiniteScalar.coerce(value)
                if best[key] is not None and finite is not None and finite == best[key]:
                    cell = f"**{cell}**"
                cells.append(cell)
            table.add_row(*cells)

        return [*table.render(), ""]

    def _write_figure_section(self, subdir: str, title: str) -> Path | None:
        scored = [r for r in self.records if r.has_inference and r.figure_subdir(subdir) is not None]
        if not scored:
            return None

        all_names: set[str] = set()
        for r in scored:
            path = r.figure_subdir(subdir)
            if path:
                all_names.update(p.name for p in path.glob("*.png"))

        if not all_names:
            return None

        sorted_names = sorted(all_names, key=ReportAssets.natural_key)

        lines  = self.assets.header(f"Figure Comparison – {title}")
        lines += [f"> Figures from the `{subdir}/` directory. Only trials with a completed inference run are shown.\n"]

        for name in sorted_names:
            lines.append(f"## `{name}`\n")
            for r in scored:
                img_path = r.figure_subdir(subdir) / name
                if img_path.exists():
                    lines.append(f"*{r.name}*  \n![]({self.assets.src(img_path)})\n")
                else:
                    lines.append(f"*{r.name}* — _(not in this run)_\n")
            lines.append("")

        slug = re.sub(r"[^a-z0-9]+", "_", subdir.lower()).strip("_")
        out  = self.out_dir / f"figures_{slug}.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def _write_gifs(self) -> Path | None:
        scored = [r for r in self.records if r.has_inference and r.animations]
        if not scored:
            return None

        all_names: set[str] = set()
        for r in scored:
            all_names.update(p.name for p in r.animations)

        sorted_names = sorted(all_names, key=ReportAssets.natural_key)

        lines  = self.assets.header("Animation Comparison")
        lines += ["> GIF animations from each trial’s latest inference run. Only trials with animations are shown.\n"]

        for name in sorted_names:
            lines.append(f"## `{name}`\n")
            for r in scored:
                gif_path = r.inference_dir / "animations" / name
                if gif_path.exists():
                    lines.append(f"*{r.name}*  \n![]({self.assets.src(gif_path)})\n")
                else:
                    lines.append(f"*{r.name}* — _(not in this run)_\n")
            lines.append("")

        out = self.out_dir / "gif_comparison.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def write_all(self) -> list[Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.logger.section("Writing comparison reports")

        written = [self._write_overview(), self._write_metrics()]
        self.logger.info(f"Overview and metrics written")

        if self.compare_images:
            for subdir, title in self.FIGURE_SUBDIRS:
                path = self._write_figure_section(subdir, title)
                if path:
                    self.logger.info(f"Figures [{subdir}] : {path.name}")
                    written.append(path)

        if self.compare_gifs:
            path = self._write_gifs()
            if path:
                self.logger.info(f"Animations : {path.name}")
                written.append(path)

        return written
