from __future__ import annotations

from pipelines.shared.trial_collection import TrialRecord
from tools.metrics.ranking             import RankingComputer, RankingResult
from tools.reporting.markdown          import MarkdownTable


class ComparisonReportBase:
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

    HEADLINE_GROUPS = [
        ("Curve error",           ["curve_rmse_gt", "curve_mae_gt", "psnr_db_gt"]),
        ("Variance explained",    ["overall_r2_gt", "pixel_r2_gt_mean"]),
        ("Structural similarity", ["ssim_gt_elev_mean"]),
        ("Shape (cosine)",        ["pixel_cosine_gt_mean"]),
        ("Peak location",         ["pixel_peak_err_units_mean_gt"]),
    ]

    def _ranking(self, metrics: list[tuple[str, str]], scored: list[TrialRecord]) -> RankingResult:
        trials = [(r.name, r.metrics) for r in scored]
        return RankingComputer(metrics, trials).compute()

    def _rank_section(self, entity: str, title: str, intro: str, metrics: list[tuple[str, str]], scored: list[TrialRecord]) -> list[str]:
        if not metrics:
            return [f"## {title}\n", "_No applicable metrics available._\n"]

        result = self._ranking(metrics, scored)
        leader = result.leader_composite()

        lines = [f"## {title}\n", f"{intro}\n"]

        table = MarkdownTable(["#", entity, "Score", "Mean rank", "Wins", "Δ", *[label for _, label in metrics]])

        for position, name in enumerate(result.order(), start=1):
            cells = []
            for key, _ in metrics:
                cell = RankingResult.format_rank(result.metric_rank(name, key))
                if name in result.metric_leaders(key):
                    cell = f"**{cell}**"
                cells.append(cell)

            delta = result.composite[name] - leader

            table.add_row(
                position,
                f"`{name}`",
                f"{result.composite[name]:.3f}",
                f"{result.mean_rank[name]:.2f}",
                result.wins[name],
                f"{delta:+.3f}",
                *cells,
            )

        lines += table.render()
        lines.append("")
        return lines

    def _grouped_section(self, entity: str, title: str, intro: str, groups: list[tuple[str, list[str]]], scored: list[TrialRecord]) -> list[str]:
        metrics = [(key, key) for _, keys in groups for key in keys]
        if not metrics:
            return [f"## {title}\n", "_No applicable metrics available._\n"]

        result    = self._ranking(metrics, scored)
        breakdown = result.group_breakdown(groups)
        if not breakdown.labels:
            return [f"## {title}\n", "_No applicable metrics available._\n"]

        leader = breakdown.leader_overall()

        lines = [f"## {title}\n", f"{intro}\n"]

        table = MarkdownTable(["#", entity, "Grouped score", "Δ", *breakdown.labels])

        for position, name in enumerate(breakdown.order(), start=1):
            cells = []
            for label in breakdown.labels:
                score = breakdown.group_score[name][label]
                rank  = breakdown.group_rank[label][name]
                cell  = f"{score:.3f} ({RankingResult.format_rank(rank)})"
                if name in breakdown.group_leaders(label):
                    cell = f"**{cell}**"
                cells.append(cell)

            delta = breakdown.overall[name] - leader

            table.add_row(position, f"`{name}`", f"{breakdown.overall[name]:.3f}", f"{delta:+.3f}", *cells)

        lines += table.render()
        lines.append("")
        return lines
