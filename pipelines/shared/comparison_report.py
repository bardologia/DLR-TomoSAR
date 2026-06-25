from __future__ import annotations

from pipelines.shared.trial_collection import TrialRecord
from tools.metrics.scoring             import FiniteScalar, MetricOrientation


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

    def _rank_metrics(self, metrics: list[tuple[str, str]], scored: list[TrialRecord]) -> tuple[dict, dict]:
        ranks : dict[str, dict[str, int]] = {r.name: {} for r in scored}

        for key, _ in metrics:
            valued  = [(r.name, value) for r in scored if (value := FiniteScalar.coerce(r.metrics.get(key))) is not None]
            reverse = MetricOrientation.direction(key) == "higher"
            ordered = sorted(valued, key=lambda item: item[1], reverse=reverse)

            for position, (name, _) in enumerate(ordered, start=1):
                ranks[name][key] = position

        worst = len(scored) + 1
        mean_ranks = {
            name: sum(ranks[name].get(key, worst) for key, _ in metrics) / len(metrics)
            for name in ranks
        }

        return ranks, mean_ranks
