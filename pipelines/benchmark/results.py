from __future__ import annotations

from dataclasses import replace
from pathlib     import Path

import numpy as np

from pipelines.shared.comparison.trial_collection import TrialCollector, TrialRecord
from pipelines.shared.training.seed_sweep        import SeedSet
from tools.metrics.scoring              import SeedAggregation
from tools.monitoring.logger            import Logger


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

        durations      = [run.training_result.get("duration_s") for run in runs]
        durations      = [value for value in durations if isinstance(value, (int, float))]

        record                 = replace(representative, name=model_name, metrics=metric_means)
        record.checkpoint      = {**representative.checkpoint, **ckpt_means}
        record.training_result = {
            "status"     : "DONE" if all(run.training_result.get("status") == "DONE" for run in runs) else "PARTIAL",
            "duration_s" : float(np.mean(durations)) if durations else None,
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
