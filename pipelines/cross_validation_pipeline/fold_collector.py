from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pipelines.benchmark_pipeline.trial_collector import TrialCollector, TrialRecord
from tools.logger import Logger


class FoldCollector(TrialCollector):
    def __init__(self, run_dir: Path, splits: list[str], logger: Logger) -> None:
        super().__init__(run_dir=run_dir, logger=logger)
        self.training_dir = run_dir / "folds"
        self.splits       = splits

    def collect_by_split(self) -> tuple[list[TrialRecord], dict[str, list[TrialRecord]]]:
        base_records = self.collect()

        records_by_split = {
            split: [self._split_view(record, split) for record in base_records]
            for split in self.splits
        }

        return base_records, records_by_split

    def _split_view(self, record: TrialRecord, split: str) -> TrialRecord:
        inference_dir = record.run_dir / "inference" / split

        if not (inference_dir / "metrics.json").exists():
            return replace(record, inference_dir=None, metrics={}, figures=[], animations=[], report_path=None)

        metrics     = self._load_json(inference_dir / "metrics.json") or {}
        figures     = sorted((inference_dir / "figures").glob("*.png")) if (inference_dir / "figures").is_dir() else []
        animations  = sorted((inference_dir / "animations").glob("*.gif")) if (inference_dir / "animations").is_dir() else []
        report_path = inference_dir / "report.md"

        return replace(
            record,
            inference_dir = inference_dir,
            metrics       = metrics,
            figures       = figures,
            animations    = animations,
            report_path   = report_path if report_path.exists() else None,
        )
