from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path

from pipelines.shared.trial_collection import TrialCollector as BaseTrialCollector
from pipelines.shared.trial_collection import TrialRecord    as BaseTrialRecord
from tools.monitoring.logger           import Logger


@dataclass
class TrialRecord(BaseTrialRecord):
    figures_dir : Path | None = None

    def figure_subdir(self, name: str) -> Path | None:
        if self.figures_dir is None:
            return None
        path = self.figures_dir / name
        return path if path.is_dir() else None


class TrialCollector(BaseTrialCollector):
    def __init__(self, runs_dir: Path, run_tags: list[str], logger: Logger) -> None:
        self.runs_dir = runs_dir
        self.run_tags = run_tags
        self.logger   = logger

    def _attach_figures(self, record: TrialRecord, inference_dir: Path) -> None:
        figures_dir = inference_dir / "figures"
        if figures_dir.is_dir():
            record.figures_dir = figures_dir

    def collect(self) -> list[TrialRecord]:
        self.logger.section("Collecting trials")
        records = []

        for tag in self.run_tags:
            run_dir = self.runs_dir / tag
            if not run_dir.is_dir():
                self.logger.error(f"Run directory not found: {run_dir}")
                continue

            record = TrialRecord(name=tag, run_dir=run_dir)
            record.checkpoint = self._read_checkpoint(run_dir)
            self._attach_inference(record)

            status = f"inference {record.inference_dir.name}" if record.has_inference else "no inference"
            self.logger.info(f"{record.name:<36} {status}")

            records.append(record)

        return records
