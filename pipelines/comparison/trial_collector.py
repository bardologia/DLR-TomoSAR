from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib     import Path

from tools.data.io           import FileIO
from tools.monitoring.logger import Logger


_CHECKPOINT_KEYS = ("best_val_loss", "best_epoch", "epoch", "global_step")


@dataclass
class TrialRecord:
    name          : str
    run_dir       : Path
    checkpoint    : dict       = field(default_factory=dict)
    inference_dir : Path | None = None
    metrics       : dict       = field(default_factory=dict)
    figures_dir   : Path | None = None
    animations    : list[Path] = field(default_factory=list)
    report_path   : Path | None = None

    @property
    def has_inference(self) -> bool:
        return self.inference_dir is not None

    def figure_subdir(self, name: str) -> Path | None:
        if self.figures_dir is None:
            return None
        path = self.figures_dir / name
        return path if path.is_dir() else None


class TrialCollector:
    def __init__(self, runs_dir: Path, run_tags: list[str], logger: Logger) -> None:
        self.runs_dir = runs_dir
        self.run_tags = run_tags
        self.logger   = logger

    def _read_checkpoint(self, run_dir: Path) -> dict:
        import torch

        checkpoint_path = next(run_dir.rglob("best_model.pt"), None)
        if checkpoint_path is None:
            return {}

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        info = {key: checkpoint.get(key) for key in _CHECKPOINT_KEYS}
        info["n_train_epochs"] = len(checkpoint.get("train_losses") or [])

        del checkpoint
        gc.collect()

        return {key: value for key, value in info.items() if value is not None}

    def _attach_inference(self, record: TrialRecord) -> None:
        inference_root = record.run_dir / "inference"
        if not inference_root.is_dir():
            return

        candidates = sorted(d for d in inference_root.iterdir() if d.is_dir() and (d / "metrics.json").exists())
        if not candidates:
            return

        inference_dir = candidates[-1]

        record.inference_dir = inference_dir
        record.metrics       = FileIO.load_json(inference_dir / "metrics.json")

        figures_dir = inference_dir / "figures"
        if figures_dir.is_dir():
            record.figures_dir = figures_dir

        animations_dir = inference_dir / "animations"
        if animations_dir.is_dir():
            record.animations = sorted(animations_dir.glob("*.gif"))

        report_path = inference_dir / "report.md"
        if report_path.exists():
            record.report_path = report_path

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
