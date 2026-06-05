from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from tools.logger import Logger

_TOTAL_PARAMS_PATTERN = re.compile(r"\*\*Total Parameters:\*\*\s*`([\d,]+)`")
_CHECKPOINT_KEYS      = ("best_val_loss", "best_epoch", "epoch", "global_step")


@dataclass
class TrialRecord:
    name            : str
    run_dir         : Path
    parameters      : int | None  = None
    size_match      : dict        = field(default_factory=dict)
    trainer_config  : dict        = field(default_factory=dict)
    run_summary     : dict        = field(default_factory=dict)
    checkpoint      : dict        = field(default_factory=dict)
    overfit         : dict        = field(default_factory=dict)
    training_result : dict        = field(default_factory=dict)
    inference_dir   : Path | None = None
    metrics         : dict        = field(default_factory=dict)
    figures         : list[Path]  = field(default_factory=list)
    animations      : list[Path]  = field(default_factory=list)
    report_path     : Path | None = None

    @property
    def has_inference(self) -> bool:
        return self.inference_dir is not None


class TrialCollector:
    def __init__(self, run_dir: Path, logger: Logger) -> None:
        self.run_dir      = run_dir
        self.training_dir = run_dir / "training"
        self.pipeline_dir = run_dir / "pipeline"
        self.logger       = logger

    def collect(self) -> list[TrialRecord]:
        size_match       = self._load_json(self.pipeline_dir / "size_match.json") or {}
        overfit_results  = {r.get("model"): r for r in self._load_json(self.pipeline_dir / "overfit_results.json") or []}
        training_results = {r.get("name"):  r for r in self._load_json(self.pipeline_dir / "training_results.json") or []}

        if not self.training_dir.is_dir():
            self.logger.error(f"No training directory found at: {self.training_dir}")
            return []

        records = []
        for trial_dir in sorted(d for d in self.training_dir.iterdir() if d.is_dir()):
            record = TrialRecord(name=trial_dir.name, run_dir=trial_dir)

            record.size_match      = size_match.get(trial_dir.name, {})
            record.trainer_config  = self._load_json(trial_dir / "docs" / "trainer_config.json") or {}
            record.run_summary     = self._load_json(trial_dir / "meta" / "run_summary.json") or {}
            record.overfit         = overfit_results.get(trial_dir.name, {})
            record.training_result = training_results.get(trial_dir.name, {})
            record.parameters      = self._parse_parameters(trial_dir, record.size_match)
            record.checkpoint      = self._read_checkpoint(trial_dir)

            self._attach_inference(record)

            status = f"inference {record.inference_dir.name}" if record.has_inference else "no inference"
            self.logger.info(f"{record.name:<22} {status}")

            records.append(record)

        return records

    def _load_json(self, path: Path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _parse_parameters(self, trial_dir: Path, size_match: dict) -> int | None:
        summary_path = trial_dir / "docs" / "model_summary.md"

        if summary_path.exists():
            match = _TOTAL_PARAMS_PATTERN.search(summary_path.read_text(encoding="utf-8", errors="ignore"))
            if match:
                return int(match.group(1).replace(",", ""))

        return size_match.get("parameters")

    def _read_checkpoint(self, trial_dir: Path) -> dict:
        import torch

        checkpoint_path = next(trial_dir.rglob("best_model.pt"), None)
        if checkpoint_path is None:
            return {}

        try:
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception:
            return {}

        info = {key: checkpoint.get(key) for key in _CHECKPOINT_KEYS}
        info["n_train_epochs"] = len(checkpoint.get("train_losses") or [])
        info["n_val_epochs"]   = len(checkpoint.get("val_losses") or [])

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
        record.metrics       = self._load_json(inference_dir / "metrics.json") or {}
        record.figures       = sorted((inference_dir / "figures").glob("*.png")) if (inference_dir / "figures").is_dir() else []
        record.animations    = sorted((inference_dir / "animations").glob("*.gif")) if (inference_dir / "animations").is_dir() else []

        report_path = inference_dir / "report.md"
        if report_path.exists():
            record.report_path = report_path
