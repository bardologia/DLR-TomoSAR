from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from configuration.benchmark_config import BenchmarkConfig
from pipelines.benchmark_pipeline.size_matcher import SizeMatcher, SizeMatchResult
from tools.logger import Logger


class SizeMatchStage:
    def __init__(self, config: BenchmarkConfig, run_tag: str, models: list[str], logger: Logger) -> None:
        self.config  = config
        self.run_tag = run_tag
        self.models  = models
        self.logger  = logger

        self.run_dir      = Path(config.paths.log_base_dir) / run_tag
        self.records_path = self.run_dir / "pipeline" / "size_match.json"
        self.report_path  = self.run_dir / "pipeline" / "size_match_report.md"

    def run(self) -> dict:
        self.logger.section("Capacity matching")
        self.logger.kv_table({
            "Reference model" : self.config.size_match.reference_model,
            "Tolerance"       : f"{100.0 * self.config.size_match.tolerance:.2f} %",
            "Max iterations"  : self.config.size_match.max_iterations,
            "In channels"     : self.config.size_match.in_channels,
            "Out channels"    : self.config.n_gaussians * 3,
        }, title="Configuration")

        cached = self._load_cached()
        if cached is not None:
            self.logger.info(f"Cached size match reused from: {self.records_path}")
            return cached

        matcher   = SizeMatcher(config=self.config, logger=self.logger)
        reference = self.config.size_match.reference_model
        target    = matcher.reference_count()

        self.logger.info(f"Reference '{reference}': {target:,} parameters")

        records = {reference: self._reference_record(reference, target)}

        for model_name in self.models:
            if model_name == reference:
                continue

            result               = matcher.match(model_name, target)
            records[model_name]  = asdict(result)

            self.logger.info(f"{model_name:<18} {result.parameters:>14,}  Δ {result.deviation_pct:+7.3f} %  (scale {result.scale:.4f}, {result.iterations} iterations)")

        self._write_records(records)
        self._write_report(records, target)

        return records

    def _reference_record(self, reference: str, target: int) -> dict:
        return asdict(SizeMatchResult(
            model         = reference,
            scale         = 1.0,
            overrides     = {},
            parameters    = target,
            target        = target,
            deviation_pct = 0.0,
            iterations    = 0,
        ))

    def _load_cached(self) -> dict | None:
        if not self.config.resume or not self.records_path.exists():
            return None

        try:
            with open(self.records_path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            return None

        if all(m in records for m in self.models):
            return records

        return None

    def _write_records(self, records: dict) -> None:
        self.records_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.records_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    def _write_report(self, records: dict, target: int) -> None:
        lines = [
            "# Capacity Matching Report",
            f"\n_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  —  run tag `{self.run_tag}`_\n",
            f"Reference model `{self.config.size_match.reference_model}` at **{target:,}** parameters.",
            f"Counting performed with {self.config.size_match.in_channels} input channels, {self.config.n_gaussians * 3} output channels, image size {self.config.training.patch_size[0]}.\n",
            "## Matched Widths\n",
            "| Model | Scale | Scaled attributes | Parameters | Δ vs reference | Iterations |",
            "| --- | --- | --- | --- | --- | --- |",
        ]

        for model_name in sorted(records.keys()):
            record    = records[model_name]
            overrides = ", ".join(f"`{k}={v}`" for k, v in record["overrides"].items()) or "_(defaults)_"
            lines.append(f"| `{model_name}` | {record['scale']:.4f} | {overrides} | {record['parameters']:,} | {record['deviation_pct']:+.3f} % | {record['iterations']} |")

        lines.append("")

        self.report_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Report written to: {self.report_path}")
