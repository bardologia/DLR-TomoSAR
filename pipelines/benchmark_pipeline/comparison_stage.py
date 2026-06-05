from __future__ import annotations

from datetime import datetime
from pathlib import Path

from configuration.benchmark_config import BenchmarkConfig
from pipelines.benchmark_pipeline.comparison_report import ComparisonReport
from pipelines.benchmark_pipeline.trial_collector import TrialCollector
from tools.logger import Logger


class ComparisonStage:
    def __init__(self, config: BenchmarkConfig, run_tag: str, logger: Logger) -> None:
        self.config  = config
        self.run_tag = run_tag
        self.logger  = logger
        self.run_dir = Path(config.paths.log_base_dir) / run_tag

    def run(self) -> Path:
        self.logger.section("Comparison reports")

        collector = TrialCollector(run_dir=self.run_dir, logger=self.logger)
        records   = collector.collect()

        out_dir = self.run_dir / "comparison" / datetime.now().strftime("%Y%m%d_%H%M%S")

        report = ComparisonReport(
            records         = records,
            out_dir         = out_dir,
            reference_model = self.config.size_match.reference_model,
            embed_images    = self.config.comparison.embed_images,
            logger          = self.logger,
        )

        written = report.write_all()

        self.logger.subsection("Reports written")
        for path in written:
            self.logger.info(f"{path}")

        return out_dir
