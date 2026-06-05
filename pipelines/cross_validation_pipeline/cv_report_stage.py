from __future__ import annotations

from datetime import datetime
from pathlib import Path

from configuration.cross_validation_config import CrossValidationConfig
from pipelines.cross_validation_pipeline.cv_report import CrossValidationReport
from pipelines.cross_validation_pipeline.fold_collector import FoldCollector
from pipelines.cross_validation_pipeline.fold_planner import FoldPlanner
from tools.logger import Logger


class CrossValidationReportStage:
    def __init__(self, config: CrossValidationConfig, run_tag: str, planner: FoldPlanner, logger: Logger) -> None:
        self.config  = config
        self.run_tag = run_tag
        self.planner = planner
        self.logger  = logger
        self.run_dir = Path(config.paths.log_base_dir) / run_tag

    def run(self) -> Path:
        self.logger.section("Cross-validation reports")

        collector = FoldCollector(run_dir=self.run_dir, splits=self.config.inference_splits, logger=self.logger)
        base_records, records_by_split = collector.collect_by_split()

        out_dir = self.run_dir / "reports" / datetime.now().strftime("%Y%m%d_%H%M%S")

        report = CrossValidationReport(
            base_records     = base_records,
            records_by_split = records_by_split,
            planner          = self.planner,
            out_dir          = out_dir,
            model_name       = self.config.model_name,
            embed_images     = self.config.comparison.embed_images,
            logger           = self.logger,
        )

        written = report.write_all()

        self.logger.subsection("Reports written")
        for path in written:
            self.logger.info(f"{path}")

        return out_dir
