from __future__ import annotations

from pathlib import Path

from configuration.patch_sweep     import PatchSweepConfig
from pipelines.patch_sweep.planner import PatchSweepPlanner
from pipelines.patch_sweep.report  import PatchSweepReport, SweepCollector
from tools                         import ExperimentStage, QueuedTrainingStage
from tools.monitoring.logger       import Logger
from tools.runtime.run_tag         import RunTag


class SweepTrainingStage(QueuedTrainingStage):
    def __init__(self, config: PatchSweepConfig, entry_script: Path, run_tag: str, planner: PatchSweepPlanner, logger: Logger) -> None:
        super().__init__(config=config, entry_script=entry_script, run_tag=run_tag, items=[unit.name for unit in planner.units()], logger=logger)
        self.planner = planner

    def _config_kv(self) -> dict:
        return {
            "Model"        : f"{self.config.backbone_name}-{self.config.backbone_head}",
            "Track counts" : sorted(self.config.track_counts),
            "Patch sizes"  : self.planner.patch_sizes(),
            "Units"        : len(self.items),
            "Epochs"       : self.config.training.epochs,
            "GPUs"         : self.config.gpus,
            "Stage dir"    : str(self.stage_dir),
        }

    def _worker_flag(self) -> str:
        return "--unit"

    def _has_checkpoint(self, item: str) -> bool:
        if not self.config.resume:
            return False

        item_dir = self.stage_dir / item
        if not item_dir.is_dir():
            return False

        return next(item_dir.rglob("best_model.pt"), None) is not None


class SweepReportStage(ExperimentStage):
    def __init__(self, config: PatchSweepConfig, run_tag: str, planner: PatchSweepPlanner, logger: Logger) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger)
        self.planner = planner

    def run(self) -> Path:
        self.logger.section("Patch sweep report")

        collector = SweepCollector(run_dir=self.run_dir, planner=self.planner, logger=self.logger)
        records   = collector.collect()

        out_dir = self.run_dir / "report" / RunTag.now()
        report  = PatchSweepReport(records=records, planner=self.planner, out_dir=out_dir, logger=self.logger)

        written = report.write_all()

        self.logger.subsection("Report written")
        for path in written:
            self.logger.info(f"{path}")

        return out_dir
