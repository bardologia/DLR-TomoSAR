from __future__ import annotations

import sys
from datetime import datetime
from pathlib  import Path

from configuration.inference       import InferenceEntryConfig
from tools.orchestration.gpu_queue import GpuJob, GpuJobResult, GpuQueue
from tools.runtime.config_cli      import ConfigCli
from tools.monitoring.logger       import Logger


class InferenceScheduler:
    def __init__(self, config: InferenceEntryConfig, entry_script: Path) -> None:
        self.config       = config
        self.entry_script = entry_script
        self.logs_dirs    = [Path(d) for d in config.logs_dirs]
        self.work_dir     = Path("logs") / "inference" / datetime.now().strftime("%Y%m%d_%H%M%S")

    def _roots(self, logger: Logger) -> list[Path]:
        present = [root for root in self.logs_dirs if root.is_dir()]
        missing = [root for root in self.logs_dirs if not root.is_dir()]

        if missing:
            logger.subsection(f"Skipping {len(missing)} absent run root(s): {', '.join(str(root) for root in missing)}")

        return present

    def _run_dirs(self, logger: Logger) -> list[Path]:
        roots = self._roots(logger)

        if self.config.run_filter:
            selected = [root / name for name in self.config.run_filter for root in roots if (root / name).is_dir()]
            missing  = [name for name in self.config.run_filter if not any((root / name).is_dir() for root in roots)]
            if missing:
                raise FileNotFoundError(f"No run directory named {missing} found under any of {[str(root) for root in roots]}")
            return sorted(selected)

        return sorted(directory for root in roots for directory in root.iterdir() if directory.is_dir())

    def _worker_config(self) -> Path:
        path = self.work_dir / "resolved_config.json"
        return ConfigCli.save_resolved(self.config, path)

    def _jobs(self, run_dirs: list[Path], config_path: Path) -> list[GpuJob]:
        jobs = []
        for run_dir in run_dirs:
            command = [sys.executable, str(self.entry_script), "--worker", "--run-dir", str(run_dir), "--config", str(config_path)]
            jobs.append(GpuJob(name=f"{run_dir.parent.name}/{run_dir.name}", command=command, log_path=self.work_dir / f"{run_dir.parent.name}__{run_dir.name}.out"))
        return jobs

    def _report(self, logger: Logger, results: list[GpuJobResult]) -> None:
        for result in results:
            logger.info(f"{result.name}  :  GPU {result.gpu}  {result.status}  ({result.duration_s / 60:.1f} min)  {result.log_file}")

        failed = [result for result in results if result.status != "DONE"]
        if failed:
            logger.error(f"{len(failed)} of {len(results)} runs failed; see the per-run logs under {self.work_dir}")

    def run(self) -> list[GpuJobResult]:
        with Logger(log_dir=str(self.work_dir), name="inference") as logger:
            logger.section("Inference")

            run_dirs    = self._run_dirs(logger)
            config_path = self._worker_config()
            jobs        = self._jobs(run_dirs, config_path)

            logger.kv_table({
                "Runs"      : len(run_dirs),
                "GPUs"      : self.config.gpus,
                "Run roots" : [str(root) for root in self.logs_dirs],
                "Filter"    : self.config.run_filter or "all run directories",
                "Workers"   : str(self.work_dir),
            }, title="Configuration")

            queue   = GpuQueue(gpus=self.config.gpus, logger=logger, poll_interval_s=self.config.poll_interval_s)
            results = queue.run(jobs)

            self._report(logger, results)

        return results
