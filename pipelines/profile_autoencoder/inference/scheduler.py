from __future__ import annotations

import sys
from datetime import datetime
from pathlib  import Path

from configuration.inference.profile_autoencoder import ProfileAeInferenceEntryConfig
from tools.orchestration.gpu_queue               import GpuJob, GpuJobResult, GpuQueue
from tools.runtime.config_cli                    import ConfigCli
from tools.monitoring.logger                     import Logger


class ProfileAeInferenceScheduler:
    def __init__(self, config: ProfileAeInferenceEntryConfig, entry_script: Path) -> None:
        self.config       = config
        self.entry_script = entry_script
        self.logs_dir     = Path(config.logs_dir)
        self.work_dir     = Path("logs") / "inference_profile_ae" / datetime.now().strftime("%Y%m%d_%H%M%S")

    def _run_dirs(self) -> list[Path]:
        if self.config.run_filter:
            return sorted(self.logs_dir / name for name in self.config.run_filter)
        return sorted(d for d in self.logs_dir.iterdir() if d.is_dir())

    def _worker_config(self) -> Path:
        path = self.work_dir / "resolved_config.json"
        return ConfigCli.save_resolved(self.config, path)

    def _jobs(self, run_dirs: list[Path], config_path: Path) -> list[GpuJob]:
        jobs = []
        for run_dir in run_dirs:
            command = [sys.executable, str(self.entry_script), "--worker", "--run-dir", str(run_dir), "--config", str(config_path)]
            jobs.append(GpuJob(name=run_dir.name, command=command, log_path=self.work_dir / f"{run_dir.name}.out"))
        return jobs

    def _report(self, logger: Logger, results: list[GpuJobResult]) -> None:
        for result in results:
            logger.info(f"{result.name}  :  GPU {result.gpu}  {result.status}  ({result.duration_s / 60:.1f} min)  {result.log_file}")

        failed = [result for result in results if result.status != "DONE"]
        if failed:
            logger.error(f"{len(failed)} of {len(results)} runs failed; see the per-run logs under {self.work_dir}")

    def run(self) -> list[GpuJobResult]:
        run_dirs    = self._run_dirs()
        config_path = self._worker_config()
        jobs        = self._jobs(run_dirs, config_path)

        with Logger(log_dir=str(self.logs_dir), name="profile_ae_inference") as logger:
            logger.section("Profile Autoencoder Inference")
            logger.kv_table({
                "Runs"     : len(run_dirs),
                "GPUs"     : self.config.gpus,
                "Logs dir" : str(self.logs_dir),
                "Filter"   : self.config.run_filter or "all run directories",
                "Workers"  : str(self.work_dir),
            }, title="Configuration")

            queue   = GpuQueue(gpus=self.config.gpus, logger=logger, poll_interval_s=self.config.poll_interval_s)
            results = queue.run(jobs)

            self._report(logger, results)

        return results
