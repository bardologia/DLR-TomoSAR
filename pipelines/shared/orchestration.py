from __future__ import annotations

import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from pipelines.shared.io import FileIO
from tools.logger        import Logger


@dataclass
class GpuJob:
    name     : str
    command  : list[str]
    log_path : Path


@dataclass
class GpuJobResult:
    name       : str
    gpu        : int
    status     : str
    returncode : int
    duration_s : float
    log_file   : str


class GpuQueue:
    def __init__(self, gpus: list[int], logger: Logger, poll_interval_s: float = 5.0) -> None:
        self.gpus            = list(gpus)
        self.logger          = logger
        self.poll_interval_s = poll_interval_s

    def run(self, jobs: list[GpuJob]) -> list[GpuJobResult]:
        queue    = list(jobs)
        gpu_pool = list(self.gpus)
        running  : list[dict]         = []
        results  : list[GpuJobResult] = []

        while queue or running:
            self._reap(running, gpu_pool, results)

            while queue and gpu_pool:
                job    = queue.pop(0)
                gpu_id = gpu_pool.pop(0)
                running.append(self._launch(job, gpu_id))

            if queue or running:
                time.sleep(self.poll_interval_s)

        return results

    def _launch(self, job: GpuJob, gpu_id: int) -> dict:
        job.log_path.parent.mkdir(parents=True, exist_ok=True)

        command = job.command + ["--gpu", str(gpu_id)]
        log_fh  = open(job.log_path, "w", encoding="utf-8")
        process = subprocess.Popen(command, stdout=log_fh, stderr=log_fh)

        self.logger.info(f"[GPU {gpu_id}] started   {job.name}")

        return {"job": job, "gpu": gpu_id, "process": process, "log_fh": log_fh, "started": time.monotonic()}

    def _reap(self, running: list[dict], gpu_pool: list[int], results: list[GpuJobResult]) -> None:
        finished = [record for record in running if record["process"].poll() is not None]

        for record in finished:
            running.remove(record)
            record["log_fh"].close()

            job        = record["job"]
            returncode = record["process"].returncode
            status     = "DONE" if returncode == 0 else "FAILED"
            duration_s = time.monotonic() - record["started"]

            if returncode == 0:
                self.logger.info(f"[GPU {record['gpu']}] finished  {job.name}  ({duration_s / 60:.1f} min)")
            else:
                self.logger.error(f"[GPU {record['gpu']}] failed    {job.name}  (exit {returncode}, see {job.log_path})")

            results.append(GpuJobResult(
                name       = job.name,
                gpu        = record["gpu"],
                status     = status,
                returncode = returncode,
                duration_s = duration_s,
                log_file   = str(job.log_path),
            ))

            gpu_pool.append(record["gpu"])


class ExperimentStage:
    def __init__(self, config, run_tag: str, logger: Logger, entry_script: Path | None = None) -> None:
        self.config       = config
        self.entry_script = entry_script
        self.run_tag      = run_tag
        self.logger       = logger
        self.run_dir      = Path(config.paths.log_base_dir) / run_tag

    def _run_queue(self, jobs: list[GpuJob]) -> list[dict]:
        queue = GpuQueue(gpus=self.config.gpus, logger=self.logger, poll_interval_s=self.config.poll_interval_s)
        return [asdict(result) for result in queue.run(jobs)]

    def _order_results(self, results: list[dict], names: list[str]) -> list[dict]:
        by_name = {result["name"]: result for result in results}
        return [by_name[name] for name in names if name in by_name]

    def _write_results(self, results, path: Path) -> None:
        FileIO.save_json(results, path, indent=2)

    def _log_failures(self, failed: list[dict], name_key: str = "name") -> None:
        for result in failed:
            log_hint = f"  (see {result['log_file']})" if result.get("log_file") else ""
            self.logger.error(f"FAILED  {result[name_key]}{log_hint}")
