from __future__ import annotations

import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib     import Path

from tools.monitoring.logger import Logger


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
    def __init__(self, gpus: list[int], logger: Logger, poll_interval_s: float = 5.0, handle_signals: bool = True, terminate_deadline_s: float = 30.0) -> None:
        self.gpus                 = list(gpus)
        self.logger               = logger
        self.poll_interval_s      = poll_interval_s
        self.handle_signals       = handle_signals
        self.terminate_deadline_s = terminate_deadline_s
        self.running              : list[dict] = []

    def _install_signal_handlers(self) -> dict:
        previous = {
            signal.SIGTERM : signal.getsignal(signal.SIGTERM),
            signal.SIGINT  : signal.getsignal(signal.SIGINT),
        }

        signal.signal(signal.SIGTERM, self._terminate_running)
        signal.signal(signal.SIGINT,  self._terminate_running)

        return previous

    def _terminate_running(self, signum, frame) -> None:
        for record in self.running:
            if record["process"].poll() is None:
                record["process"].terminate()

        deadline = time.time() + self.terminate_deadline_s
        for record in self.running:
            try:
                record["process"].wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                record["process"].kill()

            record["log_fh"].close()

        self.logger.warning(f"Received signal {signum} — terminated {len(self.running)} workers, resume with --resume")

        sys.exit(128 + signum)

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

    def _launch(self, job: GpuJob, gpu_id: int) -> dict:
        job.log_path.parent.mkdir(parents=True, exist_ok=True)

        command = job.command + ["--gpu", str(gpu_id)]
        log_fh  = open(job.log_path, "w", encoding="utf-8")
        process = subprocess.Popen(command, stdout=log_fh, stderr=log_fh)

        self.logger.info(f"[GPU {gpu_id}] started   {job.name}")

        return {"job": job, "gpu": gpu_id, "process": process, "log_fh": log_fh, "started": time.monotonic()}

    def _restore_signal_handlers(self, previous: dict) -> None:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    def run(self, jobs: list[GpuJob]) -> list[GpuJobResult]:
        queue    = list(jobs)
        gpu_pool = list(self.gpus)
        results  : list[GpuJobResult] = []

        self.running = []

        previous_handlers = self._install_signal_handlers() if self.handle_signals else {}

        try:
            while queue or self.running:
                self._reap(self.running, gpu_pool, results)

                while queue and gpu_pool:
                    job    = queue.pop(0)
                    gpu_id = gpu_pool.pop(0)
                    self.running.append(self._launch(job, gpu_id))

                if queue or self.running:
                    time.sleep(self.poll_interval_s)
        finally:
            self._restore_signal_handlers(previous_handlers)

        return results
