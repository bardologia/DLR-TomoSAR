from __future__ import annotations

import multiprocessing as mp
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from pipelines.shared.io import FileIO
from tools.logger        import Logger


class ProcessPoolRunner:
    def __init__(self, logger: Logger, max_workers: int | None = None) -> None:
        self.logger      = logger
        self.max_workers = max_workers

    def run(self, jobs: Iterable[Any], worker_fn: Callable[[Any], Any]) -> list[tuple[Any, Any]]:
        job_list = list(jobs)
        workers  = max(1, min(self.max_workers, len(job_list))) if self.max_workers is not None else len(job_list)
        results  : list[tuple[Any, Any]] = []

        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
            futures = {executor.submit(worker_fn, job): job for job in job_list}

            try:
                for future in as_completed(futures):
                    job    = futures[future]
                    result = future.result()

                    results.append((job, result))
            except Exception:
                for future in futures:
                    future.cancel()
                raise

        return results


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

    def _install_signal_handlers(self) -> dict:
        previous = {
            signal.SIGTERM : signal.getsignal(signal.SIGTERM),
            signal.SIGINT  : signal.getsignal(signal.SIGINT),
        }

        signal.signal(signal.SIGTERM, self._terminate_running)
        signal.signal(signal.SIGINT,  self._terminate_running)

        return previous

    def _restore_signal_handlers(self, previous: dict) -> None:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

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


class QueuedTrainingStage(ExperimentStage):
    stage_subdir   : str = "training"
    worker_action  : str = "train"
    summary_title  : str = "Training summary"
    cached_logname : str = "worker.log"

    def __init__(self, config, entry_script: Path, run_tag: str, items: list[str], logger: Logger) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger, entry_script=entry_script)
        self.items        = items
        self.stage_dir    = self.run_dir / self.stage_subdir
        self.results_path = self.run_dir / "pipeline" / "training_results.json"

    def _config_kv(self) -> dict:
        return {
            "Items"      : len(self.items),
            "Epochs"     : self.config.training.epochs,
            "Batch size" : self.config.training.batch_size,
            "GPUs"       : self.config.gpus,
            "Stage dir"  : str(self.stage_dir),
        }

    def _worker_flag(self) -> str:
        return "--model"

    def _worker_value(self, item: str) -> str:
        return item

    def run(self) -> list[dict]:
        self.logger.section("Training queue")
        self.logger.kv_table(self._config_kv(), title="Configuration")

        cached  = [item for item in self.items if self._has_checkpoint(item)]
        pending = [item for item in self.items if item not in cached]

        for item in cached:
            self.logger.info(f"{item}: existing checkpoint reused")

        ran = []
        if pending:
            ran = self._run_queue([self._job(item) for item in pending])

        results = self._order_results(ran + [self._cached_result(item) for item in cached], self.items)

        self._write_results(results, self.results_path)
        self._log_summary(results)

        return results

    def _job(self, item: str) -> GpuJob:
        return GpuJob(
            name     = item,
            command  = [sys.executable, str(self.entry_script), "--worker", self.worker_action, self._worker_flag(), self._worker_value(item), "--run-tag", self.run_tag, "--run-dir", str(self.run_dir)],
            log_path = self.stage_dir / item / self.cached_logname,
        )

    def _has_checkpoint(self, item: str) -> bool:
        if not self.config.resume:
            return False

        item_dir = self.stage_dir / item
        if not item_dir.is_dir():
            return False

        return next(item_dir.rglob(self.config.inference.checkpoint_name), None) is not None

    def _cached_result(self, item: str) -> dict:
        return {
            "name"       : item,
            "gpu"        : None,
            "status"     : "DONE",
            "returncode" : 0,
            "duration_s" : None,
            "log_file"   : str(self.stage_dir / item / self.cached_logname),
        }

    def _log_summary(self, results: list[dict]) -> None:
        done   = [r for r in results if r["status"] == "DONE"]
        failed = [r for r in results if r["status"] != "DONE"]

        self.logger.subsection(self.summary_title)
        self.logger.kv_table({
            "Total"  : len(results),
            "Done"   : len(done),
            "Failed" : len(failed),
        }, title=f"{len(done)}/{len(results)} finished")

        self._log_failures(failed)


class QueuedInferenceStage(ExperimentStage):
    stage_subdir  : str = "training"
    worker_action : str = "infer"
    summary_title : str = "Inference summary"
    worker_logname: str = "inference_worker.log"

    def __init__(self, config, entry_script: Path, run_tag: str, items: list[str], logger: Logger) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger, entry_script=entry_script)
        self.items        = items
        self.stage_dir    = self.run_dir / self.stage_subdir
        self.results_path = self.run_dir / "pipeline" / "inference_results.json"

    def _config_kv(self) -> dict:
        return {
            "Items" : len(self.items),
            "Split" : self.config.inference.split,
            "GPUs"  : self.config.gpus,
        }

    def _worker_flag(self) -> str:
        return "--model"

    def _worker_value(self, item: str) -> str:
        return item

    def _include_item(self, item: str) -> bool:
        return True

    def run(self) -> list[dict]:
        self.logger.section("Batch inference")
        self.logger.kv_table(self._config_kv(), title="Configuration")

        eligible = [item for item in self.items if self._include_item(item)]
        skipped  = [item for item in eligible if not self._has_checkpoint(item)]
        cached   = [item for item in eligible if item not in skipped and self._has_inference(item)]
        pending  = [item for item in eligible if item not in skipped and item not in cached]

        for item in skipped:
            self.logger.warning(f"{item}: no checkpoint, inference skipped")

        for item in cached:
            self.logger.info(f"{item}: existing inference reused")

        ran = []
        if pending:
            ran = self._run_queue([self._job(item) for item in pending])

        results = ran + [self._static_result(item, "DONE") for item in cached] + [self._static_result(item, "SKIPPED") for item in skipped]
        results = self._order_results(results, eligible)

        self._write_results(results, self.results_path)
        self._log_summary(results)

        return results

    def _job(self, item: str) -> GpuJob:
        return GpuJob(
            name     = item,
            command  = [sys.executable, str(self.entry_script), "--worker", self.worker_action, self._worker_flag(), self._worker_value(item), "--run-tag", self.run_tag, "--run-dir", str(self.run_dir)],
            log_path = self.stage_dir / item / self.worker_logname,
        )

    def _has_checkpoint(self, item: str) -> bool:
        item_dir = self.stage_dir / item
        if not item_dir.is_dir():
            return False

        return next(item_dir.rglob(self.config.inference.checkpoint_name), None) is not None

    def _has_inference(self, item: str) -> bool:
        if not self.config.resume:
            return False

        inference_dir = self.stage_dir / item / "inference"
        if not inference_dir.is_dir():
            return False

        return next(inference_dir.glob("*/metrics.json"), None) is not None

    def _static_result(self, item: str, status: str) -> dict:
        return {
            "name"       : item,
            "gpu"        : None,
            "status"     : status,
            "returncode" : 0 if status == "DONE" else None,
            "duration_s" : None,
            "log_file"   : str(self.stage_dir / item / self.worker_logname),
        }

    def _log_summary(self, results: list[dict]) -> None:
        done   = [r for r in results if r["status"] == "DONE"]
        failed = [r for r in results if r["status"] == "FAILED"]

        self.logger.subsection(self.summary_title)
        self.logger.kv_table({
            "Total"   : len(results),
            "Done"    : len(done),
            "Failed"  : len(failed),
            "Skipped" : len(results) - len(done) - len(failed),
        }, title=f"{len(done)}/{len(results)} finished")

        self._log_failures(failed)
