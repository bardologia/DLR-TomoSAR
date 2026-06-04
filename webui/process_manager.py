from __future__ import annotations

import os
import queue
import subprocess
import threading
import uuid
from collections import deque
from datetime import datetime

from project_paths import ProjectPaths
from web_logger import WebLogger


class JobStream:

    def __init__(self) -> None:
        self.buffer      = deque(maxlen=4000)
        self.subscribers = []
        self.lock        = threading.Lock()

    def publish(self, event: dict) -> None:
        with self.lock:
            self.buffer.append(event)
            for sub in list(self.subscribers):
                try:
                    sub.put_nowait(event)
                except queue.Full:
                    pass

    def subscribe(self) -> queue.Queue:
        sub = queue.Queue(maxsize=8000)
        with self.lock:
            for event in self.buffer:
                sub.put_nowait(event)
            self.subscribers.append(sub)
        return sub

    def unsubscribe(self, sub: queue.Queue) -> None:
        with self.lock:
            if sub in self.subscribers:
                self.subscribers.remove(sub)


class ProcessManager:

    def __init__(self, paths: ProjectPaths, logger: WebLogger) -> None:
        self.paths   = paths
        self.logger  = logger
        self.jobs    = {}
        self.streams = {}
        self.lock    = threading.Lock()

    def launch(self, key: str, interpreter: str) -> dict:
        script = self.paths.main_dir / f"{key}.py"
        if not script.exists():
            return {"ok": False, "error": "script not found"}

        job_id = uuid.uuid4().hex[:12]
        stream = JobStream()

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"

        try:
            process = subprocess.Popen(
                [interpreter, "-u", str(script)],
                cwd                = str(self.paths.repo_root),
                stdout             = subprocess.PIPE,
                stderr             = subprocess.STDOUT,
                env                = env,
                text               = True,
                bufsize            = 1,
            )
        except OSError as exc:
            return {"ok": False, "error": str(exc)}

        record = {
            "job_id"      : job_id,
            "script"      : key,
            "command"     : f"{interpreter} -u main/{key}.py",
            "interpreter" : interpreter,
            "status"      : "running",
            "pid"         : process.pid,
            "started"     : datetime.now().isoformat(timespec="seconds"),
            "exit_code"   : None,
        }

        with self.lock:
            self.jobs[job_id]    = record
            self.streams[job_id] = stream

        self.logger.ok(f"launched {key} as job {job_id} (pid {process.pid})")
        stream.publish({"type": "status", "status": "running", "pid": process.pid})

        worker = threading.Thread(target=self._pump, args=(job_id, process, stream), daemon=True)
        worker.start()

        return {"ok": True, "job_id": job_id}

    def _pump(self, job_id: str, process: subprocess.Popen, stream: JobStream) -> None:
        line_no = 0
        for raw in process.stdout:
            line_no += 1
            stream.publish({"type": "line", "n": line_no, "text": raw.rstrip("\n")})

        process.wait()
        code = process.returncode

        with self.lock:
            record = self.jobs.get(job_id)
            if record is not None:
                record["status"]    = "finished" if code == 0 else "failed"
                record["exit_code"] = code

        verdict = "ok" if code == 0 else "error"
        self.logger.muted(f"job {job_id} exited with code {code}")
        stream.publish({"type": "status", "status": record["status"], "code": code, "verdict": verdict})
        stream.publish({"type": "end"})

    def stop(self, job_id: str) -> dict:
        with self.lock:
            record = self.jobs.get(job_id)
        if record is None:
            return {"ok": False, "error": "unknown job"}
        if record["status"] != "running":
            return {"ok": False, "error": "job is not running"}

        try:
            subprocess.run(["kill", "-TERM", str(record["pid"])], check=False)
        except OSError as exc:
            return {"ok": False, "error": str(exc)}

        self.logger.warning(f"stop requested for job {job_id}")
        return {"ok": True}

    def list_jobs(self) -> list[dict]:
        with self.lock:
            return sorted(self.jobs.values(), key=lambda r: r["started"], reverse=True)

    def get_stream(self, job_id: str) -> JobStream | None:
        with self.lock:
            return self.streams.get(job_id)
