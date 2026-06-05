from __future__ import annotations

import codecs
import os
import queue
import re
import shlex
import subprocess
import threading
import time
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

    OVERRIDE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")

    def __init__(self, paths: ProjectPaths, logger: WebLogger) -> None:
        self.paths   = paths
        self.logger  = logger
        self.jobs    = {}
        self.streams = {}
        self.lock    = threading.Lock()

    def launch(self, key: str, interpreter: str, overrides: dict | None = None) -> dict:
        script = self.paths.main_dir / f"{key}.py"
        if not script.exists():
            return {"ok": False, "error": "script not found"}

        overrides = self._clean_overrides(overrides)
        argv      = [interpreter, "-u", str(script)]
        for path, value in overrides.items():
            argv += [f"--{path}", value]

        job_id = uuid.uuid4().hex[:12]
        stream = JobStream()

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["FORCE_COLOR"]      = "1"
        env["COLUMNS"]          = "120"
        env["LINES"]            = "32"

        try:
            process = subprocess.Popen(
                argv,
                cwd    = str(self.paths.repo_root),
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                env    = env,
            )
        except OSError as exc:
            return {"ok": False, "error": str(exc)}

        record = {
            "job_id"      : job_id,
            "script"      : key,
            "command"     : self._render_command(interpreter, key, overrides),
            "interpreter" : interpreter,
            "overrides"   : overrides,
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

    def _clean_overrides(self, overrides: dict | None) -> dict:
        cleaned = {}
        for path, value in (overrides or {}).items():
            if not isinstance(path, str) or not self.OVERRIDE_NAME.match(path):
                continue
            cleaned[path] = str(value)
        return cleaned

    def _render_command(self, interpreter: str, key: str, overrides: dict) -> str:
        parts = [interpreter, "-u", f"main/{key}.py"]
        for path, value in overrides.items():
            parts += [f"--{path}", shlex.quote(value)]
        return " ".join(parts)

    def _pump(self, job_id: str, process: subprocess.Popen, stream: JobStream) -> None:
        fd      = process.stdout.fileno()
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            text = decoder.decode(chunk)
            if text:
                stream.publish({"type": "chunk", "data": text})

        tail = decoder.decode(b"", final=True)
        if tail:
            stream.publish({"type": "chunk", "data": tail})

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

    def stop_all(self, grace: float = 8.0) -> int:
        with self.lock:
            running = [dict(r) for r in self.jobs.values() if r["status"] == "running"]

        if not running:
            return 0

        for record in running:
            subprocess.run(["kill", "-TERM", str(record["pid"])], check=False)
            self.logger.warning(f"watchdog stop for job {record['job_id']} (pid {record['pid']})")

        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            with self.lock:
                alive = [r for r in running if self.jobs[r["job_id"]]["status"] == "running"]
            if not alive:
                return len(running)
            time.sleep(0.5)

        with self.lock:
            stubborn = [r for r in running if self.jobs[r["job_id"]]["status"] == "running"]
        for record in stubborn:
            subprocess.run(["kill", "-KILL", str(record["pid"])], check=False)
            self.logger.error(f"force kill for job {record['job_id']} (pid {record['pid']})")

        return len(running)

    def list_jobs(self) -> list[dict]:
        with self.lock:
            return sorted(self.jobs.values(), key=lambda r: r["started"], reverse=True)

    def get_stream(self, job_id: str) -> JobStream | None:
        with self.lock:
            return self.streams.get(job_id)
