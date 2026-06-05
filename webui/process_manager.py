from __future__ import annotations

import codecs
import os
import queue
import re
import shlex
import signal
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

    def launch(self, key: str, interpreter: str, overrides: dict | None = None, follow_up: str | None = None) -> dict:
        script = self.paths.main_dir / f"{key}.py"
        if not script.exists():
            return {"ok": False, "error": "script not found"}

        record = self._make_record(key, interpreter, self._clean_overrides(overrides))
        stream = JobStream()

        with self.lock:
            self.jobs[record["job_id"]]    = record
            self.streams[record["job_id"]] = stream

        error = self._start(record, stream)
        if error is not None:
            with self.lock:
                self.jobs.pop(record["job_id"], None)
                self.streams.pop(record["job_id"], None)
            return {"ok": False, "error": error}

        if follow_up:
            self._schedule(record, follow_up)

        return {"ok": True, "job_id": record["job_id"]}

    def _make_record(self, key: str, interpreter: str, overrides: dict) -> dict:
        return {
            "job_id"      : uuid.uuid4().hex[:12],
            "script"      : key,
            "command"     : self._render_command(interpreter, key, overrides),
            "interpreter" : interpreter,
            "overrides"   : overrides,
            "status"      : "pending",
            "pid"         : None,
            "started"     : datetime.now().isoformat(timespec="seconds"),
            "exit_code"   : None,
            "follow_of"   : None,
            "follow_up"   : None,
        }

    def _start(self, record: dict, stream: JobStream) -> str | None:
        script = self.paths.main_dir / f"{record['script']}.py"
        argv   = [record["interpreter"], "-u", str(script)]
        for path, value in record["overrides"].items():
            argv += [f"--{path}", value]

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["FORCE_COLOR"]      = "1"
        env["COLUMNS"]          = "120"
        env["LINES"]            = "32"

        try:
            process = subprocess.Popen(
                argv,
                cwd               = str(self.paths.repo_root),
                stdout            = subprocess.PIPE,
                stderr            = subprocess.STDOUT,
                env               = env,
                start_new_session = True,
            )
        except OSError as exc:
            return str(exc)

        with self.lock:
            record["status"]  = "running"
            record["pid"]     = process.pid
            record["started"] = datetime.now().isoformat(timespec="seconds")

        self.logger.ok(f"launched {record['script']} as job {record['job_id']} (pid {process.pid})")
        stream.publish({"type": "status", "status": "running", "pid": process.pid})

        worker = threading.Thread(target=self._pump, args=(record["job_id"], process, stream), daemon=True)
        worker.start()
        return None

    def _schedule(self, parent: dict, key: str) -> None:
        script = self.paths.main_dir / f"{key}.py"
        if not script.exists():
            self.logger.warning(f"follow-up {key} skipped, script not found")
            return

        record              = self._make_record(key, parent["interpreter"], {})
        record["status"]    = "scheduled"
        record["follow_of"] = parent["job_id"]

        stream = JobStream()
        stream.publish({"type": "status", "status": "scheduled", "after": parent["script"]})

        with self.lock:
            parent["follow_up"]            = record["job_id"]
            self.jobs[record["job_id"]]    = record
            self.streams[record["job_id"]] = stream

        self.logger.muted(f"scheduled {key} as job {record['job_id']} after {parent['script']} ({parent['job_id']})")

    def _resolve_follow_up(self, follow_id: str, code: int) -> None:
        with self.lock:
            record = self.jobs.get(follow_id)
            stream = self.streams.get(follow_id)
        if record is None or stream is None or record["status"] != "scheduled":
            return

        if code == 0:
            error = self._start(record, stream)
            if error is None:
                return
            with self.lock:
                record["status"] = "failed"
            stream.publish({"type": "status", "status": "failed", "code": None, "verdict": "error"})
            stream.publish({"type": "end"})
            self.logger.error(f"scheduled job {follow_id} failed to start: {error}")
            return

        with self.lock:
            record["status"] = "cancelled"
        stream.publish({"type": "status", "status": "cancelled", "code": None})
        stream.publish({"type": "end"})
        self.logger.warning(f"scheduled job {follow_id} cancelled, parent exited with code {code}")

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
            follow_id = record["follow_up"] if record else None

        if follow_id:
            self._resolve_follow_up(follow_id, code)

        verdict = "ok" if code == 0 else "error"
        self.logger.muted(f"job {job_id} exited with code {code}")
        stream.publish({"type": "status", "status": record["status"], "code": code, "verdict": verdict})
        stream.publish({"type": "end"})

    def stop(self, job_id: str) -> dict:
        with self.lock:
            record = self.jobs.get(job_id)
            stream = self.streams.get(job_id)
        if record is None:
            return {"ok": False, "error": "unknown job"}

        if record["status"] == "scheduled":
            with self.lock:
                record["status"] = "cancelled"
            if stream is not None:
                stream.publish({"type": "status", "status": "cancelled", "code": None})
                stream.publish({"type": "end"})
            self.logger.warning(f"scheduled job {job_id} cancelled by user")
            return {"ok": True}

        if record["status"] != "running":
            return {"ok": False, "error": "job is not running"}

        self._signal_group(record["pid"], signal.SIGTERM)
        self.logger.warning(f"stop requested for job {job_id}")
        return {"ok": True}

    def stop_all(self, grace: float = 8.0) -> int:
        with self.lock:
            running = [dict(r) for r in self.jobs.values() if r["status"] == "running"]

        if not running:
            return 0

        for record in running:
            self._signal_group(record["pid"], signal.SIGTERM)
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
            self._signal_group(record["pid"], signal.SIGKILL)
            self.logger.error(f"force kill for job {record['job_id']} (pid {record['pid']})")

        return len(running)

    @staticmethod
    def _signal_group(pid: int, sig: signal.Signals) -> None:
        try:
            os.killpg(pid, sig)
            return
        except (ProcessLookupError, PermissionError):
            pass

        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def list_jobs(self) -> list[dict]:
        with self.lock:
            return sorted(self.jobs.values(), key=lambda r: r["started"], reverse=True)

    def get_stream(self, job_id: str) -> JobStream | None:
        with self.lock:
            return self.streams.get(job_id)
