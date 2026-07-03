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
from datetime    import datetime

from project_paths import ProjectPaths
from web_logger    import WebLogger


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
    DETACHED_PID  = re.compile(r"detached .* as pid (\d+)")

    def __init__(self, paths: ProjectPaths, logger: WebLogger) -> None:
        self.paths   = paths
        self.logger  = logger
        self.jobs    = {}
        self.streams = {}
        self.lock    = threading.Lock()

    def launch(self, key: str, interpreter: str, overrides: dict | None = None, follow_up: str | None = None, detach: bool = False) -> dict:
        script = self.paths.script_entry(key)["path"]
        if not script.exists():
            return {"ok": False, "error": "script not found"}

        if follow_up and not detach:
            if not self.paths.has_script(follow_up):
                return {"ok": False, "error": f"unknown follow-up script '{follow_up}'"}
            if not self.paths.script_entry(follow_up)["path"].exists():
                return {"ok": False, "error": f"follow-up script '{follow_up}' not found"}

        record = self._make_record(key, interpreter, self._clean_overrides(overrides), detach)
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

        if follow_up and detach:
            self.logger.warning(f"follow-up {follow_up} ignored, {key} launched detached")
        elif follow_up:
            self._schedule(record, follow_up)

        return {"ok": True, "job_id": record["job_id"]}

    def _make_record(self, key: str, interpreter: str, overrides: dict, detach: bool = False) -> dict:
        return {
            "job_id"      : uuid.uuid4().hex[:12],
            "script"      : key,
            "command"     : self._render_command(interpreter, key, overrides, detach),
            "interpreter" : interpreter,
            "overrides"   : overrides,
            "detach"      : detach,
            "status"      : "pending",
            "pid"         : None,
            "started"     : datetime.now().isoformat(timespec="seconds"),
            "exit_code"   : None,
            "follow_of"   : None,
            "follow_up"   : None,
        }

    def _runtime_env(self, interpreter: str) -> dict:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["FORCE_COLOR"]      = "1"
        env["COLUMNS"]          = "120"
        env["LINES"]            = "32"
        env.setdefault("QT_QPA_PLATFORM", "offscreen")

        library_dir  = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(interpreter))), "lib")
        library_path = env.get("LD_LIBRARY_PATH", "")
        if library_dir not in library_path.split(":"):
            env["LD_LIBRARY_PATH"] = library_dir + (":" + library_path if library_path else "")

        return env

    def _start(self, record: dict, stream: JobStream) -> str | None:
        entry = self.paths.script_entry(record["script"])
        argv  = [record["interpreter"], "-u", str(entry["path"]), *entry["args"]]
        for path, value in record["overrides"].items():
            argv += [f"--{path}", value]
        if record.get("detach"):
            argv.append("--detach")

        env = self._runtime_env(record["interpreter"])

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
                raise ValueError(f"invalid override key '{path}'")
            cleaned[path] = str(value)
        return cleaned

    def _render_command(self, interpreter: str, key: str, overrides: dict, detach: bool = False) -> str:
        entry = self.paths.script_entry(key)
        parts = [interpreter, "-u", entry["rel"], *entry["args"]]
        for path, value in overrides.items():
            parts += [f"--{path}", shlex.quote(value)]
        if detach:
            parts.append("--detach")
        return " ".join(parts)

    def _pump(self, job_id: str, process: subprocess.Popen, stream: JobStream) -> None:
        fd      = process.stdout.fileno()
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        with self.lock:
            detached = bool(self.jobs.get(job_id, {}).get("detach"))
        head = ""

        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            text = decoder.decode(chunk)
            if text:
                stream.publish({"type": "chunk", "data": text})
                if detached and len(head) < 4096:
                    head += text

        tail = decoder.decode(b"", final=True)
        if tail:
            stream.publish({"type": "chunk", "data": tail})
            if detached and len(head) < 4096:
                head += tail

        process.wait()
        code = process.returncode

        detached_pid = self._parse_detached_pid(head) if detached else None
        if detached_pid and code == 0:
            self._adopt_detached(job_id, detached_pid, stream)
            return

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

    def _parse_detached_pid(self, text: str) -> int | None:
        match = self.DETACHED_PID.search(text)
        return int(match.group(1)) if match else None

    def _adopt_detached(self, job_id: str, pid: int, stream: JobStream) -> None:
        token = self._pid_start_token(pid)

        with self.lock:
            record = self.jobs.get(job_id)
            if record is None:
                return
            record["status"]           = "running"
            record["pid"]              = pid
            record["detached_running"] = True

        self.logger.ok(f"job {job_id} detached, now tracking live pid {pid}")
        stream.publish({"type": "status", "status": "running", "pid": pid, "detached": True})

        watcher = threading.Thread(target=self._watch_detached, args=(job_id, pid, token, stream), daemon=True)
        watcher.start()

    def _watch_detached(self, job_id: str, pid: int, token: tuple | None, stream: JobStream) -> None:
        while self._pid_alive(pid) and self._pid_start_token(pid) == token:
            time.sleep(2.0)
            with self.lock:
                if self.jobs.get(job_id) is None:
                    return

        with self.lock:
            record = self.jobs.get(job_id)
            if record is None:
                return
            record["status"]    = "finished"
            record["exit_code"] = None

        self.logger.muted(f"detached job {job_id} (pid {pid}) exited, exit status unknown")
        stream.publish({"type": "status", "status": "finished", "code": None, "verdict": "unknown"})
        stream.publish({"type": "end"})

    def _pid_alive(self, pid: int) -> bool:
        try:
            stat = open(f"/proc/{pid}/stat").read()
        except OSError:
            return False
        try:
            return stat[stat.rindex(")") + 2] != "Z"
        except (ValueError, IndexError):
            return False

    def _pid_start_token(self, pid: int) -> tuple | None:
        try:
            stat = open(f"/proc/{pid}/stat").read()
        except OSError:
            return None
        try:
            fields = stat[stat.rindex(")") + 2 :].split()
            return (pid, fields[19])
        except (ValueError, IndexError):
            return None

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


class ProcessNuke:

    def __init__(self, logger: WebLogger) -> None:
        self.logger = logger
        self.uid    = os.getuid()

    def _spared_pids(self) -> set[int]:
        spared = {0, 1}
        pid    = os.getpid()

        while pid > 1:
            spared.add(pid)
            pid = self._ppid(pid)

        return spared

    def _ppid(self, pid: int) -> int:
        try:
            stat   = open(f"/proc/{pid}/stat").read()
            fields = stat[stat.rindex(")") + 2 :].split()
            return int(fields[1])
        except (OSError, ValueError, IndexError):
            return 0

    def _targets(self, spared: set[int]) -> list[int]:
        return [pid for pid in self._user_pids() if pid not in spared]

    def _user_pids(self) -> list[int]:
        pids = []
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                if os.stat(f"/proc/{pid}").st_uid == self.uid:
                    pids.append(pid)
            except OSError:
                continue
        return pids

    def _terminate(self, targets: list[int], sig: signal.Signals) -> int:
        hit = 0
        for pid in targets:
            try:
                os.kill(pid, sig)
                hit += 1
            except (ProcessLookupError, PermissionError):
                continue
        return hit

    def _alive(self, targets: list[int]) -> list[int]:
        return [pid for pid in targets if os.path.exists(f"/proc/{pid}")]

    def nuke(self, grace: float = 4.0) -> dict:
        spared  = self._spared_pids()
        targets = self._targets(spared)

        if not targets:
            self.logger.muted("nuke requested, no processes to kill")
            return {"ok": True, "signalled": 0, "killed": 0}

        signalled = self._terminate(targets, signal.SIGTERM)
        self.logger.error(f"NUKE: SIGTERM sent to {signalled} of {len(targets)} user processes")

        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not self._alive(targets):
                return {"ok": True, "signalled": signalled, "killed": 0}
            time.sleep(0.25)

        stubborn = self._alive(targets)
        killed   = self._terminate(stubborn, signal.SIGKILL)
        self.logger.error(f"NUKE: SIGKILL sent to {killed} surviving processes")

        return {"ok": True, "signalled": signalled, "killed": killed}
