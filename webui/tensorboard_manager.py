from __future__ import annotations

import socket
import subprocess
import threading
import time
import urllib.request
import uuid
from datetime import datetime

from project_paths import ProjectPaths
from web_logger import WebLogger


class TensorboardManager:

    TRAINING_LOGDIRS = {
        "single_train" : ("logdir",),
        "batch_train"  : ("base.logdir",),
    }

    BASE_PORT         = 6007
    STARTUP_TIMEOUT_S = 90.0
    PROBE_INTERVAL_S  = 0.5

    def __init__(self, paths: ProjectPaths, logger: WebLogger) -> None:
        self.paths     = paths
        self.logger    = logger
        self.instances = {}
        self.lock      = threading.Lock()

    def logdir_keys(self, script_key: str) -> tuple | None:
        return self.TRAINING_LOGDIRS.get(script_key)

    def ensure(self, logdir: str, interpreter: str) -> dict:
        logdir = str(logdir)

        with self.lock:
            for record in self.instances.values():
                if record["logdir"] == logdir and record["status"] in ("starting", "running") and record["process"].poll() is None:
                    return {"ok": True, **self._view(record)}

        port  = self._free_port()
        tb_id = uuid.uuid4().hex[:8]

        argv = [
            interpreter, "-m", "tensorboard.main",
            "--logdir",          logdir,
            "--host",            "127.0.0.1",
            "--port",            str(port),
            "--path_prefix",     f"/tb/{tb_id}",
            "--reload_interval", "30",
        ]

        try:
            process = subprocess.Popen(argv, cwd=str(self.paths.repo_root), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            return {"ok": False, "error": str(exc)}

        record = {
            "id"      : tb_id,
            "logdir"  : logdir,
            "port"    : port,
            "pid"     : process.pid,
            "status"  : "starting",
            "started" : datetime.now().isoformat(timespec="seconds"),
            "process" : process,
        }

        with self.lock:
            self.instances[tb_id] = record

        self.logger.ok(f"tensorboard {tb_id} starting on port {port} for {logdir}")

        threading.Thread(target=self._probe, args=(record,), daemon=True).start()

        return {"ok": True, **self._view(record)}

    def get(self, tb_id: str) -> dict | None:
        with self.lock:
            return self.instances.get(tb_id)

    def list_instances(self) -> list[dict]:
        with self.lock:
            records = list(self.instances.values())

        for record in records:
            if record["status"] == "running" and record["process"].poll() is not None:
                record["status"] = "stopped"

        return [self._view(r) for r in sorted(records, key=lambda r: r["started"], reverse=True)]

    def stop(self, tb_id: str) -> dict:
        with self.lock:
            record = self.instances.get(tb_id)

        if record is None:
            return {"ok": False, "error": "unknown tensorboard instance"}

        if record["process"].poll() is None:
            record["process"].terminate()

        record["status"] = "stopped"
        self.logger.warning(f"tensorboard {tb_id} stopped")
        return {"ok": True}

    def stop_all(self) -> None:
        with self.lock:
            records = list(self.instances.values())

        for record in records:
            if record["process"].poll() is None:
                record["process"].terminate()
            record["status"] = "stopped"

    def _probe(self, record: dict) -> None:
        url      = f"http://127.0.0.1:{record['port']}/tb/{record['id']}/"
        deadline = time.monotonic() + self.STARTUP_TIMEOUT_S

        while time.monotonic() < deadline:
            if record["process"].poll() is not None:
                record["status"] = "failed"
                self.logger.error(f"tensorboard {record['id']} exited during startup")
                return

            try:
                with urllib.request.urlopen(url, timeout=2):
                    record["status"] = "running"
                    self.logger.ok(f"tensorboard {record['id']} ready at /tb/{record['id']}/")
                    return
            except OSError:
                time.sleep(self.PROBE_INTERVAL_S)

        record["status"] = "failed"
        if record["process"].poll() is None:
            record["process"].terminate()
        self.logger.error(f"tensorboard {record['id']} failed to start within {self.STARTUP_TIMEOUT_S:.0f}s")

    def _view(self, record: dict) -> dict:
        return {
            "id"      : record["id"],
            "logdir"  : record["logdir"],
            "port"    : record["port"],
            "pid"     : record["pid"],
            "status"  : record["status"],
            "started" : record["started"],
            "url"     : f"/tb/{record['id']}/",
        }

    @classmethod
    def _free_port(cls) -> int:
        for port in range(cls.BASE_PORT, cls.BASE_PORT + 100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]
