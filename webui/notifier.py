from __future__ import annotations

import json
import re
import socket
import threading
import urllib.error
import urllib.request
from datetime import datetime

from web_logger import WebLogger


class JobNotifier:

    SETTINGS_NAME  = "notifier.json"
    TOPIC_PATTERN  = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
    SEND_TIMEOUT_S = 10.0
    DEFAULTS       = {"enabled": False, "topic": "", "server": "https://ntfy.sh", "min_runtime_s": 60.0}

    def __init__(self, paths, logger: WebLogger) -> None:
        self.paths    = paths
        self.logger   = logger
        self.lock     = threading.Lock()
        self.path     = paths.logs_dir / self.SETTINGS_NAME
        self.settings = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return dict(self.DEFAULTS)
        loaded = json.loads(self.path.read_text())
        return {key: loaded[key] for key in self.DEFAULTS}

    def state(self) -> dict:
        with self.lock:
            return {**self.settings, "settings_path": str(self.path)}

    def configure(self, payload: dict) -> dict:
        enabled       = bool(payload.get("enabled", False))
        topic         = str(payload.get("topic", "")).strip()
        server        = str(payload.get("server", self.DEFAULTS["server"])).strip().rstrip("/")
        min_runtime_s = float(payload.get("min_runtime_s", self.DEFAULTS["min_runtime_s"]))

        if topic and not self.TOPIC_PATTERN.match(topic):
            return {"ok": False, "error": "topic may only contain letters, digits, '-' and '_' (max 64 chars)"}
        if enabled and not topic:
            return {"ok": False, "error": "set a topic before enabling notifications"}
        if not server.startswith(("http://", "https://")):
            return {"ok": False, "error": "server must be an http(s) URL"}
        if min_runtime_s < 0:
            return {"ok": False, "error": "min runtime must be non-negative"}

        with self.lock:
            self.settings = {"enabled": enabled, "topic": topic, "server": server, "min_runtime_s": min_runtime_s}
            self._persist()

        self.logger.ok(f"notifications {'enabled' if enabled else 'disabled'} (topic '{topic}', min runtime {min_runtime_s:.0f}s)")
        return {"ok": True, **self.state()}

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.settings, indent=2) + "\n")

    def test(self) -> dict:
        with self.lock:
            if not self.settings["topic"]:
                return {"ok": False, "error": "set a topic first"}

        error = self._send("DLR-TomoSAR test notification", f"the console on {socket.gethostname()} can reach you", "default")
        if error is not None:
            return {"ok": False, "error": error}
        return {"ok": True}

    def _runtime_s(self, record: dict) -> float:
        started = datetime.fromisoformat(record["started"])
        return max(0.0, (datetime.now() - started).total_seconds())

    def _runtime_label(self, seconds: float) -> str:
        minutes, secs  = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes:02d}m"
        if minutes:
            return f"{minutes}m {secs:02d}s"
        return f"{secs}s"

    def _describe(self, record: dict, runtime_s: float) -> tuple[str, str, str]:
        script = record.get("script") or "job"
        code   = record.get("exit_code")

        if record["status"] == "failed":
            title    = f"{script} FAILED" + (f" (exit {code})" if code is not None else "")
            priority = "high"
        else:
            title    = f"{script} finished" + ("" if code == 0 else " (exit status unknown)")
            priority = "default"

        body = f"runtime {self._runtime_label(runtime_s)} on {socket.gethostname()} (job {record['job_id']})"
        return title, body, priority

    def _send(self, title: str, body: str, priority: str) -> str | None:
        with self.lock:
            url = f"{self.settings['server']}/{self.settings['topic']}"

        request = urllib.request.Request(url, data=body.encode("utf-8"), method="POST")
        request.add_header("Title", title)
        request.add_header("Priority", priority)

        try:
            with urllib.request.urlopen(request, timeout=self.SEND_TIMEOUT_S) as response:
                response.read()
            return None
        except (urllib.error.URLError, OSError) as exc:
            return str(exc)

    def job_finished(self, record: dict) -> None:
        with self.lock:
            enabled       = self.settings["enabled"] and bool(self.settings["topic"])
            min_runtime_s = self.settings["min_runtime_s"]

        if not enabled:
            return
        if record.get("stopped"):
            return
        if record["status"] not in ("failed", "finished"):
            return

        runtime_s = self._runtime_s(record)
        if record["status"] == "finished" and runtime_s < min_runtime_s:
            return

        title, body, priority = self._describe(record, runtime_s)
        error = self._send(title, body, priority)
        if error is not None:
            self.logger.error(f"notification for job {record['job_id']} failed: {error}")
