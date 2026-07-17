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
    DEFAULTS       = {"enabled": False, "topic": "", "server": "https://ntfy.sh"}

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
        enabled = bool(payload.get("enabled", False))
        topic   = str(payload.get("topic", "")).strip()
        server  = str(payload.get("server", self.DEFAULTS["server"])).strip().rstrip("/")

        if topic and not self.TOPIC_PATTERN.match(topic):
            return {"ok": False, "error": "topic may only contain letters, digits, '-' and '_' (max 64 chars)"}
        if enabled and not topic:
            return {"ok": False, "error": "set a topic before enabling notifications"}
        if not server.startswith(("http://", "https://")):
            return {"ok": False, "error": "server must be an http(s) URL"}

        with self.lock:
            self.settings = {"enabled": enabled, "topic": topic, "server": server}
            self._persist()

        self.logger.ok(f"notifications {'enabled' if enabled else 'disabled'} (topic '{topic}')")
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

    def runtime_s(self, record: dict) -> float:
        started = datetime.fromisoformat(record["started"])
        return max(0.0, (datetime.now() - started).total_seconds())

    def runtime_label(self, seconds: float) -> str:
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

        if record.get("stopped"):
            title    = f"{script} stopped" + (f" (exit {code})" if code is not None else "")
            priority = "default"
        elif record["status"] == "failed":
            title    = f"{script} FAILED" + (f" (exit {code})" if code is not None else "")
            priority = "high"
        else:
            title    = f"{script} finished" + ("" if code == 0 else " (exit status unknown)")
            priority = "default"

        body = self._with_description(record, f"runtime {self.runtime_label(runtime_s)} on {socket.gethostname()} (job {record['job_id']})")
        return title, body, priority

    def _with_description(self, record: dict, body: str) -> str:
        description = record.get("description") or ""
        return f"{description}\n{body}" if description else body

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

    def _deliver(self, title: str, body: str, priority: str) -> None:
        error = self._send(title, body, priority)
        if error is not None:
            self.logger.error(f"notification '{title}' failed: {error}")

    def _dispatch(self, title: str, body: str, priority: str) -> None:
        threading.Thread(target=self._deliver, args=(title, body, priority), daemon=True).start()

    def _enabled(self) -> bool:
        with self.lock:
            return self.settings["enabled"] and bool(self.settings["topic"])

    def push(self, title: str, body: str, priority: str = "default") -> None:
        with self.lock:
            if not self.settings["topic"]:
                return

        self._deliver(title, body, priority)

    def job_started(self, record: dict) -> None:
        if not self._enabled():
            return

        script = record.get("script") or "job"
        body   = self._with_description(record, f"running on {socket.gethostname()} (job {record['job_id']}, pid {record.get('pid')})")
        self._dispatch(f"{script} started", body, "default")

    def job_finished(self, record: dict) -> None:
        if not self._enabled():
            return
        if record["status"] not in ("failed", "finished"):
            return

        runtime_s             = self.runtime_s(record)
        title, body, priority = self._describe(record, runtime_s)
        self._dispatch(title, body, priority)
