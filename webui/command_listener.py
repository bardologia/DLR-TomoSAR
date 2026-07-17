from __future__ import annotations

import http.client
import json
import socket
import threading
import time
import urllib.error
import urllib.request

from notifier        import JobNotifier
from process_manager import ProcessManager, ProcessNuke
from project_paths   import ProjectPaths
from system_monitor  import SystemMonitor
from web_logger      import WebLogger


class CommandListener:

    SETTINGS_NAME   = "commands.json"
    CONFIRM_S       = 60.0
    READ_TIMEOUT_S  = 120.0
    RECONNECT_MIN_S = 2.0
    RECONNECT_MAX_S = 60.0
    DEFAULTS        = {"enabled": False, "topic": "", "server": "https://ntfy.sh"}

    HELP = (
        "status — jobs, queue and gpus\n"
        "stop <job> — stop one job (id prefix)\n"
        "stop all — arm, then 'stop all confirm'\n"
        "nuke — arm, then 'nuke confirm'\n"
        "gpus <job> — show the live gpu pool\n"
        "gpus <job> 0,1 — resize the pool ('none' parks)"
    )

    def __init__(self, paths: ProjectPaths, logger: WebLogger, notifier: JobNotifier, processes: ProcessManager, nuke: ProcessNuke, system: SystemMonitor) -> None:
        self.paths     = paths
        self.logger    = logger
        self.notifier  = notifier
        self.processes = processes
        self.nuke      = nuke
        self.system    = system

        self.lock       = threading.Lock()
        self.path       = paths.logs_dir / self.SETTINGS_NAME
        self.settings   = self._load()
        self.armed      = None
        self.stop_event = None
        self.response   = None
        self.started    = time.time()

    def _load(self) -> dict:
        if not self.path.exists():
            return dict(self.DEFAULTS)
        loaded = json.loads(self.path.read_text())
        return {key: loaded[key] for key in self.DEFAULTS}

    def state(self) -> dict:
        with self.lock:
            listening = self.stop_event is not None and not self.stop_event.is_set()
            return {**self.settings, "listening": listening, "settings_path": str(self.path)}

    def configure(self, payload: dict) -> dict:
        enabled = bool(payload.get("enabled", False))
        topic   = str(payload.get("topic", "")).strip()
        server  = str(payload.get("server", self.DEFAULTS["server"])).strip().rstrip("/")

        if topic and not JobNotifier.TOPIC_PATTERN.match(topic):
            return {"ok": False, "error": "topic may only contain letters, digits, '-' and '_' (max 64 chars)"}
        if enabled and not topic:
            return {"ok": False, "error": "set a command topic before enabling remote commands"}
        if not server.startswith(("http://", "https://")):
            return {"ok": False, "error": "server must be an http(s) URL"}

        notify = self.notifier.state()
        if topic and topic == notify["topic"]:
            return {"ok": False, "error": "the command topic must differ from the notification topic"}
        if enabled and not notify["topic"]:
            return {"ok": False, "error": "set a notification topic first — command replies are pushed there"}

        with self.lock:
            self.settings = {"enabled": enabled, "topic": topic, "server": server}
            self._persist()

        self._apply()
        self.logger.ok(f"remote commands {'enabled' if enabled else 'disabled'} (topic '{topic}')")
        return {"ok": True, **self.state()}

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.settings, indent=2) + "\n")

    def start(self) -> None:
        self._apply()

    def _apply(self) -> None:
        with self.lock:
            enabled  = self.settings["enabled"] and bool(self.settings["topic"])
            previous = self.stop_event
            response = self.response

            if previous is not None:
                previous.set()

            self.response   = None
            self.stop_event = threading.Event() if enabled else None
            current         = self.stop_event

        self._close(response)

        if current is not None:
            threading.Thread(target=self._listen, args=(current,), name="CommandListener", daemon=True).start()

    def _close(self, response) -> None:
        if response is None:
            return
        try:
            response.close()
        except OSError:
            pass

    def _connect(self):
        with self.lock:
            url = f"{self.settings['server']}/{self.settings['topic']}/json"

        request = urllib.request.Request(url, headers={"Accept": "application/x-ndjson"})
        return urllib.request.urlopen(request, timeout=self.READ_TIMEOUT_S)

    def _reply_topic_clashes(self) -> bool:
        with self.lock:
            topic = self.settings["topic"]
        return self.notifier.state()["topic"] == topic

    def _job_line(self, record: dict) -> str:
        runtime     = self.notifier.runtime_label(self.notifier.runtime_s(record))
        line        = f"{record['job_id'][:6]} {record['script']} · {runtime}"
        description = record.get("description") or ""
        return f"{line}\n  {description}" if description else line

    def _gpu_line(self, device: dict) -> str:
        owners = {proc["owner"] for proc in device["procs"] if proc["owner"]}
        used   = round(device["mem_used"] / 1024)
        total  = round(device["mem_total"] / 1024)
        label  = "free" if not owners else ("yours" if owners == {self.system.user} else ", ".join(sorted(owners)))
        return f"gpu{device['index']} {device['util']}% {used}/{total}GB · {label}"

    def _cmd_status(self) -> tuple[str, str, str]:
        jobs    = self.processes.list_jobs()
        running = [record for record in jobs if record["status"] == "running"]
        waiting = [record for record in jobs if record["status"] in ("queued", "scheduled")]

        lines  = [self._job_line(record) for record in running]
        lines += [f"{record['job_id'][:6]} {record['script']} · {record['status']}" for record in waiting]
        if not lines:
            lines = ["no jobs running or queued"]

        gpus = [self._gpu_line(device) for device in self.system.gpu_occupancy()]
        body = "\n".join(lines + gpus)
        return f"{socket.gethostname()}: {len(running)} running, {len(waiting)} waiting", body, "default"

    def _match_job(self, prefix: str, statuses: tuple) -> tuple[dict | None, str | None]:
        matches = [record for record in self.processes.list_jobs() if record["status"] in statuses and record["job_id"].startswith(prefix.lower())]

        if not matches:
            return None, f"no {'/'.join(statuses)} job matches '{prefix}' — send 'status' for the list"
        if len(matches) > 1:
            listing = ", ".join(f"{record['job_id'][:6]} {record['script']}" for record in matches)
            return None, f"'{prefix}' is ambiguous: {listing}"
        return matches[0], None

    def _cmd_stop(self, prefix: str) -> tuple[str, str, str]:
        record, error = self._match_job(prefix, ("running", "queued", "scheduled"))
        if record is None:
            return "stop failed", error, "default"

        result = self.processes.stop(record["job_id"])
        if not result.get("ok"):
            return "stop failed", result.get("error", "unknown error"), "default"
        return "stop requested", f"{record['script']} (job {record['job_id'][:6]})", "default"

    def _cmd_gpus(self, prefix: str, spec: str | None) -> tuple[str, str, str]:
        record, error = self._match_job(prefix, ("running",))
        if record is None:
            return "gpus failed", error, "default"

        if spec is None:
            result = self.processes.gpu_pool(record["job_id"])
            if not result.get("ok"):
                return "gpus failed", result.get("error", "unknown error"), "default"
            if not result.get("live"):
                return "gpu pool", f"{record['script']} has no live gpu pool", "default"
            return "gpu pool", f"{record['script']} on gpus {result['gpus']}", "default"

        park = spec == "none"
        try:
            gpus = [] if park else [int(cell) for cell in spec.split(",") if cell != ""]
        except ValueError:
            return "gpus failed", f"'{spec}' is not a gpu list — send 'gpus {prefix} 0,1' or 'gpus {prefix} none'", "default"

        result = self.processes.set_gpus(record["job_id"], gpus, park=park)
        if not result.get("ok"):
            return "gpus failed", result.get("error", "unknown error"), "default"
        if result.get("parked"):
            return "gpu pool set", f"{record['script']} parked — runs in flight finish, nothing new starts", "default"
        return "gpu pool set", f"{record['script']} now on gpus {result['gpus']}", "default"

    def _arm(self, action: str, warning: str) -> tuple[str, str, str]:
        with self.lock:
            self.armed = {"action": action, "deadline": time.monotonic() + self.CONFIRM_S}
        return f"{action} armed", f"reply '{action} confirm' within {int(self.CONFIRM_S)}s — {warning}", "high"

    def _take_armed(self, action: str) -> bool:
        with self.lock:
            armed      = self.armed
            self.armed = None
        return armed is not None and armed["action"] == action and time.monotonic() < armed["deadline"]

    def _cmd_stop_all_confirm(self) -> tuple[str, str, str]:
        if not self._take_armed("stop all"):
            return "nothing armed", f"send 'stop all' first — confirmations expire after {int(self.CONFIRM_S)}s", "default"

        stopped = self.processes.stop_all()
        return "stop all done", f"{stopped} jobs stopped, queue cleared", "high"

    def _cmd_nuke_confirm(self) -> tuple[str, str, str]:
        if not self._take_armed("nuke"):
            return "nothing armed", f"send 'nuke' first — confirmations expire after {int(self.CONFIRM_S)}s", "default"

        self.processes.clear_queue()
        result = self.nuke.nuke()
        return "nuke done", f"SIGTERM to {result['signalled']} processes, SIGKILL to {result['killed']} survivors", "high"

    def handle(self, text: str) -> tuple[str, str, str] | None:
        tokens = text.strip().split()
        if not tokens:
            return None
        words = [token.lower() for token in tokens]

        if words == ["help"]:
            return "commands", self.HELP, "default"
        if words == ["status"]:
            return self._cmd_status()
        if words == ["nuke"]:
            return self._arm("nuke", "kills every process under this user")
        if words == ["nuke", "confirm"]:
            return self._cmd_nuke_confirm()
        if words == ["stop", "all"]:
            return self._arm("stop all", "stops every job and clears the launch queue")
        if words == ["stop", "all", "confirm"]:
            return self._cmd_stop_all_confirm()
        if len(words) == 2 and words[0] == "stop":
            return self._cmd_stop(words[1])
        if len(words) in (2, 3) and words[0] == "gpus":
            return self._cmd_gpus(words[1], words[2] if len(words) == 3 else None)

        return "unknown command", f"'{text.strip()[:80]}' not understood — send 'help' for the list", "default"

    def _handle_line(self, raw: bytes) -> None:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            return

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return

        if event.get("event") != "message":
            return
        if float(event.get("time") or 0.0) < self.started - 5.0:
            return
        if self._reply_topic_clashes():
            self.logger.error("remote command dropped: command topic equals the notification topic")
            return

        reply = self.handle(str(event.get("message") or ""))
        if reply is None:
            return

        title, body, priority = reply
        self.logger.ok(f"remote command '{str(event.get('message') or '').strip()[:80]}' → {title}")
        self.notifier.push(title, body, priority)

    def _listen(self, stop: threading.Event) -> None:
        backoff = self.RECONNECT_MIN_S

        while not stop.is_set():
            try:
                response = self._connect()
            except (urllib.error.URLError, OSError):
                stop.wait(backoff)
                backoff = min(backoff * 2.0, self.RECONNECT_MAX_S)
                continue

            with self.lock:
                if stop.is_set():
                    self._close(response)
                    return
                self.response = response

            backoff = self.RECONNECT_MIN_S
            try:
                while not stop.is_set():
                    try:
                        raw = response.readline()
                    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError, AttributeError):
                        break
                    if not raw:
                        break
                    self._handle_line(raw)
            finally:
                with self.lock:
                    if self.response is response:
                        self.response = None
                self._close(response)

            stop.wait(self.RECONNECT_MIN_S)
