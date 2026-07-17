from __future__ import annotations

import socket
import threading
import time

from notifier        import JobNotifier
from process_manager import ProcessManager, ProcessNuke
from system_monitor  import SystemMonitor
from telegram_bot    import TelegramBot
from web_logger      import WebLogger


class CommandListener:

    CONFIRM_S       = 60.0
    POLL_TIMEOUT_S  = 20.0
    RECONNECT_MIN_S = 2.0
    RECONNECT_MAX_S = 60.0

    HELP = (
        "status — jobs, queue and gpus\n"
        "stop <job> — stop one job (id prefix)\n"
        "stop all — arm, then 'stop all confirm'\n"
        "nuke — arm, then 'nuke confirm'\n"
        "gpus <job> — show the live gpu pool\n"
        "gpus <job> 0,1 — resize the pool ('none' parks)"
    )

    def __init__(self, bot: TelegramBot, logger: WebLogger, notifier: JobNotifier, processes: ProcessManager, nuke: ProcessNuke, system: SystemMonitor) -> None:
        self.bot       = bot
        self.logger    = logger
        self.notifier  = notifier
        self.processes = processes
        self.nuke      = nuke
        self.system    = system

        self.lock       = threading.Lock()
        self.armed      = None
        self.stop_event = None
        self.started    = time.time()

    def state(self) -> dict:
        with self.lock:
            return {"listening": self.stop_event is not None and not self.stop_event.is_set()}

    def start(self) -> None:
        self.apply()

    def apply(self) -> None:
        with self.lock:
            enabled  = self.bot.commands_enabled()
            previous = self.stop_event

            if previous is not None:
                previous.set()

            self.stop_event = threading.Event() if enabled else None
            current         = self.stop_event

        if current is not None:
            threading.Thread(target=self._listen, args=(current,), name="CommandListener", daemon=True).start()

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

    def _cmd_status(self) -> tuple[str, str]:
        jobs    = self.processes.list_jobs()
        running = [record for record in jobs if record["status"] == "running"]
        waiting = [record for record in jobs if record["status"] in ("queued", "scheduled")]

        lines  = [self._job_line(record) for record in running]
        lines += [f"{record['job_id'][:6]} {record['script']} · {record['status']}" for record in waiting]
        if not lines:
            lines = ["no jobs running or queued"]

        gpus = [self._gpu_line(device) for device in self.system.gpu_occupancy()]
        body = "\n".join(lines + gpus)
        return f"{socket.gethostname()}: {len(running)} running, {len(waiting)} waiting", body

    def _match_job(self, prefix: str, statuses: tuple) -> tuple[dict | None, str | None]:
        matches = [record for record in self.processes.list_jobs() if record["status"] in statuses and record["job_id"].startswith(prefix.lower())]

        if not matches:
            return None, f"no {'/'.join(statuses)} job matches '{prefix}' — send 'status' for the list"
        if len(matches) > 1:
            listing = ", ".join(f"{record['job_id'][:6]} {record['script']}" for record in matches)
            return None, f"'{prefix}' is ambiguous: {listing}"
        return matches[0], None

    def _cmd_stop(self, prefix: str) -> tuple[str, str]:
        record, error = self._match_job(prefix, ("running", "queued", "scheduled"))
        if record is None:
            return "stop failed", error

        result = self.processes.stop(record["job_id"])
        if not result.get("ok"):
            return "stop failed", result.get("error", "unknown error")
        return "stop requested", f"{record['script']} (job {record['job_id'][:6]})"

    def _cmd_gpus(self, prefix: str, spec: str | None) -> tuple[str, str]:
        record, error = self._match_job(prefix, ("running",))
        if record is None:
            return "gpus failed", error

        if spec is None:
            result = self.processes.gpu_pool(record["job_id"])
            if not result.get("ok"):
                return "gpus failed", result.get("error", "unknown error")
            if not result.get("live"):
                return "gpu pool", f"{record['script']} has no live gpu pool"
            return "gpu pool", f"{record['script']} on gpus {result['gpus']}"

        park = spec == "none"
        try:
            gpus = [] if park else [int(cell) for cell in spec.split(",") if cell != ""]
        except ValueError:
            return "gpus failed", f"'{spec}' is not a gpu list — send 'gpus {prefix} 0,1' or 'gpus {prefix} none'"

        result = self.processes.set_gpus(record["job_id"], gpus, park=park)
        if not result.get("ok"):
            return "gpus failed", result.get("error", "unknown error")
        if result.get("parked"):
            return "gpu pool set", f"{record['script']} parked — runs in flight finish, nothing new starts"
        return "gpu pool set", f"{record['script']} now on gpus {result['gpus']}"

    def _arm(self, action: str, warning: str) -> tuple[str, str]:
        with self.lock:
            self.armed = {"action": action, "deadline": time.monotonic() + self.CONFIRM_S}
        return f"{action} armed", f"reply '{action} confirm' within {int(self.CONFIRM_S)}s — {warning}"

    def _take_armed(self, action: str) -> bool:
        with self.lock:
            armed      = self.armed
            self.armed = None
        return armed is not None and armed["action"] == action and time.monotonic() < armed["deadline"]

    def _cmd_stop_all_confirm(self) -> tuple[str, str]:
        if not self._take_armed("stop all"):
            return "nothing armed", f"send 'stop all' first — confirmations expire after {int(self.CONFIRM_S)}s"

        stopped = self.processes.stop_all()
        return "stop all done", f"{stopped} jobs stopped, queue cleared"

    def _cmd_nuke_confirm(self) -> tuple[str, str]:
        if not self._take_armed("nuke"):
            return "nothing armed", f"send 'nuke' first — confirmations expire after {int(self.CONFIRM_S)}s"

        self.processes.clear_queue()
        result = self.nuke.nuke()
        return "nuke done", f"SIGTERM to {result['signalled']} processes, SIGKILL to {result['killed']} survivors"

    def handle(self, text: str) -> tuple[str, str] | None:
        tokens = text.strip().split()
        if not tokens:
            return None
        words = [token.lower() for token in tokens]

        if words == ["help"]:
            return "commands", self.HELP
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

        return "unknown command", f"'{text.strip()[:80]}' not understood — send 'help' for the list"

    def _handle_update(self, update: dict) -> None:
        message = update.get("message") or {}
        chat_id = str(((message.get("chat") or {}).get("id", "")))
        text    = str(message.get("text") or "")

        if chat_id != self.bot.chat_id():
            self.logger.warning(f"telegram message from unpaired chat {chat_id or '?'} ignored")
            return
        if float(message.get("date") or 0.0) < self.started - 5.0:
            return

        reply = self.handle(text)
        if reply is None:
            return

        title, body = reply
        self.logger.ok(f"remote command '{text.strip()[:80]}' → {title}")
        error = self.bot.send(f"{title}\n{body}")
        if error is not None:
            self.logger.error(f"remote command reply failed: {error}")

    def _drain(self) -> int | None:
        updates, error = self.bot.get_updates(-1, 0.0)
        if error is not None or not updates:
            return None
        return updates[-1]["update_id"] + 1

    def _listen(self, stop: threading.Event) -> None:
        offset  = self._drain()
        backoff = self.RECONNECT_MIN_S

        while not stop.is_set():
            updates, error = self.bot.get_updates(offset, self.POLL_TIMEOUT_S)

            if error is not None:
                self.logger.error(f"telegram poll failed: {error}")
                stop.wait(backoff)
                backoff = min(backoff * 2.0, self.RECONNECT_MAX_S)
                continue

            backoff = self.RECONNECT_MIN_S
            for update in updates:
                offset = update["update_id"] + 1
                if stop.is_set():
                    break
                self._handle_update(update)
