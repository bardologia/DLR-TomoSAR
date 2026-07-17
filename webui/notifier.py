from __future__ import annotations

import socket
from datetime import datetime

from telegram_bot import TelegramBot
from web_logger   import WebLogger


class JobNotifier:

    def __init__(self, bot: TelegramBot, logger: WebLogger) -> None:
        self.bot    = bot
        self.logger = logger

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

    def _describe(self, record: dict, runtime_s: float) -> tuple[str, str]:
        script = record.get("script") or "job"
        code   = record.get("exit_code")

        if record.get("stopped"):
            title = f"{script} stopped" + (f" (exit {code})" if code is not None else "")
        elif record["status"] == "failed":
            title = f"{script} FAILED" + (f" (exit {code})" if code is not None else "")
        else:
            title = f"{script} finished" + ("" if code == 0 else " (exit status unknown)")

        body = self._with_description(record, f"runtime {self.runtime_label(runtime_s)} on {socket.gethostname()} (job {record['job_id']})")
        return title, body

    def _with_description(self, record: dict, body: str) -> str:
        description = record.get("description") or ""
        return f"{description}\n{body}" if description else body

    def job_started(self, record: dict) -> None:
        if not self.bot.notify_enabled():
            return

        script = record.get("script") or "job"
        body   = self._with_description(record, f"running on {socket.gethostname()} (job {record['job_id']}, pid {record.get('pid')})")
        self.bot.send_async(f"{script} started\n{body}")

    def job_finished(self, record: dict) -> None:
        if not self.bot.notify_enabled():
            return
        if record["status"] not in ("failed", "finished"):
            return

        title, body = self._describe(record, self.runtime_s(record))
        self.bot.send_async(f"{title}\n{body}")
