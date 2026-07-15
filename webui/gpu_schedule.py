from __future__ import annotations

import json
import threading
import time
from datetime import datetime

from tools.orchestration.gpu_queue import GpuPoolFile

from process_manager import ProcessManager
from project_paths   import ProjectPaths
from web_logger      import WebLogger


class WeekWindow:

    DAYS         = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    DAY_MINUTES  = 24 * 60
    BOUNDS       = {"start_day": 6, "start_hour": 23, "end_day": 6, "end_hour": 23}

    def __init__(self, start_day: int, start_hour: int, end_day: int, end_hour: int) -> None:
        self.start_day  = start_day
        self.start_hour = start_hour
        self.end_day    = end_day
        self.end_hour   = end_hour

    @classmethod
    def validate(cls, payload: dict) -> "WeekWindow":
        values = {}

        for key, maximum in cls.BOUNDS.items():
            value = payload[key]
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
                raise ValueError(f"'{key}' must be a whole number in [0, {maximum}], got {value!r}")
            values[key] = value

        window = cls(**values)
        if window.start_minute() == window.end_minute():
            raise ValueError("the weekend window starts and ends at the same moment, so it would never switch")

        return window

    def start_minute(self) -> int:
        return self.start_day * self.DAY_MINUTES + self.start_hour * 60

    def end_minute(self) -> int:
        return self.end_day * self.DAY_MINUTES + self.end_hour * 60

    def contains(self, moment: datetime) -> bool:
        now   = moment.weekday() * self.DAY_MINUTES + moment.hour * 60 + moment.minute
        start = self.start_minute()
        end   = self.end_minute()

        if start < end:
            return start <= now < end

        return now >= start or now < end

    def label(self) -> str:
        return f"{self.DAYS[self.start_day]} {self.start_hour:02d}:00 to {self.DAYS[self.end_day]} {self.end_hour:02d}:00"

    def as_dict(self) -> dict:
        return {"start_day": self.start_day, "start_hour": self.start_hour, "end_day": self.end_day, "end_hour": self.end_hour}


class GpuSchedule:

    INTERVAL  = 30.0
    FILE_NAME = "gpu_schedule.json"
    GPU_KEYS  = ("weekday_gpus", "weekend_gpus")

    DEFAULTS = {
        "enabled"      : False,
        "weekday_gpus" : [0],
        "weekend_gpus" : [0, 1, 2, 3],
        "start_day"    : 4,
        "start_hour"   : 18,
        "end_day"      : 0,
        "end_hour"     : 8,
    }

    def __init__(self, paths: ProjectPaths, logger: WebLogger, processes: ProcessManager) -> None:
        self.paths     = paths
        self.logger    = logger
        self.processes = processes
        self.lock      = threading.Lock()
        self.path      = self.paths.logs_dir / self.FILE_NAME
        self.applied   = {}
        self.settings  = dict(self.DEFAULTS)
        self.last_tick = None

        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return

        try:
            self.settings = self.validate(json.loads(self.path.read_text(encoding="utf-8")))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError, OSError) as error:
            self.logger.error(f"ignoring unreadable GPU schedule {self.path}: {error}; the built-in defaults apply until it is fixed")

    @classmethod
    def validate(cls, payload) -> dict:
        if not isinstance(payload, dict):
            raise ValueError(f"expected an object holding the schedule, got {payload!r}")

        missing = [key for key in cls.DEFAULTS if key not in payload]
        if missing:
            raise ValueError(f"the schedule must define {missing}")

        if not isinstance(payload["enabled"], bool):
            raise ValueError(f"'enabled' must be true or false, got {payload['enabled']!r}")

        settings = {"enabled": payload["enabled"], **WeekWindow.validate(payload).as_dict()}

        for key in cls.GPU_KEYS:
            gpus = GpuPoolFile.validate({"gpus": payload[key]})
            if not gpus:
                raise ValueError(f"'{key}' must hold at least one GPU; park an experiment from its console tile instead")
            settings[key] = gpus

        return settings

    def window(self) -> WeekWindow:
        with self.lock:
            return WeekWindow(self.settings["start_day"], self.settings["start_hour"], self.settings["end_day"], self.settings["end_hour"])

    def phase(self, moment: datetime) -> str:
        return "weekend" if self.window().contains(moment) else "weekday"

    def gpus_for(self, phase: str) -> list[int]:
        with self.lock:
            return list(self.settings[f"{phase}_gpus"])

    def tick(self, moment: datetime) -> list[str]:
        with self.lock:
            enabled        = self.settings["enabled"]
            self.last_tick = moment.isoformat(timespec="seconds")

        phase   = self.phase(moment)
        applied = []
        seen    = set()

        for record in self.processes.list_jobs():
            job_id = record["job_id"]

            if record["status"] != "running" or not self.processes.gpu_pool(job_id).get("live"):
                continue

            seen.add(job_id)
            previous             = self.applied.get(job_id)
            self.applied[job_id] = phase

            if not enabled or previous is None or previous == phase:
                continue

            gpus   = self.gpus_for(phase)
            result = self.processes.set_gpus(job_id, gpus)
            if result.get("ok"):
                applied.append(job_id)
                self.logger.ok(f"gpu schedule moved job {job_id} onto the {phase} pool {gpus}")

        self.applied = {job_id: phase for job_id, phase in self.applied.items() if job_id in seen}

        return applied

    def _watch(self) -> None:
        while True:
            try:
                self.tick(datetime.now())
            except Exception as error:
                self.logger.error(f"gpu schedule tick failed: {error}")

            time.sleep(self.INTERVAL)

    def start(self) -> None:
        worker = threading.Thread(target=self._watch, daemon=True)
        worker.start()
        self.logger.muted(f"gpu schedule armed from {self.path} (enabled={self.settings['enabled']}, every {self.INTERVAL:.0f}s)")

    def state(self) -> dict:
        with self.lock:
            settings = dict(self.settings)
            last     = self.last_tick

        phase = self.phase(datetime.now())

        return {
            **settings,
            "phase"     : phase,
            "gpus_now"  : list(settings[f"{phase}_gpus"]),
            "window"    : self.window().label(),
            "last_tick" : last,
            "path"      : str(self.path),
        }

    def update(self, payload) -> dict:
        try:
            settings = self.validate(payload)
        except (ValueError, TypeError) as error:
            return {"ok": False, "error": str(error)}

        with self.lock:
            self.settings = settings

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

        self.logger.ok(f"gpu schedule saved: {'on' if settings['enabled'] else 'off'}, weekday {settings['weekday_gpus']}, weekend {settings['weekend_gpus']}")

        return {"ok": True, **self.state()}
