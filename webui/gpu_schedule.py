from __future__ import annotations

import json
import threading
import time
from datetime import datetime

from tools.orchestration.gpu_queue import GpuPoolFile

from process_manager import ProcessManager
from project_paths   import ProjectPaths
from system_monitor  import SystemMonitor
from web_logger      import WebLogger


class Window:

    DAY_MINUTES = 24 * 60
    BOUNDS      = {}
    SAME_MOMENT = ""

    @classmethod
    def validate(cls, payload: dict) -> "Window":
        values = {}

        for key, maximum in cls.BOUNDS.items():
            value = payload[key]
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
                raise ValueError(f"'{key}' must be a whole number in [0, {maximum}], got {value!r}")
            values[key] = value

        window = cls(**values)
        if window.start_minute() == window.end_minute():
            raise ValueError(cls.SAME_MOMENT)

        return window

    def contains(self, moment: datetime) -> bool:
        now   = self.minute_of(moment)
        start = self.start_minute()
        end   = self.end_minute()

        if start < end:
            return start <= now < end

        return now >= start or now < end


class WeekWindow(Window):

    DAYS        = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    BOUNDS      = {"start_day": 6, "start_hour": 23, "end_day": 6, "end_hour": 23}
    SAME_MOMENT = "the weekend window starts and ends at the same moment, so it would never switch"

    def __init__(self, start_day: int, start_hour: int, end_day: int, end_hour: int) -> None:
        self.start_day  = start_day
        self.start_hour = start_hour
        self.end_day    = end_day
        self.end_hour   = end_hour

    def minute_of(self, moment: datetime) -> int:
        return moment.weekday() * self.DAY_MINUTES + moment.hour * 60 + moment.minute

    def start_minute(self) -> int:
        return self.start_day * self.DAY_MINUTES + self.start_hour * 60

    def end_minute(self) -> int:
        return self.end_day * self.DAY_MINUTES + self.end_hour * 60

    def label(self) -> str:
        return f"{self.DAYS[self.start_day]} {self.start_hour:02d}:00 to {self.DAYS[self.end_day]} {self.end_hour:02d}:00"

    def as_dict(self) -> dict:
        return {"start_day": self.start_day, "start_hour": self.start_hour, "end_day": self.end_day, "end_hour": self.end_hour}


class NightWindow(Window):

    BOUNDS      = {"night_start_hour": 23, "night_end_hour": 23}
    SAME_MOMENT = "the night window starts and ends at the same hour, so it would never switch"

    def __init__(self, night_start_hour: int, night_end_hour: int) -> None:
        self.night_start_hour = night_start_hour
        self.night_end_hour   = night_end_hour

    def minute_of(self, moment: datetime) -> int:
        return moment.hour * 60 + moment.minute

    def start_minute(self) -> int:
        return self.night_start_hour * 60

    def end_minute(self) -> int:
        return self.night_end_hour * 60

    def label(self) -> str:
        return f"{self.night_start_hour:02d}:00 to {self.night_end_hour:02d}:00 every day"

    def as_dict(self) -> dict:
        return {"night_start_hour": self.night_start_hour, "night_end_hour": self.night_end_hour}


class GpuAvailability:

    def __init__(self, system: SystemMonitor) -> None:
        self.system = system

    def busy(self) -> set[int]:
        held = set()

        for device in self.system.gpu_occupancy():
            owners = {proc["owner"] for proc in device["procs"] if proc["owner"]}
            if owners - {self.system.user}:
                held.add(device["index"])

        return held

    def grant(self, target: list[int]) -> list[int]:
        busy = self.busy()
        return [gpu for gpu in target if gpu not in busy]


class GpuSchedule:

    INTERVAL       = 30.0
    SWEEP_INTERVAL = 600.0
    FILE_NAME      = "gpu_schedule.json"
    GPU_KEYS       = ("weekday_gpus", "night_gpus", "weekend_gpus")

    DEFAULTS = {
        "enabled"          : False,
        "greedy"           : True,
        "weekday_gpus"     : [2, 3],
        "night_gpus"       : [0, 1, 2, 3],
        "weekend_gpus"     : [0, 1, 2, 3],
        "start_day"        : 4,
        "start_hour"       : 18,
        "end_day"          : 0,
        "end_hour"         : 8,
        "night_start_hour" : 20,
        "night_end_hour"   : 8,
    }

    def __init__(self, paths: ProjectPaths, logger: WebLogger, processes: ProcessManager, system: SystemMonitor) -> None:
        self.paths        = paths
        self.logger       = logger
        self.processes    = processes
        self.availability = GpuAvailability(system)
        self.lock         = threading.Lock()
        self.path         = self.paths.logs_dir / self.FILE_NAME
        self.applied      = {}
        self.settings     = dict(self.DEFAULTS)
        self.last_tick    = None
        self.last_sweep   = None
        self.swept_at     = None

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

        for key in ("enabled", "greedy"):
            if not isinstance(payload[key], bool):
                raise ValueError(f"'{key}' must be true or false, got {payload[key]!r}")

        settings = {"enabled": payload["enabled"], "greedy": payload["greedy"], **WeekWindow.validate(payload).as_dict(), **NightWindow.validate(payload).as_dict()}

        for key in cls.GPU_KEYS:
            gpus = GpuPoolFile.validate({"gpus": payload[key]})
            if not gpus:
                raise ValueError(f"'{key}' must hold at least one GPU; park an experiment from its console tile instead")
            settings[key] = gpus

        return settings

    def week_window(self) -> WeekWindow:
        with self.lock:
            return WeekWindow(self.settings["start_day"], self.settings["start_hour"], self.settings["end_day"], self.settings["end_hour"])

    def night_window(self) -> NightWindow:
        with self.lock:
            return NightWindow(self.settings["night_start_hour"], self.settings["night_end_hour"])

    def phase(self, moment: datetime) -> str:
        if self.week_window().contains(moment):
            return "weekend"

        if self.night_window().contains(moment):
            return "night"

        return "weekday"

    def gpus_for(self, phase: str) -> list[int]:
        with self.lock:
            return list(self.settings[f"{phase}_gpus"])

    def _live_jobs(self) -> list[str]:
        live = []

        for record in self.processes.list_jobs():
            job_id = record["job_id"]
            if record["status"] == "running" and self.processes.gpu_pool(job_id).get("live"):
                live.append(job_id)

        return live

    def _cross(self, job_id: str, phase: str) -> list[str]:
        target   = self.gpus_for(phase)
        granted  = self.availability.grant(target)
        withheld = [gpu for gpu in target if gpu not in granted]

        self.applied[job_id]["withheld"] = withheld

        if not granted:
            self.logger.warning(f"gpu schedule left job {job_id} where it is: every GPU in the {phase} pool {target} is busy with someone else's work")
            return []

        result = self.processes.set_gpus(job_id, granted)
        if not result.get("ok"):
            return []

        if withheld:
            self.logger.ok(f"gpu schedule moved job {job_id} onto the {phase} pool {granted}; {withheld} held by someone else")
            return [job_id]

        self.logger.ok(f"gpu schedule moved job {job_id} onto the {phase} pool {granted}")
        return [job_id]

    def tick(self, moment: datetime) -> list[str]:
        with self.lock:
            enabled        = self.settings["enabled"]
            self.last_tick = moment.isoformat(timespec="seconds")

        phase   = self.phase(moment)
        applied = []
        live    = self._live_jobs()

        for job_id in live:
            previous             = self.applied.get(job_id)
            self.applied[job_id] = {"phase": phase, "withheld": []} if previous is None else {**previous, "phase": phase}

            if not enabled or previous is None or previous["phase"] == phase:
                continue

            applied += self._cross(job_id, phase)

        self.applied = {job_id: record for job_id, record in self.applied.items() if job_id in live}

        return applied

    def _claim(self, job_id: str, current: list[int], freed: list[int]) -> list[str]:
        merged = sorted(set(current) | set(freed))
        if merged == sorted(current):
            return []

        result = self.processes.set_gpus(job_id, merged)
        if not result.get("ok"):
            return []

        self.logger.ok(f"gpu schedule grew job {job_id} onto {merged}: {freed} came free")
        return [job_id]

    def sweep(self, moment: datetime) -> list[str]:
        with self.lock:
            greedy          = self.settings["enabled"] and self.settings["greedy"]
            self.last_sweep = moment.isoformat(timespec="seconds")

        if not greedy:
            return []

        claimed = []

        for job_id in self._live_jobs():
            record = self.applied.get(job_id)
            if record is None or not record["withheld"]:
                continue

            current = self.processes.gpu_pool(job_id).get("gpus", [])
            if not current:
                continue

            freed = self.availability.grant(record["withheld"])
            if not freed:
                continue

            claimed += self._claim(job_id, current, freed)
            record["withheld"] = [gpu for gpu in record["withheld"] if gpu not in freed]

        return claimed

    def _due(self, now: float) -> bool:
        if self.swept_at is None or now - self.swept_at >= self.SWEEP_INTERVAL:
            self.swept_at = now
            return True

        return False

    def _watch(self) -> None:
        while True:
            try:
                moment = datetime.now()
                self.tick(moment)

                if self._due(time.monotonic()):
                    self.sweep(moment)
            except Exception as error:
                self.logger.error(f"gpu schedule tick failed: {error}")

            time.sleep(self.INTERVAL)

    def start(self) -> None:
        worker = threading.Thread(target=self._watch, daemon=True)
        worker.start()
        self.logger.muted(f"gpu schedule armed from {self.path} (enabled={self.settings['enabled']}, greedy={self.settings['greedy']}, every {self.INTERVAL:.0f}s)")

    def state(self) -> dict:
        with self.lock:
            settings = dict(self.settings)
            last     = self.last_tick
            swept    = self.last_sweep

        phase   = self.phase(datetime.now())
        waiting = sorted({gpu for record in self.applied.values() for gpu in record["withheld"]})

        return {
            **settings,
            "phase"        : phase,
            "gpus_now"     : list(settings[f"{phase}_gpus"]),
            "waiting"      : waiting,
            "window"       : self.week_window().label(),
            "night_window" : self.night_window().label(),
            "last_tick"    : last,
            "last_sweep"   : swept,
            "path"         : str(self.path),
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

        self.logger.ok(f"gpu schedule saved: {'on' if settings['enabled'] else 'off'}, greedy {'on' if settings['greedy'] else 'off'}, weekday {settings['weekday_gpus']}, night {settings['night_gpus']}, weekend {settings['weekend_gpus']}")

        return {"ok": True, **self.state()}
