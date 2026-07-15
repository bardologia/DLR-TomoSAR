from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib  import Path

import pytest

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from gpu_schedule import GpuSchedule, NightWindow, WeekWindow
from web_logger   import WebLogger

FRIDAY_MORNING   = datetime(2026, 7, 17, 9, 0)
FRIDAY_EVENING   = datetime(2026, 7, 17, 18, 30)
SATURDAY         = datetime(2026, 7, 18, 12, 0)
SUNDAY_NIGHT     = datetime(2026, 7, 19, 23, 30)
MONDAY_EARLY     = datetime(2026, 7, 20, 7, 30)
MONDAY_WORKDAY   = datetime(2026, 7, 20, 9, 0)
WEDNESDAY        = datetime(2026, 7, 22, 15, 0)
WEDNESDAY_NIGHT  = datetime(2026, 7, 22, 21, 0)
THURSDAY_EARLY   = datetime(2026, 7, 23, 2, 0)
THURSDAY_WORKDAY = datetime(2026, 7, 23, 9, 0)


class StubPaths:
    def __init__(self, root: Path) -> None:
        self.logs_dir = root / "logs"


class StubProcesses:
    def __init__(self) -> None:
        self.jobs   = []
        self.pools  = {}
        self.writes = []

    def list_jobs(self) -> list[dict]:
        return list(self.jobs)

    def gpu_pool(self, job_id: str) -> dict:
        return self.pools.get(job_id, {"ok": True, "supported": False, "live": False})

    def set_gpus(self, job_id: str, gpus) -> dict:
        self.writes.append((job_id, list(gpus)))
        return {"ok": True, "gpus": list(gpus), "parked": False}

    def add_running_fan_out(self, job_id: str) -> None:
        self.jobs.append({"job_id": job_id, "status": "running", "script": "train_backbone"})
        self.pools[job_id] = {"ok": True, "supported": True, "live": True, "gpus": [0]}


@pytest.fixture
def schedule(tmp_path):
    processes = StubProcesses()
    schedule  = GpuSchedule(StubPaths(tmp_path), WebLogger(), processes)
    schedule.settings["enabled"] = True
    return schedule, processes


def test_default_window_spans_friday_evening_to_monday_morning():
    window = WeekWindow(**{key: GpuSchedule.DEFAULTS[key] for key in ("start_day", "start_hour", "end_day", "end_hour")})

    assert window.contains(FRIDAY_EVENING) is True
    assert window.contains(SATURDAY)       is True
    assert window.contains(SUNDAY_NIGHT)   is True
    assert window.contains(MONDAY_EARLY)   is True

    assert window.contains(FRIDAY_MORNING) is False
    assert window.contains(MONDAY_WORKDAY) is False
    assert window.contains(WEDNESDAY)      is False


def test_window_that_does_not_wrap_the_week():
    window = WeekWindow(start_day=1, start_hour=8, end_day=3, end_hour=8)

    assert window.contains(datetime(2026, 7, 21, 9, 0))  is True
    assert window.contains(datetime(2026, 7, 22, 23, 0)) is True
    assert window.contains(datetime(2026, 7, 23, 9, 0))  is False
    assert window.contains(datetime(2026, 7, 20, 9, 0))  is False


def test_window_boundaries_are_inclusive_at_the_start_and_exclusive_at_the_end():
    window = WeekWindow(**{key: GpuSchedule.DEFAULTS[key] for key in ("start_day", "start_hour", "end_day", "end_hour")})

    assert window.contains(datetime(2026, 7, 17, 18, 0)) is True
    assert window.contains(datetime(2026, 7, 17, 17, 59)) is False
    assert window.contains(datetime(2026, 7, 20, 7, 59)) is True
    assert window.contains(datetime(2026, 7, 20, 8, 0))  is False


def test_default_night_window_spans_the_evening_to_the_next_morning():
    window = NightWindow(**{key: GpuSchedule.DEFAULTS[key] for key in ("night_start_hour", "night_end_hour")})

    assert window.contains(WEDNESDAY_NIGHT)   is True
    assert window.contains(THURSDAY_EARLY)    is True
    assert window.contains(datetime(2026, 7, 22, 20, 0)) is True

    assert window.contains(WEDNESDAY)         is False
    assert window.contains(THURSDAY_WORKDAY)  is False
    assert window.contains(datetime(2026, 7, 22, 19, 59)) is False
    assert window.contains(datetime(2026, 7, 23, 8, 0))   is False


def test_phase_names_follow_the_window(schedule):
    scheduler, _processes = schedule

    assert scheduler.phase(SATURDAY)         == "weekend"
    assert scheduler.phase(MONDAY_WORKDAY)   == "weekday"
    assert scheduler.phase(WEDNESDAY_NIGHT)  == "night"
    assert scheduler.phase(THURSDAY_EARLY)   == "night"


def test_the_weekend_window_outranks_the_night_window(schedule):
    scheduler, _processes = schedule

    assert scheduler.phase(SUNDAY_NIGHT)   == "weekend"
    assert scheduler.phase(FRIDAY_EVENING) == "weekend"


def test_transition_into_the_night_grows_the_pool_and_the_morning_shrinks_it(schedule):
    scheduler, processes = schedule
    processes.add_running_fan_out("job1")

    scheduler.tick(WEDNESDAY)
    scheduler.tick(WEDNESDAY_NIGHT)
    scheduler.tick(THURSDAY_EARLY)
    scheduler.tick(THURSDAY_WORKDAY)

    assert processes.writes == [("job1", [0, 1, 2, 3]), ("job1", [2, 3])]


def test_first_tick_records_the_phase_without_touching_the_launch_selection(schedule):
    scheduler, processes = schedule
    processes.add_running_fan_out("job1")

    scheduler.tick(WEDNESDAY)

    assert processes.writes    == []
    assert scheduler.applied   == {"job1": "weekday"}


def test_transition_into_the_weekend_grows_the_pool(schedule):
    scheduler, processes = schedule
    processes.add_running_fan_out("job1")

    scheduler.tick(WEDNESDAY)
    scheduler.tick(FRIDAY_EVENING)

    assert processes.writes == [("job1", [0, 1, 2, 3])]


def test_transition_back_to_the_week_shrinks_the_pool(schedule):
    scheduler, processes = schedule
    processes.add_running_fan_out("job1")

    scheduler.tick(SATURDAY)
    scheduler.tick(MONDAY_WORKDAY)

    assert processes.writes == [("job1", [2, 3])]


def test_ticks_inside_one_phase_never_rewrite_the_pool(schedule):
    scheduler, processes = schedule
    processes.add_running_fan_out("job1")

    scheduler.tick(SATURDAY)
    scheduler.tick(SUNDAY_NIGHT)
    scheduler.tick(MONDAY_EARLY)

    assert processes.writes == []


def test_a_manual_resize_survives_until_the_next_transition(schedule):
    scheduler, processes = schedule
    processes.add_running_fan_out("job1")

    scheduler.tick(SATURDAY)
    processes.pools["job1"]["gpus"] = [2]

    scheduler.tick(SUNDAY_NIGHT)
    assert processes.writes == []

    scheduler.tick(MONDAY_WORKDAY)
    assert processes.writes == [("job1", [2, 3])]


def test_a_disabled_schedule_tracks_phases_without_writing(schedule):
    scheduler, processes = schedule
    scheduler.settings["enabled"] = False
    processes.add_running_fan_out("job1")

    scheduler.tick(WEDNESDAY)
    scheduler.tick(SATURDAY)

    assert processes.writes  == []
    assert scheduler.applied == {"job1": "weekend"}


def test_jobs_without_a_live_pool_are_ignored(schedule):
    scheduler, processes = schedule
    processes.jobs.append({"job_id": "single", "status": "running", "script": "train_backbone"})

    scheduler.tick(WEDNESDAY)
    scheduler.tick(SATURDAY)

    assert processes.writes  == []
    assert scheduler.applied == {}


def test_finished_jobs_are_forgotten(schedule):
    scheduler, processes = schedule
    processes.add_running_fan_out("job1")

    scheduler.tick(WEDNESDAY)
    assert scheduler.applied == {"job1": "weekday"}

    processes.jobs[0]["status"] = "finished"
    scheduler.tick(SATURDAY)

    assert scheduler.applied == {}


def test_update_persists_and_reloads(tmp_path):
    processes = StubProcesses()
    payload   = {**GpuSchedule.DEFAULTS, "enabled": True, "weekday_gpus": [1], "night_gpus": [1, 2], "weekend_gpus": [0, 1, 2]}

    scheduler = GpuSchedule(StubPaths(tmp_path), WebLogger(), processes)
    result    = scheduler.update(payload)

    assert result["ok"] is True
    assert json.loads(scheduler.path.read_text())["weekday_gpus"] == [1]

    reloaded = GpuSchedule(StubPaths(tmp_path), WebLogger(), processes)

    assert reloaded.settings["enabled"]      is True
    assert reloaded.settings["weekday_gpus"] == [1]
    assert reloaded.settings["night_gpus"]   == [1, 2]
    assert reloaded.settings["weekend_gpus"] == [0, 1, 2]


def test_state_reports_the_pool_that_applies_right_now(schedule):
    scheduler, _processes = schedule

    state = scheduler.state()

    assert state["phase"] in ("weekday", "night", "weekend")
    assert state["gpus_now"] == state[f"{state['phase']}_gpus"]
    assert "friday 18:00 to monday 08:00" == state["window"]
    assert "20:00 to 08:00 every day"     == state["night_window"]


@pytest.mark.parametrize("payload, reason", [
    ({"enabled": "yes"},                    "true or false"),
    ({"weekday_gpus": []},                  "at least one GPU"),
    ({"weekend_gpus": [0, 0]},              "repeat"),
    ({"weekday_gpus": [-1]},                "non-negative"),
    ({"start_day": 9},                      "[0, 6]"),
    ({"start_hour": 24},                    "[0, 23]"),
    ({"night_gpus": []},                    "at least one GPU"),
    ({"night_start_hour": 24},              "[0, 23]"),
    ({"start_day": 4, "start_hour": 18, "end_day": 4, "end_hour": 18}, "never switch"),
    ({"night_start_hour": 20, "night_end_hour": 20}, "never switch"),
])
def test_update_rejects_an_invalid_schedule(tmp_path, payload, reason):
    scheduler = GpuSchedule(StubPaths(tmp_path), WebLogger(), StubProcesses())
    before    = dict(scheduler.settings)

    result = scheduler.update({**GpuSchedule.DEFAULTS, **payload})

    assert result["ok"] is False
    assert reason in result["error"]
    assert scheduler.settings == before
    assert not scheduler.path.exists()


def test_update_rejects_a_partial_schedule(tmp_path):
    scheduler = GpuSchedule(StubPaths(tmp_path), WebLogger(), StubProcesses())

    result = scheduler.update({"enabled": True})

    assert result["ok"] is False
    assert "must define" in result["error"]


def test_an_unreadable_schedule_file_falls_back_to_defaults_loudly(tmp_path):
    paths = StubPaths(tmp_path)
    paths.logs_dir.mkdir(parents=True)
    (paths.logs_dir / GpuSchedule.FILE_NAME).write_text("{not json")

    scheduler = GpuSchedule(paths, WebLogger(), StubProcesses())

    assert scheduler.settings == GpuSchedule.DEFAULTS
