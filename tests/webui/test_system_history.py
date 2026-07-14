from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from system_monitor import SystemHistory, SystemMonitor

FAKE_DEVICES = [
    {"index": 0, "util": 50,   "mem_used": 8000, "mem_total": 16000, "name": "GPU A", "temp": 40, "power": 30, "power_limit": 90, "uuid": "GPU-a"},
    {"index": 1, "util": None, "mem_used": 0,    "mem_total": 0,     "name": "GPU B", "temp": 40, "power": 30, "power_limit": 90, "uuid": "GPU-b"},
]


class StubPaths:

    def __init__(self, root: Path) -> None:
        self.repo_root = root


@pytest.fixture
def monitor(tmp_path, monkeypatch):
    monkeypatch.setattr(SystemMonitor, "_du_loop", lambda self: None)
    monkeypatch.setattr(SystemHistory, "sample_loop", lambda self: None)
    instance = SystemMonitor(StubPaths(tmp_path))
    monkeypatch.setattr(instance, "_gpu_devices", lambda: FAKE_DEVICES)
    return instance


@pytest.fixture
def history(monitor):
    return SystemHistory(monitor)


def test_sample_records_all_series(history):
    history.sample()
    history.sample()

    state = history.state()
    assert len(state["cpu"]) == 2
    assert len(state["ram"]) == 2
    assert set(state["gpus"]) == {"0", "1"}
    assert state["gpus"]["0"]["util"] == [50, 50]
    assert state["gpus"]["0"]["mem"]  == [50.0, 50.0]
    assert state["gpus"]["1"]["util"] == [0.0, 0.0]
    assert state["gpus"]["1"]["mem"]  == [0.0, 0.0]


def test_samples_are_bounded_percentages(history):
    history.sample()
    history.sample()

    state = history.state()
    for series in (state["cpu"], state["ram"]):
        assert all(isinstance(value, float) and 0.0 <= value <= 100.0 for value in series)


def test_ring_buffer_caps_history(monitor, monkeypatch):
    monkeypatch.setattr(SystemHistory, "MAX_SAMPLES", 5)
    capped = SystemHistory(monitor)

    for _ in range(8):
        capped.sample()

    state = capped.state()
    assert len(state["cpu"]) == 5
    assert len(state["gpus"]["0"]["util"]) == 5
    assert state["max_samples"] == 5


def test_history_state_is_independent_of_monitor_snapshots(history, monitor):
    history.sample()
    monitor.snapshot()
    history.sample()

    assert len(history.state()["cpu"]) == 2


def test_snapshot_embeds_history(monitor):
    snapshot = monitor.snapshot()

    assert "history" in snapshot
    assert set(snapshot["history"]) == {"period_s", "max_samples", "cpu", "ram", "gpus"}
