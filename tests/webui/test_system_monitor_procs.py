from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from system_monitor import ActiveUsers, SystemHistory, SystemMonitor


class StubPaths:

    def __init__(self, root: Path) -> None:
        self.repo_root = root


@pytest.fixture
def monitor(tmp_path, monkeypatch):
    monkeypatch.setattr(SystemMonitor, "_du_loop", lambda self: None)
    monkeypatch.setattr(SystemHistory, "sample_loop", lambda self: None)
    monkeypatch.setattr(ActiveUsers, "sample_loop", lambda self: None)
    return SystemMonitor(StubPaths(tmp_path))


def test_rows_report_thread_counts(monitor):
    rows = monitor._procs({})

    assert rows
    assert all(isinstance(row["threads"], int) and row["threads"] >= 1 for row in rows)


def test_own_process_thread_count_tracks_spawned_threads(monitor):
    release = threading.Event()
    workers = [threading.Thread(target=release.wait, daemon=True) for _ in range(5)]
    for worker in workers:
        worker.start()

    try:
        monitor.PROC_LIMIT = 100000
        row = next(r for r in monitor._procs({}) if r["pid"] == os.getpid())
        assert row["threads"] >= 6
    finally:
        release.set()
        for worker in workers:
            worker.join()
