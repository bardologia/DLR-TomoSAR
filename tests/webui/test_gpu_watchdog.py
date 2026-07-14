from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from gpu_watchdog import GpuWatchdog
from web_logger   import WebLogger


class StubSystem:

    def __init__(self) -> None:
        self.user      = "me"
        self.occupancy = []
        self.owners    = {}

    def gpu_occupancy(self) -> list[dict]:
        return [{**device, "procs": [dict(proc) for proc in device["procs"]]} for device in self.occupancy]

    def pid_owner(self, pid: int) -> str | None:
        return self.owners.get(pid)


class StubPaths:

    def __init__(self, root: Path) -> None:
        self.gpu_guard_dir = root


class StubProcesses:

    def __init__(self) -> None:
        self.jobs  = {}
        self.fates = {}

    def job_for_pid(self, pid: int) -> str | None:
        return self.jobs.get(pid)

    def job_fate(self, job_id: str, pid: int) -> str:
        return self.fates.get(job_id, "unknown")


def _device(procs: list[dict]) -> dict:
    return {"index": 0, "name": "A100", "util": 50.0, "mem_used": 1000.0, "mem_total": 40000.0, "uuid": "GPU-0", "procs": procs}


def _proc(pid: int, owner: str) -> dict:
    return {"pid": pid, "owner": owner, "mem": 1000.0, "cmd": f"python train_{pid}.py"}


def _watchdog(tmp_path: Path) -> tuple[GpuWatchdog, StubSystem, StubProcesses]:
    system    = StubSystem()
    processes = StubProcesses()
    watchdog  = GpuWatchdog(system, StubPaths(tmp_path), WebLogger(), processes)
    return watchdog, system, processes


def _invaded(watchdog: GpuWatchdog, system: StubSystem) -> None:
    system.occupancy = [_device([_proc(100, "me")])]
    system.owners    = {100: "me", 200: "them"}
    watchdog._evaluate()

    system.occupancy = [_device([_proc(100, "me"), _proc(200, "them")])]
    watchdog._evaluate()


def test_shared_gpu_at_startup_is_not_an_intrusion(tmp_path):
    watchdog, system, _ = _watchdog(tmp_path)

    system.occupancy = [_device([_proc(100, "me"), _proc(200, "them")])]
    system.owners    = {100: "me", 200: "them"}
    watchdog._evaluate()

    assert watchdog.state()["active"] == []
    assert watchdog.state()["count"]  == 0


def test_joining_an_occupied_gpu_is_not_an_intrusion(tmp_path):
    watchdog, system, _ = _watchdog(tmp_path)

    system.occupancy = [_device([_proc(200, "them")])]
    system.owners    = {200: "them"}
    watchdog._evaluate()

    system.occupancy = [_device([_proc(100, "me"), _proc(200, "them")])]
    system.owners    = {100: "me", 200: "them"}
    watchdog._evaluate()

    assert watchdog.state()["active"] == []
    assert watchdog.state()["count"]  == 0


def test_late_arrival_on_my_gpu_is_an_intrusion(tmp_path):
    watchdog, system, _ = _watchdog(tmp_path)

    _invaded(watchdog, system)

    state = watchdog.state()
    assert state["count"] == 1
    assert [record["user"] for record in state["active"]] == ["them"]
    assert state["active"][0]["mine_pids"] == [100]


def test_clean_finish_during_intrusion_is_not_critical(tmp_path):
    watchdog, system, processes = _watchdog(tmp_path)
    processes.jobs  = {100: "j1"}
    processes.fates = {"j1": "finished"}

    _invaded(watchdog, system)

    system.occupancy = [_device([_proc(200, "them")])]
    system.owners    = {200: "them"}
    watchdog._evaluate()

    state = watchdog.state()
    assert state["critical"] == 0
    assert state["active"]   == []
    assert watchdog.incidents[("GPU-0", 200)]["status"] == "ended"


def test_crash_during_intrusion_window_is_critical(tmp_path):
    watchdog, system, processes = _watchdog(tmp_path)
    processes.jobs  = {100: "j1"}
    processes.fates = {"j1": "crashed"}

    _invaded(watchdog, system)

    system.occupancy = [_device([_proc(200, "them")])]
    system.owners    = {200: "them"}
    watchdog._evaluate()

    state = watchdog.state()
    assert state["critical"] == 1
    assert state["active"][0]["status"]    == "critical"
    assert state["active"][0]["dead_pids"] == [100]


def test_user_stopped_job_is_not_critical(tmp_path):
    watchdog, system, processes = _watchdog(tmp_path)
    processes.jobs  = {100: "j1"}
    processes.fates = {"j1": "stopped"}

    _invaded(watchdog, system)

    system.occupancy = [_device([_proc(200, "them")])]
    system.owners    = {200: "them"}
    watchdog._evaluate()

    state = watchdog.state()
    assert state["critical"] == 0
    assert watchdog.incidents[("GPU-0", 200)]["status"] == "ended"


def test_untracked_pid_exit_is_not_critical(tmp_path):
    watchdog, system, _ = _watchdog(tmp_path)

    _invaded(watchdog, system)

    system.occupancy = [_device([_proc(200, "them")])]
    system.owners    = {200: "them"}
    watchdog._evaluate()

    state = watchdog.state()
    assert state["critical"] == 0
    assert watchdog.incidents[("GPU-0", 200)]["status"] == "ended"


def test_pending_fate_defers_until_exit_resolves(tmp_path):
    watchdog, system, processes = _watchdog(tmp_path)
    processes.jobs  = {100: "j1"}
    processes.fates = {"j1": "pending"}

    _invaded(watchdog, system)

    system.occupancy = [_device([_proc(200, "them")])]
    system.owners    = {200: "them"}
    watchdog._evaluate()
    assert watchdog.incidents[("GPU-0", 200)]["status"] == "active"

    processes.fates = {"j1": "crashed"}
    watchdog._evaluate()

    state = watchdog.state()
    assert state["critical"] == 1
    assert state["active"][0]["dead_pids"] == [100]
