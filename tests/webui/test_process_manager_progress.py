from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from notifier        import JobNotifier
from process_manager import ProcessManager
from web_logger      import WebLogger

from tools.orchestration.gpu_queue import GpuProgressFile

SLEEP_LONG = "import time\ntime.sleep(30)\n"


class StubPaths:

    def __init__(self, root: Path) -> None:
        self.repo_root     = root
        self.main_dir      = root / "main"
        self.logs_dir      = root / "logs"
        self.gpu_pools_dir = root / "logs" / "gpu_pools"

    def has_script(self, key: str) -> bool:
        return (self.main_dir / "analysis" / f"{key}.py").exists()

    def script_entry(self, key: str) -> dict:
        path = self.main_dir / "analysis" / f"{key}.py"
        return {"path": path, "rel": f"main/analysis/{key}.py", "args": []}


class StubDescriber:

    def describe(self, key: str, interpreter: str, overrides: dict | None) -> str:
        return f"stub description for {key}"


@pytest.fixture
def manager(tmp_path):
    scripts = tmp_path / "main" / "analysis"
    scripts.mkdir(parents=True)
    (scripts / "train_backbone.py").write_text(SLEEP_LONG)
    (scripts / "train_jepa.py").write_text(SLEEP_LONG)

    paths  = StubPaths(tmp_path)
    logger = WebLogger()
    yield ProcessManager(paths, logger, JobNotifier(paths, logger), StubDescriber())


def _wait_running(manager: ProcessManager, job_id: str, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with manager.lock:
            if manager.jobs[job_id]["status"] == "running":
                return True
        time.sleep(0.05)
    return False


def _wait_done(manager: ProcessManager, job_id: str, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with manager.lock:
            if manager.jobs[job_id]["status"] != "running":
                return True
        time.sleep(0.05)
    return False


def _progress_path(manager: ProcessManager, job_id: str) -> Path:
    with manager.lock:
        return GpuProgressFile.resolve(Path(manager.jobs[job_id]["overrides"]["gpus_file"]))


def _snapshot(done: int = 12, failed: int = 1, total: int = 30) -> dict:
    return {
        "total"        : total,
        "done"         : done,
        "failed"       : failed,
        "queued"       : total - done - failed - 2,
        "running"      : [{"name": "aug-on/seed3", "gpu": 0, "elapsed_s": 310.0}, {"name": "aug-off/seed1", "gpu": 1, "elapsed_s": 95.0}],
        "workers"      : 2,
        "failed_units" : ["aug-off/seed0"] if failed else [],
        "average_s"    : 600.0,
        "elapsed_s"    : 4200.0,
        "eta_s"        : 5400.0,
        "total_s"      : 9600.0,
        "started_at"   : "2026-07-17T10:00:00",
        "finish_at"    : "2026-07-17T14:30:00",
        "updated_at"   : "2026-07-17T13:00:00",
    }


def test_progress_reports_unsupported_for_non_pool_scripts(manager):
    result = manager.launch("train_jepa", sys.executable)

    assert manager.progress(result["job_id"]) == {"ok": True, "supported": False, "live": False}

    manager.stop(result["job_id"])


def test_progress_before_the_file_exists_is_not_live(manager):
    result = manager.launch("train_backbone", sys.executable)
    job_id = result["job_id"]

    assert _wait_running(manager, job_id)

    info = manager.progress(job_id)

    assert info["ok"] is True
    assert info["supported"] is True
    assert info["live"] is False
    assert "progress" not in info
    assert info["path"].endswith(f"{job_id}_progress.json")

    manager.stop(job_id)


def test_progress_reads_the_live_snapshot(manager):
    result = manager.launch("train_backbone", sys.executable)
    job_id = result["job_id"]

    assert _wait_running(manager, job_id)

    path = _progress_path(manager, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_snapshot()))

    info = manager.progress(job_id)

    assert info["ok"] is True
    assert info["live"] is True
    assert info["progress"] == _snapshot()

    manager.stop(job_id)


def test_list_jobs_embeds_the_progress_of_running_fan_outs(manager):
    result = manager.launch("train_backbone", sys.executable)
    job_id = result["job_id"]

    assert _wait_running(manager, job_id)

    path = _progress_path(manager, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_snapshot()))

    record = next(r for r in manager.list_jobs() if r["job_id"] == job_id)

    assert record["progress"] == _snapshot()

    manager.stop(job_id)


def test_list_jobs_leaves_other_jobs_without_progress(manager):
    result = manager.launch("train_jepa", sys.executable)
    job_id = result["job_id"]

    assert _wait_running(manager, job_id)

    record = next(r for r in manager.list_jobs() if r["job_id"] == job_id)

    assert record["progress"] is None

    manager.stop(job_id)


def test_progress_of_a_finished_job_keeps_the_snapshot_but_is_not_live(manager):
    result = manager.launch("train_backbone", sys.executable)
    job_id = result["job_id"]

    assert _wait_running(manager, job_id)

    path = _progress_path(manager, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_snapshot(done=30, failed=0)))

    manager.stop(job_id)
    assert _wait_done(manager, job_id)

    info = manager.progress(job_id)

    assert info["live"] is False
    assert info["progress"]["done"] == 30

    record = next(r for r in manager.list_jobs() if r["job_id"] == job_id)
    assert record["progress"] is None


def test_progress_rejects_an_unreadable_file(manager):
    result = manager.launch("train_backbone", sys.executable)
    job_id = result["job_id"]

    assert _wait_running(manager, job_id)

    path = _progress_path(manager, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{oops")

    info = manager.progress(job_id)

    assert info["ok"] is False
    assert "unreadable progress file" in info["error"]

    manager.stop(job_id)


def test_progress_reports_an_unknown_job(manager):
    assert manager.progress("nope") == {"ok": False, "error": "unknown job"}
