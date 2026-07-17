from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from notifier        import JobNotifier
from telegram_bot    import TelegramBot
from process_manager import ProcessManager
from web_logger      import WebLogger

SLEEP_OK   = "import time\ntime.sleep(0.6)\n"
SLEEP_FAIL = "import sys, time\ntime.sleep(0.6)\nsys.exit(3)\n"
SLEEP_LONG = "import time\ntime.sleep(30)\n"
WRITER     = "import pathlib, sys\npathlib.Path('order.txt').open('a').write(pathlib.Path(__file__).stem + '\\n')\n"
ARGS_DUMP  = "import pathlib, sys\npathlib.Path('argv.txt').write_text(' '.join(sys.argv[1:]))\n"


class StubPaths:

    def __init__(self, root: Path) -> None:
        self.repo_root = root
        self.main_dir  = root / "main"
        self.logs_dir  = root / "logs"

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
    (scripts / "sleep_ok.py").write_text(SLEEP_OK)
    (scripts / "sleep_fail.py").write_text(SLEEP_FAIL)
    (scripts / "sleep_long.py").write_text(SLEEP_LONG)
    (scripts / "writer_a.py").write_text(WRITER)
    (scripts / "writer_b.py").write_text(WRITER)
    (scripts / "args_dump.py").write_text(ARGS_DUMP)

    paths  = StubPaths(tmp_path)
    logger = WebLogger()
    yield ProcessManager(paths, logger, JobNotifier(TelegramBot(paths, logger), logger), StubDescriber())


def _record(manager: ProcessManager, job_id: str) -> dict:
    with manager.lock:
        return dict(manager.jobs[job_id])


def _wait_status(manager: ProcessManager, job_id: str, status: str, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _record(manager, job_id)["status"] == status:
            return True
        time.sleep(0.05)
    return False


def test_enqueue_idle_starts_immediately(manager):
    result = manager.enqueue("sleep_ok", sys.executable)

    assert result["ok"]
    assert result["queued"] is False
    assert _wait_status(manager, result["job_id"], "finished")


def test_enqueue_waits_for_running_job(manager):
    running = manager.launch("sleep_ok", sys.executable)
    queued  = manager.enqueue("sleep_ok", sys.executable)

    assert queued["ok"]
    assert queued["queued"] is True
    assert _record(manager, queued["job_id"])["status"] == "queued"

    assert _wait_status(manager, queued["job_id"], "finished")
    assert _record(manager, running["job_id"])["status"] == "finished"


def test_queued_job_starts_after_predecessor_fails(manager):
    failing = manager.launch("sleep_fail", sys.executable)
    queued  = manager.enqueue("sleep_ok", sys.executable)

    assert _wait_status(manager, queued["job_id"], "finished")
    assert _record(manager, failing["job_id"])["status"] == "failed"


def test_queued_jobs_run_in_order(manager, tmp_path):
    manager.launch("sleep_ok", sys.executable)
    first  = manager.enqueue("writer_a", sys.executable)
    second = manager.enqueue("writer_b", sys.executable)

    assert _wait_status(manager, first["job_id"], "finished")
    assert _wait_status(manager, second["job_id"], "finished")
    assert (tmp_path / "order.txt").read_text().splitlines() == ["writer_a", "writer_b"]


def test_queued_launch_keeps_overrides(manager, tmp_path):
    manager.launch("sleep_ok", sys.executable)
    queued = manager.enqueue("args_dump", sys.executable, {"training.seed": "7"})

    assert _wait_status(manager, queued["job_id"], "finished")
    assert (tmp_path / "argv.txt").read_text() == "--training.seed 7"


def test_cancelled_queued_job_is_skipped(manager, tmp_path):
    manager.launch("sleep_ok", sys.executable)
    first  = manager.enqueue("writer_a", sys.executable)
    second = manager.enqueue("writer_b", sys.executable)

    assert manager.stop(first["job_id"])["ok"]
    assert _record(manager, first["job_id"])["status"] == "cancelled"

    assert _wait_status(manager, second["job_id"], "finished")
    assert _record(manager, first["job_id"])["status"] == "cancelled"
    assert (tmp_path / "order.txt").read_text().splitlines() == ["writer_b"]


def test_stop_all_purges_queue(manager):
    running = manager.launch("sleep_long", sys.executable)
    queued  = manager.enqueue("sleep_ok", sys.executable)

    assert manager.stop_all() == 1
    assert _record(manager, queued["job_id"])["status"] == "cancelled"
    assert _record(manager, queued["job_id"])["pid"] is None

    assert _wait_status(manager, running["job_id"], "failed")
    assert _record(manager, queued["job_id"])["status"] == "cancelled"


def test_queue_respects_follow_up_chain(manager, tmp_path):
    manager.launch("sleep_ok", sys.executable)
    parent = manager.enqueue("writer_a", sys.executable, follow_up="args_dump")
    last   = manager.enqueue("writer_b", sys.executable)

    assert _wait_status(manager, last["job_id"], "finished")

    follow_id = _record(manager, parent["job_id"])["follow_up"]
    assert follow_id is not None
    assert _record(manager, follow_id)["status"] == "finished"

    order = (tmp_path / "order.txt").read_text().splitlines()
    assert order == ["writer_a", "writer_b"]
    assert (tmp_path / "argv.txt").exists()


def test_enqueue_rejects_missing_script(manager):
    result = manager.enqueue("missing", sys.executable)
    assert result == {"ok": False, "error": "script not found"}


def test_notifications_fire_on_start_and_finish_for_direct_and_queued(manager):
    events = []
    manager.notifier.job_started  = lambda record: events.append(("started", record["script"]))
    manager.notifier.job_finished = lambda record: events.append(("finished", record["script"]))

    manager.launch("sleep_ok", sys.executable)
    queued = manager.enqueue("writer_a", sys.executable)

    assert _wait_status(manager, queued["job_id"], "finished")
    assert events == [("started", "sleep_ok"), ("finished", "sleep_ok"), ("started", "writer_a"), ("finished", "writer_a")]
