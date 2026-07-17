from __future__ import annotations

import json
import sys
import threading
import time
from datetime    import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib     import Path
from types       import SimpleNamespace

import pytest

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from command_listener import CommandListener
from notifier         import JobNotifier
from telegram_bot     import TelegramBot
from web_logger       import WebLogger

TOKEN = "123456789:" + "A" * 35


class StubPaths:

    def __init__(self, root: Path) -> None:
        self.logs_dir = root / "logs"


class StubBot:

    def __init__(self) -> None:
        self.paired   = "777"
        self.commands = True
        self.sent     = []
        self.updates  = []

    def chat_id(self) -> str:
        return self.paired

    def notify_enabled(self) -> bool:
        return False

    def commands_enabled(self) -> bool:
        return self.commands

    def send(self, text: str) -> str | None:
        self.sent.append(text)
        return None

    def send_async(self, text: str) -> None:
        self.sent.append(text)

    def get_updates(self, offset, timeout_s):
        if self.updates:
            return self.updates.pop(0)
        return [], None


class StubProcesses:

    def __init__(self, jobs: list | None = None) -> None:
        self.jobs           = jobs if jobs is not None else _default_jobs()
        self.stopped        = []
        self.stop_all_calls = 0
        self.cleared        = 0
        self.gpu_sets       = []
        self.stop_result    = {"ok": True}
        self.pool_result    = {"ok": True, "supported": True, "live": True, "gpus": [0, 1]}
        self.set_result     = {"ok": True, "gpus": [0], "parked": False}

    def list_jobs(self) -> list:
        return list(self.jobs)

    def stop(self, job_id: str) -> dict:
        self.stopped.append(job_id)
        return self.stop_result

    def stop_all(self, grace: float = 8.0) -> int:
        self.stop_all_calls += 1
        return 3

    def clear_queue(self) -> int:
        self.cleared += 1
        return 0

    def gpu_pool(self, job_id: str) -> dict:
        return self.pool_result

    def set_gpus(self, job_id: str, gpus, park: bool = False) -> dict:
        self.gpu_sets.append((job_id, gpus, park))
        return self.set_result


class StubNuke:

    def __init__(self) -> None:
        self.calls = 0

    def nuke(self, grace: float = 4.0) -> dict:
        self.calls += 1
        return {"ok": True, "signalled": 5, "killed": 1}


class StubSystem:

    def __init__(self) -> None:
        self.user = "bard"

    def gpu_occupancy(self) -> list:
        return [{"index": 0, "util": 92, "mem_used": 38912, "mem_total": 40960, "procs": [{"owner": "bard"}]}]


class TelegramStub(BaseHTTPRequestHandler):

    sent    = []
    pending = []

    def do_POST(self) -> None:
        length  = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")

        if self.path.endswith("/sendMessage"):
            TelegramStub.sent.append(payload)
            self._reply({"ok": True, "result": {"message_id": len(TelegramStub.sent)}})
            return

        if self.path.endswith("/getUpdates"):
            offset = payload.get("offset")
            if offset == -1:
                batch = TelegramStub.pending[-1:]
            elif offset is None:
                batch = list(TelegramStub.pending)
            else:
                batch = [update for update in TelegramStub.pending if update["update_id"] >= offset]
            if not batch:
                time.sleep(0.1)
            self._reply({"ok": True, "result": batch})
            return

        self._reply({"ok": False, "description": "unknown method"})

    def _reply(self, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        return


def _default_jobs() -> list:
    return [
        _job("abc123def456", "running", description="single training · resunet-conv"),
        _job("abd999000111", "queued", script="trials_backbone"),
    ]


def _job(job_id: str, status: str = "running", script: str = "train_backbone", description: str = "") -> dict:
    started = (datetime.now() - timedelta(seconds=600)).isoformat(timespec="seconds")
    return {"job_id": job_id, "script": script, "status": status, "started": started, "description": description}


def _update(update_id: int, chat_id: int | str, text: str, when: float | None = None) -> dict:
    return {"update_id": update_id, "message": {"chat": {"id": chat_id}, "date": when if when is not None else time.time(), "text": text}}


def _drain(calls: list, count: int, timeout: float = 5.0) -> list:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(calls) >= count:
            return calls
        time.sleep(0.02)
    return calls


@pytest.fixture
def rig():
    bot       = StubBot()
    logger    = WebLogger()
    notifier  = JobNotifier(bot, logger)
    processes = StubProcesses()
    nuke      = StubNuke()
    system    = StubSystem()
    listener  = CommandListener(bot, logger, notifier, processes, nuke, system)

    return SimpleNamespace(listener=listener, bot=bot, processes=processes, nuke=nuke, system=system)


@pytest.fixture
def stub_server():
    TelegramStub.sent    = []
    TelegramStub.pending = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), TelegramStub)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_empty_message_is_ignored(rig):
    assert rig.listener.handle("") is None
    assert rig.listener.handle("   ") is None


def test_help_lists_commands(rig):
    title, body = rig.listener.handle("help")
    assert title == "commands"
    for word in ("status", "stop", "nuke", "gpus", "confirm"):
        assert word in body


def test_unknown_command_points_to_help(rig):
    title, body = rig.listener.handle("reboot please")
    assert title == "unknown command"
    assert "reboot please" in body
    assert "help" in body


def test_status_reports_jobs_and_gpus(rig):
    title, body = rig.listener.handle("status")
    assert "1 running, 1 waiting" in title
    assert "abc123 train_backbone" in body
    assert "single training · resunet-conv" in body
    assert "abd999 trials_backbone · queued" in body
    assert "gpu0 92% 38/40GB · yours" in body


def test_status_with_no_jobs(rig):
    rig.processes.jobs = []
    title, body = rig.listener.handle("status")
    assert "0 running, 0 waiting" in title
    assert "no jobs running or queued" in body


def test_stop_by_unique_prefix(rig):
    title, body = rig.listener.handle("stop abc")
    assert title == "stop requested"
    assert rig.processes.stopped == ["abc123def456"]


def test_stop_ambiguous_prefix(rig):
    title, body = rig.listener.handle("stop ab")
    assert title == "stop failed"
    assert "ambiguous" in body
    assert rig.processes.stopped == []


def test_stop_unknown_prefix(rig):
    title, body = rig.listener.handle("stop zzz")
    assert title == "stop failed"
    assert rig.processes.stopped == []


def test_stop_propagates_backend_error(rig):
    rig.processes.stop_result = {"ok": False, "error": "job is not running"}
    title, body = rig.listener.handle("stop abc")
    assert title == "stop failed"
    assert body  == "job is not running"


def test_nuke_arms_without_executing(rig):
    title, body = rig.listener.handle("nuke")
    assert title == "nuke armed"
    assert "nuke confirm" in body
    assert rig.nuke.calls == 0


def test_nuke_confirm_executes(rig):
    rig.listener.handle("nuke")
    title, body = rig.listener.handle("nuke confirm")
    assert title == "nuke done"
    assert rig.nuke.calls        == 1
    assert rig.processes.cleared == 1


def test_confirm_without_arm_refuses(rig):
    title, body = rig.listener.handle("nuke confirm")
    assert title == "nothing armed"
    assert rig.nuke.calls == 0


def test_expired_confirm_refuses(rig):
    rig.listener.armed = {"action": "nuke", "deadline": time.monotonic() - 1.0}
    title, body = rig.listener.handle("nuke confirm")
    assert title == "nothing armed"
    assert rig.nuke.calls == 0


def test_confirm_is_single_use(rig):
    rig.listener.handle("nuke")
    rig.listener.handle("nuke confirm")
    title, body = rig.listener.handle("nuke confirm")
    assert title == "nothing armed"
    assert rig.nuke.calls == 1


def test_mismatched_confirm_refuses_and_disarms(rig):
    rig.listener.handle("nuke")
    title, body = rig.listener.handle("stop all confirm")
    assert title == "nothing armed"

    title, body = rig.listener.handle("nuke confirm")
    assert title == "nothing armed"
    assert rig.nuke.calls               == 0
    assert rig.processes.stop_all_calls == 0


def test_stop_all_arms_then_executes(rig):
    title, body = rig.listener.handle("stop all")
    assert title == "stop all armed"
    assert rig.processes.stop_all_calls == 0

    title, body = rig.listener.handle("stop all confirm")
    assert title == "stop all done"
    assert "3 jobs stopped" in body
    assert rig.processes.stop_all_calls == 1


def test_gpus_query_reports_pool(rig):
    title, body = rig.listener.handle("gpus abc")
    assert title == "gpu pool"
    assert "[0, 1]" in body


def test_gpus_query_without_live_pool(rig):
    rig.processes.pool_result = {"ok": True, "supported": True, "live": False}
    title, body = rig.listener.handle("gpus abc")
    assert title == "gpu pool"
    assert "no live gpu pool" in body


def test_gpus_resize(rig):
    rig.processes.set_result = {"ok": True, "gpus": [0, 2], "parked": False}
    title, body = rig.listener.handle("gpus abc 0,2")
    assert title == "gpu pool set"
    assert rig.processes.gpu_sets == [("abc123def456", [0, 2], False)]


def test_gpus_none_parks(rig):
    rig.processes.set_result = {"ok": True, "gpus": [], "parked": True}
    title, body = rig.listener.handle("gpus abc none")
    assert title == "gpu pool set"
    assert "parked" in body
    assert rig.processes.gpu_sets == [("abc123def456", [], True)]


def test_gpus_rejects_bad_list(rig):
    title, body = rig.listener.handle("gpus abc 0,x")
    assert title == "gpus failed"
    assert rig.processes.gpu_sets == []


def test_gpus_only_matches_running_jobs(rig):
    title, body = rig.listener.handle("gpus abd")
    assert title == "gpus failed"
    assert rig.processes.gpu_sets == []


def test_paired_update_is_answered(rig):
    rig.listener._handle_update(_update(1, 777, "help"))
    assert len(rig.bot.sent) == 1
    assert rig.bot.sent[0].startswith("commands\n")


def test_unpaired_chat_is_ignored(rig):
    rig.listener._handle_update(_update(1, 888, "help"))
    rig.listener._handle_update(_update(2, 888, "nuke confirm"))
    assert rig.bot.sent == []
    assert rig.nuke.calls == 0


def test_stale_update_is_dropped(rig):
    rig.listener._handle_update(_update(1, 777, "help", when=rig.listener.started - 100.0))
    assert rig.bot.sent == []


def test_update_without_text_is_ignored(rig):
    rig.listener._handle_update({"update_id": 1, "message": {"chat": {"id": 777}, "date": time.time()}})
    assert rig.bot.sent == []


def test_drain_skips_backlog(rig):
    rig.bot.updates = [([_update(4, 777, "old"), _update(7, 777, "older")], None)]
    assert rig.listener._drain() == 8


def test_drain_with_empty_backlog(rig):
    rig.bot.updates = [([], None)]
    assert rig.listener._drain() is None


def test_state_reflects_listener(rig):
    assert rig.listener.state()["listening"] is False
    rig.bot.updates = [([], None)] * 50
    rig.listener.apply()
    assert rig.listener.state()["listening"] is True

    rig.bot.commands = False
    rig.listener.apply()
    assert rig.listener.state()["listening"] is False


def test_end_to_end_over_http(tmp_path, stub_server):
    paths        = StubPaths(tmp_path)
    logger       = WebLogger()
    bot          = TelegramBot(paths, logger)
    bot.API_BASE = stub_server
    assert bot.configure({"commands_enabled": True, "token": TOKEN, "chat_id": "777"})["ok"]

    notifier  = JobNotifier(bot, logger)
    processes = StubProcesses()
    listener  = CommandListener(bot, logger, notifier, processes, StubNuke(), StubSystem())

    TelegramStub.pending = [_update(1, 777, "status", when=time.time() - 1000.0)]
    listener.apply()
    assert listener.state()["listening"] is True

    time.sleep(0.5)
    TelegramStub.pending.append(_update(2, 777, "help"))

    sent = _drain(TelegramStub.sent, 1, timeout=8.0)
    assert bot.configure({"commands_enabled": False, "token": TOKEN, "chat_id": "777"})["ok"]
    listener.apply()
    assert listener.state()["listening"] is False

    assert len(sent) == 1
    assert sent[0]["chat_id"] == "777"
    assert sent[0]["text"].startswith("commands\n")
