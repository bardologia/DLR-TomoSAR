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
from web_logger       import WebLogger


class StubPaths:

    def __init__(self, root: Path) -> None:
        self.logs_dir = root / "logs"


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


class NtfyStub(BaseHTTPRequestHandler):

    posts   = []
    streams = 0

    def do_GET(self) -> None:
        if self.path != "/cmd-topic/json":
            self.send_response(404)
            self.end_headers()
            return

        NtfyStub.streams += 1
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        self.wfile.write((json.dumps({"event": "open"}) + "\n").encode("utf-8"))
        self.wfile.flush()

        if NtfyStub.streams == 1:
            message = {"id": "m1", "time": time.time(), "event": "message", "topic": "cmd-topic", "message": "help"}
            self.wfile.write((json.dumps(message) + "\n").encode("utf-8"))
            self.wfile.flush()

        time.sleep(0.5)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        NtfyStub.posts.append({
            "path"     : self.path,
            "title"    : self.headers.get("Title"),
            "priority" : self.headers.get("Priority"),
            "body"     : self.rfile.read(length).decode("utf-8"),
        })
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

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


def _drain(calls: list, count: int, timeout: float = 5.0) -> list:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(calls) >= count:
            return calls
        time.sleep(0.02)
    return calls


def _event_line(message: str, when: float) -> bytes:
    return (json.dumps({"id": "m1", "time": when, "event": "message", "topic": "cmd-topic", "message": message}) + "\n").encode("utf-8")


@pytest.fixture
def rig(tmp_path, monkeypatch):
    paths    = StubPaths(tmp_path)
    logger   = WebLogger()
    notifier = JobNotifier(paths, logger)
    assert notifier.configure({"enabled": False, "topic": "reply-topic"})["ok"]

    processes = StubProcesses()
    nuke      = StubNuke()
    system    = StubSystem()
    listener  = CommandListener(paths, logger, notifier, processes, nuke, system)
    monkeypatch.setattr(listener, "_apply", lambda: None)

    return SimpleNamespace(listener=listener, notifier=notifier, processes=processes, nuke=nuke, system=system, paths=paths)


@pytest.fixture
def pushed(rig, monkeypatch):
    calls = []
    monkeypatch.setattr(rig.notifier, "push", lambda title, body, priority="default": calls.append((title, body, priority)))
    return calls


@pytest.fixture
def ntfy_server():
    NtfyStub.posts   = []
    NtfyStub.streams = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), NtfyStub)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_defaults_are_disabled(rig):
    state = rig.listener.state()
    assert state["enabled"]   is False
    assert state["topic"]     == ""
    assert state["server"]    == "https://ntfy.sh"
    assert state["listening"] is False


def test_configure_persists_across_instances(rig, tmp_path):
    assert rig.listener.configure({"enabled": True, "topic": "cmd-topic"})["ok"]

    reloaded = CommandListener(rig.paths, WebLogger(), rig.notifier, rig.processes, rig.nuke, rig.system)
    state    = reloaded.state()
    assert state["enabled"] is True
    assert state["topic"]   == "cmd-topic"


def test_configure_rejects_invalid_topic(rig):
    assert not rig.listener.configure({"enabled": True, "topic": "has spaces"})["ok"]
    assert not rig.listener.configure({"enabled": True, "topic": "a/b"})["ok"]
    assert not rig.listener.configure({"enabled": True, "topic": "x" * 65})["ok"]


def test_configure_requires_topic_to_enable(rig):
    assert not rig.listener.configure({"enabled": True, "topic": ""})["ok"]
    assert rig.listener.configure({"enabled": False, "topic": ""})["ok"]


def test_configure_rejects_invalid_server(rig):
    assert not rig.listener.configure({"enabled": True, "topic": "cmd-topic", "server": "ntfy.sh"})["ok"]


def test_configure_rejects_notify_topic_reuse(rig):
    result = rig.listener.configure({"enabled": True, "topic": "reply-topic"})
    assert not result["ok"]
    assert "differ" in result["error"]


def test_configure_requires_notify_topic(tmp_path, monkeypatch):
    paths    = StubPaths(tmp_path)
    notifier = JobNotifier(paths, WebLogger())
    listener = CommandListener(paths, WebLogger(), notifier, StubProcesses(), StubNuke(), StubSystem())
    monkeypatch.setattr(listener, "_apply", lambda: None)

    result = listener.configure({"enabled": True, "topic": "cmd-topic"})
    assert not result["ok"]
    assert "notification topic" in result["error"]


def test_empty_message_is_ignored(rig):
    assert rig.listener.handle("") is None
    assert rig.listener.handle("   ") is None


def test_help_lists_commands(rig):
    title, body, priority = rig.listener.handle("help")
    assert title == "commands"
    for word in ("status", "stop", "nuke", "gpus", "confirm"):
        assert word in body


def test_unknown_command_points_to_help(rig):
    title, body, priority = rig.listener.handle("reboot please")
    assert title == "unknown command"
    assert "reboot please" in body
    assert "help" in body


def test_status_reports_jobs_and_gpus(rig):
    title, body, priority = rig.listener.handle("status")
    assert "1 running, 1 waiting" in title
    assert "abc123 train_backbone" in body
    assert "single training · resunet-conv" in body
    assert "abd999 trials_backbone · queued" in body
    assert "gpu0 92% 38/40GB · yours" in body


def test_status_with_no_jobs(rig):
    rig.processes.jobs = []
    title, body, priority = rig.listener.handle("status")
    assert "0 running, 0 waiting" in title
    assert "no jobs running or queued" in body


def test_stop_by_unique_prefix(rig):
    title, body, priority = rig.listener.handle("stop abc")
    assert title == "stop requested"
    assert rig.processes.stopped == ["abc123def456"]


def test_stop_ambiguous_prefix(rig):
    title, body, priority = rig.listener.handle("stop ab")
    assert title == "stop failed"
    assert "ambiguous" in body
    assert rig.processes.stopped == []


def test_stop_unknown_prefix(rig):
    title, body, priority = rig.listener.handle("stop zzz")
    assert title == "stop failed"
    assert "no" in body
    assert rig.processes.stopped == []


def test_stop_propagates_backend_error(rig):
    rig.processes.stop_result = {"ok": False, "error": "job is not running"}
    title, body, priority = rig.listener.handle("stop abc")
    assert title == "stop failed"
    assert body  == "job is not running"


def test_nuke_arms_without_executing(rig):
    title, body, priority = rig.listener.handle("nuke")
    assert title    == "nuke armed"
    assert priority == "high"
    assert "nuke confirm" in body
    assert rig.nuke.calls == 0


def test_nuke_confirm_executes(rig):
    rig.listener.handle("nuke")
    title, body, priority = rig.listener.handle("nuke confirm")
    assert title    == "nuke done"
    assert priority == "high"
    assert rig.nuke.calls        == 1
    assert rig.processes.cleared == 1


def test_confirm_without_arm_refuses(rig):
    title, body, priority = rig.listener.handle("nuke confirm")
    assert title == "nothing armed"
    assert rig.nuke.calls == 0


def test_expired_confirm_refuses(rig):
    rig.listener.armed = {"action": "nuke", "deadline": time.monotonic() - 1.0}
    title, body, priority = rig.listener.handle("nuke confirm")
    assert title == "nothing armed"
    assert rig.nuke.calls == 0


def test_confirm_is_single_use(rig):
    rig.listener.handle("nuke")
    rig.listener.handle("nuke confirm")
    title, body, priority = rig.listener.handle("nuke confirm")
    assert title == "nothing armed"
    assert rig.nuke.calls == 1


def test_mismatched_confirm_refuses_and_disarms(rig):
    rig.listener.handle("nuke")
    title, body, priority = rig.listener.handle("stop all confirm")
    assert title == "nothing armed"

    title, body, priority = rig.listener.handle("nuke confirm")
    assert title == "nothing armed"
    assert rig.nuke.calls            == 0
    assert rig.processes.stop_all_calls == 0


def test_stop_all_arms_then_executes(rig):
    title, body, priority = rig.listener.handle("stop all")
    assert title == "stop all armed"
    assert rig.processes.stop_all_calls == 0

    title, body, priority = rig.listener.handle("stop all confirm")
    assert title == "stop all done"
    assert "3 jobs stopped" in body
    assert rig.processes.stop_all_calls == 1


def test_gpus_query_reports_pool(rig):
    title, body, priority = rig.listener.handle("gpus abc")
    assert title == "gpu pool"
    assert "[0, 1]" in body


def test_gpus_query_without_live_pool(rig):
    rig.processes.pool_result = {"ok": True, "supported": True, "live": False}
    title, body, priority = rig.listener.handle("gpus abc")
    assert title == "gpu pool"
    assert "no live gpu pool" in body


def test_gpus_resize(rig):
    rig.processes.set_result = {"ok": True, "gpus": [0, 2], "parked": False}
    title, body, priority = rig.listener.handle("gpus abc 0,2")
    assert title == "gpu pool set"
    assert rig.processes.gpu_sets == [("abc123def456", [0, 2], False)]


def test_gpus_none_parks(rig):
    rig.processes.set_result = {"ok": True, "gpus": [], "parked": True}
    title, body, priority = rig.listener.handle("gpus abc none")
    assert title == "gpu pool set"
    assert "parked" in body
    assert rig.processes.gpu_sets == [("abc123def456", [], True)]


def test_gpus_rejects_bad_list(rig):
    title, body, priority = rig.listener.handle("gpus abc 0,x")
    assert title == "gpus failed"
    assert rig.processes.gpu_sets == []


def test_gpus_only_matches_running_jobs(rig):
    title, body, priority = rig.listener.handle("gpus abd")
    assert title == "gpus failed"
    assert rig.processes.gpu_sets == []


def test_fresh_message_is_dispatched(rig, pushed):
    rig.listener._handle_line(_event_line("help", time.time()))
    assert len(pushed) == 1
    assert pushed[0][0] == "commands"


def test_stale_message_is_dropped(rig, pushed):
    rig.listener._handle_line(_event_line("help", rig.listener.started - 100.0))
    assert pushed == []


def test_non_message_events_are_ignored(rig, pushed):
    rig.listener._handle_line(b'{"event": "open"}\n')
    rig.listener._handle_line(b'{"event": "keepalive"}\n')
    rig.listener._handle_line(b"not json\n")
    rig.listener._handle_line(b"\n")
    assert pushed == []


def test_topic_clash_drops_message(rig, pushed):
    with rig.listener.lock:
        rig.listener.settings = {"enabled": True, "topic": "reply-topic", "server": "https://ntfy.sh"}
    rig.listener._handle_line(_event_line("help", time.time()))
    assert pushed == []


def test_end_to_end_over_http(tmp_path, ntfy_server):
    paths    = StubPaths(tmp_path)
    logger   = WebLogger()
    notifier = JobNotifier(paths, logger)
    assert notifier.configure({"enabled": False, "topic": "reply-topic", "server": ntfy_server})["ok"]

    listener = CommandListener(paths, logger, notifier, StubProcesses(), StubNuke(), StubSystem())
    assert listener.configure({"enabled": True, "topic": "cmd-topic", "server": ntfy_server})["ok"]
    assert listener.state()["listening"] is True

    posts = _drain(NtfyStub.posts, 1, timeout=8.0)
    assert listener.configure({"enabled": False, "topic": "cmd-topic", "server": ntfy_server})["ok"]
    assert listener.state()["listening"] is False

    assert len(posts) >= 1
    assert posts[0]["path"]  == "/reply-topic"
    assert posts[0]["title"] == "commands"
