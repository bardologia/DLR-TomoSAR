from __future__ import annotations

import sys
import threading
from datetime    import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib     import Path

import pytest

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from notifier   import JobNotifier
from web_logger import WebLogger


class StubPaths:

    def __init__(self, root: Path) -> None:
        self.logs_dir = root / "logs"


class CaptureHandler(BaseHTTPRequestHandler):

    requests = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        CaptureHandler.requests.append({
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


@pytest.fixture
def notifier(tmp_path):
    return JobNotifier(StubPaths(tmp_path), WebLogger())


@pytest.fixture
def sent(notifier, monkeypatch):
    calls = []
    monkeypatch.setattr(notifier, "_send", lambda title, body, priority: calls.append((title, body, priority)) or None)
    return calls


@pytest.fixture
def capture_server():
    CaptureHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _record(status: str = "finished", exit_code: int | None = 0, ago_s: float = 3600.0, stopped: bool = False) -> dict:
    started = (datetime.now() - timedelta(seconds=ago_s)).isoformat(timespec="seconds")
    record  = {"job_id": "abc123def456", "script": "train_backbone", "status": status, "exit_code": exit_code, "started": started}
    if stopped:
        record["stopped"] = True
    return record


def _arm(notifier, min_runtime_s: float = 60.0) -> None:
    result = notifier.configure({"enabled": True, "topic": "unit-test-topic", "server": "http://127.0.0.1:9", "min_runtime_s": min_runtime_s})
    assert result["ok"]


def test_defaults_are_disabled(notifier):
    state = notifier.state()
    assert state["enabled"] is False
    assert state["topic"]   == ""
    assert state["server"]  == "https://ntfy.sh"


def test_configure_persists_across_instances(notifier, tmp_path):
    _arm(notifier, min_runtime_s=300.0)

    reloaded = JobNotifier(StubPaths(tmp_path), WebLogger())
    state    = reloaded.state()
    assert state["enabled"]       is True
    assert state["topic"]         == "unit-test-topic"
    assert state["min_runtime_s"] == 300.0


def test_configure_rejects_invalid_topic(notifier):
    assert not notifier.configure({"enabled": True, "topic": "has spaces"})["ok"]
    assert not notifier.configure({"enabled": True, "topic": "a/b"})["ok"]
    assert not notifier.configure({"enabled": True, "topic": "x" * 65})["ok"]


def test_configure_requires_topic_to_enable(notifier):
    assert not notifier.configure({"enabled": True, "topic": ""})["ok"]
    assert notifier.configure({"enabled": False, "topic": ""})["ok"]


def test_configure_rejects_invalid_server(notifier):
    assert not notifier.configure({"enabled": True, "topic": "ok-topic", "server": "ntfy.sh"})["ok"]


def test_disabled_notifier_sends_nothing(notifier, sent):
    notifier.job_finished(_record())
    assert sent == []


def test_short_success_is_skipped(notifier, sent):
    _arm(notifier, min_runtime_s=60.0)
    notifier.job_finished(_record(ago_s=5.0))
    assert sent == []


def test_long_success_notifies(notifier, sent):
    _arm(notifier, min_runtime_s=60.0)
    notifier.job_finished(_record(ago_s=7500.0))

    assert len(sent) == 1
    title, body, priority = sent[0]
    assert title    == "train_backbone finished"
    assert priority == "default"
    assert "2h 05m" in body
    assert "abc123def456" in body


def test_failure_notifies_regardless_of_runtime(notifier, sent):
    _arm(notifier, min_runtime_s=60.0)
    notifier.job_finished(_record(status="failed", exit_code=1, ago_s=5.0))

    assert len(sent) == 1
    title, body, priority = sent[0]
    assert title    == "train_backbone FAILED (exit 1)"
    assert priority == "high"


def test_stopped_job_stays_silent(notifier, sent):
    _arm(notifier)
    notifier.job_finished(_record(status="failed", exit_code=-15, stopped=True))
    notifier.job_finished(_record(status="finished", exit_code=None, stopped=True))
    assert sent == []


def test_unknown_exit_is_labelled(notifier, sent):
    _arm(notifier, min_runtime_s=60.0)
    notifier.job_finished(_record(exit_code=None, ago_s=600.0))

    assert len(sent) == 1
    assert sent[0][0] == "train_backbone finished (exit status unknown)"


def test_delivery_over_http(notifier, capture_server):
    result = notifier.configure({"enabled": True, "topic": "unit-test-topic", "server": capture_server, "min_runtime_s": 60.0})
    assert result["ok"]

    notifier.job_finished(_record(ago_s=600.0))

    assert len(CaptureHandler.requests) == 1
    request = CaptureHandler.requests[0]
    assert request["path"]     == "/unit-test-topic"
    assert request["title"]    == "train_backbone finished"
    assert request["priority"] == "default"
    assert "10m 00s" in request["body"]


def test_test_message_requires_topic(notifier):
    assert not notifier.test()["ok"]


def test_test_message_is_delivered(notifier, capture_server):
    result = notifier.configure({"enabled": False, "topic": "unit-test-topic", "server": capture_server, "min_runtime_s": 60.0})
    assert result["ok"]

    assert notifier.test()["ok"]
    assert len(CaptureHandler.requests) == 1
    assert CaptureHandler.requests[0]["title"] == "DLR-TomoSAR test notification"
