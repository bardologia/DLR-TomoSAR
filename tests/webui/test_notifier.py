from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib  import Path

import pytest

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from notifier   import JobNotifier
from web_logger import WebLogger


class StubBot:

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.sent    = []

    def notify_enabled(self) -> bool:
        return self.enabled

    def send_async(self, text: str) -> None:
        self.sent.append(text)


@pytest.fixture
def bot():
    return StubBot()


@pytest.fixture
def notifier(bot):
    return JobNotifier(bot, WebLogger())


def _record(status: str = "finished", exit_code: int | None = 0, ago_s: float = 3600.0, stopped: bool = False, description: str = "") -> dict:
    started = (datetime.now() - timedelta(seconds=ago_s)).isoformat(timespec="seconds")
    record  = {"job_id": "abc123def456", "script": "train_backbone", "status": status, "exit_code": exit_code, "started": started, "pid": 4242, "description": description}
    if stopped:
        record["stopped"] = True
    return record


def test_disabled_bot_sends_nothing(notifier, bot):
    bot.enabled = False
    notifier.job_started(_record(status="running"))
    notifier.job_finished(_record())
    assert bot.sent == []


def test_start_notifies(notifier, bot):
    notifier.job_started(_record(status="running"))

    assert len(bot.sent) == 1
    assert bot.sent[0].startswith("train_backbone started\n")
    assert "abc123def456" in bot.sent[0]
    assert "4242" in bot.sent[0]


def test_start_carries_description(notifier, bot):
    notifier.job_started(_record(status="running", description="presence trials experiment · resunet-conv"))

    assert "\npresence trials experiment · resunet-conv\n" in bot.sent[0]


def test_finish_carries_description_and_runtime(notifier, bot):
    notifier.job_finished(_record(ago_s=600.0, description="single training · resunet-conv"))

    assert len(bot.sent) == 1
    assert bot.sent[0].startswith("train_backbone finished\nsingle training · resunet-conv\n")
    assert "10m 00s" in bot.sent[0]


def test_record_without_description_keeps_plain_body(notifier, bot):
    notifier.job_finished({key: value for key, value in _record(ago_s=600.0).items() if key != "description"})

    assert bot.sent[0].startswith("train_backbone finished\nruntime 10m 00s")


def test_long_success_runtime_label(notifier, bot):
    notifier.job_finished(_record(ago_s=7500.0))

    assert "2h 05m" in bot.sent[0]


def test_failure_notifies(notifier, bot):
    notifier.job_finished(_record(status="failed", exit_code=1, ago_s=5.0))

    assert bot.sent[0].startswith("train_backbone FAILED (exit 1)\n")


def test_stopped_job_notifies_as_stopped(notifier, bot):
    notifier.job_finished(_record(status="failed", exit_code=-15, stopped=True))

    assert bot.sent[0].startswith("train_backbone stopped (exit -15)\n")


def test_cancelled_job_stays_silent(notifier, bot):
    notifier.job_finished(_record(status="cancelled", exit_code=None))
    assert bot.sent == []


def test_unknown_exit_is_labelled(notifier, bot):
    notifier.job_finished(_record(exit_code=None, ago_s=600.0))

    assert bot.sent[0].startswith("train_backbone finished (exit status unknown)\n")
