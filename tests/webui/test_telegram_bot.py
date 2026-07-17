from __future__ import annotations

import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib     import Path

import pytest

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from telegram_bot import TelegramBot
from web_logger   import WebLogger

TOKEN = "123456789:" + "A" * 35


class StubPaths:

    def __init__(self, root: Path) -> None:
        self.logs_dir = root / "logs"


class TelegramStub(BaseHTTPRequestHandler):

    sent      = []
    pending   = []
    fail_send = False

    def do_POST(self) -> None:
        length  = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")

        if self.path.endswith("/sendMessage"):
            if TelegramStub.fail_send:
                self._reply({"ok": False, "description": "chat not found"})
                return
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


def _update(update_id: int, chat_id: int, text: str, when: float | None = None, name: str = "victor") -> dict:
    return {"update_id": update_id, "message": {"chat": {"id": chat_id, "username": name}, "date": when if when is not None else time.time(), "text": text}}


@pytest.fixture
def stub_server():
    TelegramStub.sent      = []
    TelegramStub.pending   = []
    TelegramStub.fail_send = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), TelegramStub)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture
def bot(tmp_path, stub_server):
    instance          = TelegramBot(StubPaths(tmp_path), WebLogger())
    instance.API_BASE = stub_server
    return instance


def test_defaults_are_disabled(bot):
    state = bot.state()
    assert state["notify_enabled"]   is False
    assert state["commands_enabled"] is False
    assert state["token"]            == ""
    assert state["chat_id"]          == ""
    assert bot.notify_enabled()   is False
    assert bot.commands_enabled() is False


def test_configure_persists_across_instances(bot, tmp_path):
    assert bot.configure({"notify_enabled": True, "commands_enabled": True, "token": TOKEN, "chat_id": "777"})["ok"]

    reloaded = TelegramBot(StubPaths(tmp_path), WebLogger())
    state    = reloaded.state()
    assert state["notify_enabled"]   is True
    assert state["commands_enabled"] is True
    assert state["token"]            == TOKEN
    assert state["chat_id"]          == "777"


def test_configure_rejects_bad_token(bot):
    assert not bot.configure({"token": "not-a-token"})["ok"]
    assert not bot.configure({"token": "123:short"})["ok"]


def test_configure_rejects_bad_chat_id(bot):
    assert not bot.configure({"token": TOKEN, "chat_id": "12ab"})["ok"]


def test_configure_requires_token_and_chat_to_enable(bot):
    assert not bot.configure({"notify_enabled": True})["ok"]
    assert not bot.configure({"notify_enabled": True, "token": TOKEN})["ok"]
    assert not bot.configure({"commands_enabled": True, "chat_id": "777"})["ok"]
    assert bot.configure({"token": TOKEN})["ok"]
    assert bot.configure({"notify_enabled": True, "commands_enabled": True, "token": TOKEN, "chat_id": "777"})["ok"]


def test_negative_chat_id_is_accepted(bot):
    assert bot.configure({"token": TOKEN, "chat_id": "-100123"})["ok"]


def test_send_delivers_to_chat(bot):
    assert bot.configure({"token": TOKEN, "chat_id": "777"})["ok"]

    assert bot.send("hello there") is None
    assert len(TelegramStub.sent) == 1
    assert TelegramStub.sent[0]["chat_id"] == "777"
    assert TelegramStub.sent[0]["text"]    == "hello there"


def test_send_surfaces_api_error(bot):
    assert bot.configure({"token": TOKEN, "chat_id": "777"})["ok"]
    TelegramStub.fail_send = True

    assert bot.send("hello") == "chat not found"


def test_test_requires_pairing(bot):
    assert not bot.test()["ok"]


def test_test_sends_hostname(bot):
    assert bot.configure({"token": TOKEN, "chat_id": "777"})["ok"]

    assert bot.test()["ok"]
    assert socket.gethostname() in TelegramStub.sent[0]["text"]


def test_detect_requires_token(bot):
    result = bot.detect_chats()
    assert not result["ok"]
    assert "token" in result["error"]


def test_detect_refuses_while_listening(bot):
    assert bot.configure({"commands_enabled": True, "token": TOKEN, "chat_id": "777"})["ok"]

    result = bot.detect_chats()
    assert not result["ok"]
    assert "disable remote commands" in result["error"]


def test_detect_lists_distinct_chats(bot):
    assert bot.configure({"token": TOKEN})["ok"]
    TelegramStub.pending = [_update(1, 777, "hi"), _update(2, 777, "again"), _update(3, 888, "other", name="someone")]

    result = bot.detect_chats()
    assert result["ok"]
    assert {chat["id"] for chat in result["chats"]} == {777, 888}
    named = {chat["id"]: chat["name"] for chat in result["chats"]}
    assert named[777] == "victor"


def test_detect_with_no_messages(bot):
    assert bot.configure({"token": TOKEN})["ok"]

    result = bot.detect_chats()
    assert result["ok"]
    assert result["chats"] == []


def test_get_updates_honours_offset(bot):
    assert bot.configure({"token": TOKEN})["ok"]
    TelegramStub.pending = [_update(1, 777, "a"), _update(2, 777, "b")]

    updates, error = bot.get_updates(2, 0.0)
    assert error is None
    assert [update["update_id"] for update in updates] == [2]

    updates, error = bot.get_updates(-1, 0.0)
    assert [update["update_id"] for update in updates] == [2]
