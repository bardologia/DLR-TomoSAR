from __future__ import annotations

import json
import re
import socket
import threading
import urllib.error
import urllib.request

from project_paths import ProjectPaths
from web_logger    import WebLogger


class TelegramBot:

    SETTINGS_NAME  = "telegram.json"
    API_BASE       = "https://api.telegram.org"
    TOKEN_PATTERN  = re.compile(r"^\d+:[A-Za-z0-9_-]{30,64}$")
    CHAT_PATTERN   = re.compile(r"^-?\d+$")
    SEND_TIMEOUT_S = 10.0
    DEFAULTS       = {"notify_enabled": False, "commands_enabled": False, "token": "", "chat_id": ""}

    def __init__(self, paths: ProjectPaths, logger: WebLogger) -> None:
        self.paths    = paths
        self.logger   = logger
        self.lock     = threading.Lock()
        self.path     = paths.logs_dir / self.SETTINGS_NAME
        self.settings = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return dict(self.DEFAULTS)
        loaded = json.loads(self.path.read_text())
        return {key: loaded[key] for key in self.DEFAULTS}

    def state(self) -> dict:
        with self.lock:
            return {**self.settings, "settings_path": str(self.path)}

    def configure(self, payload: dict) -> dict:
        notify   = bool(payload.get("notify_enabled", False))
        commands = bool(payload.get("commands_enabled", False))
        token    = str(payload.get("token", "")).strip()
        chat_id  = str(payload.get("chat_id", "")).strip()

        if token and not self.TOKEN_PATTERN.match(token):
            return {"ok": False, "error": "that does not look like a bot token (expected '<digits>:<35-char key>' from @BotFather)"}
        if chat_id and not self.CHAT_PATTERN.match(chat_id):
            return {"ok": False, "error": "chat id must be a number — use detect after messaging the bot"}
        if (notify or commands) and not token:
            return {"ok": False, "error": "set a bot token before enabling"}
        if (notify or commands) and not chat_id:
            return {"ok": False, "error": "set a chat id before enabling — message the bot, then press detect"}

        with self.lock:
            self.settings = {"notify_enabled": notify, "commands_enabled": commands, "token": token, "chat_id": chat_id}
            self._persist()

        self.logger.ok(f"telegram notify {'on' if notify else 'off'}, remote commands {'on' if commands else 'off'}")
        return {"ok": True, **self.state()}

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.settings, indent=2) + "\n")

    def notify_enabled(self) -> bool:
        with self.lock:
            return self.settings["notify_enabled"] and bool(self.settings["token"]) and bool(self.settings["chat_id"])

    def commands_enabled(self) -> bool:
        with self.lock:
            return self.settings["commands_enabled"] and bool(self.settings["token"]) and bool(self.settings["chat_id"])

    def chat_id(self) -> str:
        with self.lock:
            return self.settings["chat_id"]

    def _call(self, method: str, payload: dict, timeout_s: float) -> tuple[list | dict | None, str | None]:
        with self.lock:
            token = self.settings["token"]

        request = urllib.request.Request(f"{self.API_BASE}/bot{token}/{method}", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")

        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            try:
                description = json.loads(error.read().decode("utf-8")).get("description", str(error))
            except (json.JSONDecodeError, OSError):
                description = str(error)
            return None, str(description)
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
            return None, str(error)

        if not body.get("ok"):
            return None, str(body.get("description", "telegram error"))
        return body.get("result"), None

    def send(self, text: str) -> str | None:
        with self.lock:
            chat_id = self.settings["chat_id"]

        result, error = self._call("sendMessage", {"chat_id": chat_id, "text": text}, self.SEND_TIMEOUT_S)
        return error

    def _deliver(self, text: str) -> None:
        error = self.send(text)
        if error is not None:
            self.logger.error(f"telegram message failed: {error}")

    def send_async(self, text: str) -> None:
        threading.Thread(target=self._deliver, args=(text,), daemon=True).start()

    def get_updates(self, offset: int | None, timeout_s: float) -> tuple[list | None, str | None]:
        payload = {"timeout": int(timeout_s), "allowed_updates": ["message"]}
        if offset is not None:
            payload["offset"] = offset
        return self._call("getUpdates", payload, timeout_s + 10.0)

    def detect_chats(self) -> dict:
        with self.lock:
            token       = self.settings["token"]
            commands_on = self.settings["commands_enabled"]

        if not token:
            return {"ok": False, "error": "save a bot token first"}
        if commands_on:
            return {"ok": False, "error": "disable remote commands while pairing — the listener consumes the messages detect needs"}

        updates, error = self._call("getUpdates", {"timeout": 0, "allowed_updates": ["message"]}, self.SEND_TIMEOUT_S)
        if error is not None:
            return {"ok": False, "error": error}

        chats = {}
        for update in updates:
            chat = (update.get("message") or {}).get("chat") or {}
            if "id" not in chat:
                continue
            name            = chat.get("username") or " ".join(filter(None, [chat.get("first_name"), chat.get("last_name")])) or chat.get("title") or str(chat["id"])
            chats[chat["id"]] = name

        return {"ok": True, "chats": [{"id": chat_id, "name": name} for chat_id, name in chats.items()]}

    def test(self) -> dict:
        with self.lock:
            if not self.settings["token"] or not self.settings["chat_id"]:
                return {"ok": False, "error": "set a bot token and chat id first"}

        error = self.send(f"DLR-TomoSAR test message\nthe console on {socket.gethostname()} can reach you")
        if error is not None:
            return {"ok": False, "error": error}
        return {"ok": True}
