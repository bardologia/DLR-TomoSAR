from __future__ import annotations

import threading
from pathlib import Path


class CatalogRoots:

    NOT_SET = "set the runs directory in the Results tab first"

    def __init__(self) -> None:
        self.lock  = threading.Lock()
        self.roots = set()

    @staticmethod
    def resolve(raw: str, empty_error: str = NOT_SET) -> tuple[Path | None, str]:
        raw = (raw or "").strip()
        if not raw:
            return None, empty_error

        root = Path(raw).expanduser()
        if not root.is_absolute():
            return None, "an absolute path is required"

        root = root.resolve()
        if not root.is_dir():
            return None, f"not a directory: {root}"

        return root, ""

    def add(self, root: Path) -> None:
        with self.lock:
            self.roots.add(str(root))

    def known(self, raw: str) -> bool:
        return raw in self.snapshot()

    def contains(self, target: Path) -> bool:
        return any(target.is_relative_to(root) for root in self.snapshot())

    def enclosing(self, target: Path) -> Path | None:
        matches = [root for root in self.snapshot() if target.is_relative_to(root)]
        if not matches:
            return None
        return Path(max(matches, key=len))

    def snapshot(self) -> tuple:
        with self.lock:
            return tuple(self.roots)

    def open(self, raw: str, empty_error: str = NOT_SET) -> tuple[Path | None, str]:
        root, error = self.resolve(raw, empty_error)
        if error:
            return None, error

        self.add(root)
        return root, ""
