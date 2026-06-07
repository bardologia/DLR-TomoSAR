from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from web_logger import WebLogger


class ResultsBrowser:

    IMAGE_SUFFIXES     = {".png", ".jpg", ".jpeg", ".svg", ".webp"}
    ANIMATION_SUFFIXES = {".gif"}
    CONFIG_SUFFIXES    = {".json", ".yaml", ".yml", ".toml", ".ini"}
    MARKDOWN_SUFFIXES  = {".md"}

    SKIPPED_DIRS   = {"__pycache__", ".git", ".ipynb_checkpoints"}
    MAX_DEPTH      = 10
    MAX_TEXT_BYTES = 262144

    STAGE_MARKERS = (
        ("benchmark",        ("pipeline/resolved_config.json", "results/benchmark_overview.md", "benchmark_overview.md")),
        ("preprocess",       ("images/slc", "images/interferograms", "images/dem")),
        ("param extraction", ("images/colormaps", "images/example_fits", "fit_report.md")),
        ("training",         ("docs/trainer_config.json", "checkpoints", "tensorboard")),
        ("inference",        ("inference", "figures", "animations")),
    )

    def __init__(self, logger: WebLogger) -> None:
        self.logger = logger
        self.roots  = set()

    def tree(self, raw_path: str) -> dict:
        root = Path(raw_path).expanduser()
        if not raw_path.strip() or not root.is_absolute():
            return {"ok": False, "error": "an absolute path is required"}

        root = root.resolve()
        if not root.is_dir():
            return {"ok": False, "error": f"not a directory: {root}"}

        self.roots.add(str(root))
        self.logger.info(f"results: opened {root}")

        return {
            "ok"    : True,
            "root"  : str(root),
            "name"  : root.name,
            "stage" : self._stage(root),
            "tree"  : self._node(root, root, 0),
        }

    def folder(self, raw_root: str, rel: str) -> dict:
        if raw_root not in self.roots:
            return {"ok": False, "error": "path not opened"}

        folder = (Path(raw_root) / rel).resolve() if rel else Path(raw_root)
        if not folder.is_relative_to(raw_root) or not folder.is_dir():
            return {"ok": False, "error": "unknown folder"}

        markdown, images, animations, configs, other = [], [], [], [], []

        for entry in sorted(folder.iterdir()):
            if not entry.is_file():
                continue

            suffix = entry.suffix.lower()

            if suffix in self.MARKDOWN_SUFFIXES:
                markdown.append({"name": entry.name, "text": self._read_text(entry)})
            elif suffix in self.IMAGE_SUFFIXES:
                images.append({"name": entry.stem, "url": self._url(entry)})
            elif suffix in self.ANIMATION_SUFFIXES:
                animations.append({"name": entry.stem, "url": self._url(entry)})
            elif suffix in self.CONFIG_SUFFIXES:
                configs.append({"name": entry.name, "kind": suffix.lstrip("."), "text": self._read_text(entry)})
            else:
                other.append({"name": entry.name, "size": entry.stat().st_size})

        return {
            "ok"         : True,
            "root"       : raw_root,
            "rel"        : rel,
            "abs"        : str(folder),
            "markdown"   : markdown,
            "images"     : images,
            "animations" : animations,
            "configs"    : configs,
            "other"      : other,
        }

    def file_path(self, raw_path: str) -> Path | None:
        target = Path(raw_path).resolve()
        if not any(target.is_relative_to(root) for root in self.roots):
            return None
        if not target.is_file():
            return None
        return target

    def _node(self, directory: Path, root: Path, depth: int) -> dict:
        counts   = {"markdown": 0, "images": 0, "animations": 0, "configs": 0, "other": 0}
        children = []

        try:
            entries = sorted(directory.iterdir())
        except OSError:
            entries = []

        for entry in entries:
            if entry.is_dir():
                if entry.name in self.SKIPPED_DIRS or entry.name.startswith("."):
                    continue
                if depth < self.MAX_DEPTH:
                    children.append(self._node(entry, root, depth + 1))
                continue

            suffix = entry.suffix.lower()

            if suffix in self.MARKDOWN_SUFFIXES:
                counts["markdown"] += 1
            elif suffix in self.IMAGE_SUFFIXES:
                counts["images"] += 1
            elif suffix in self.ANIMATION_SUFFIXES:
                counts["animations"] += 1
            elif suffix in self.CONFIG_SUFFIXES:
                counts["configs"] += 1
            else:
                counts["other"] += 1

        rel = "" if directory == root else str(directory.relative_to(root))

        return {
            "name"     : directory.name,
            "rel"      : rel,
            "counts"   : counts,
            "children" : children,
        }

    def _stage(self, root: Path) -> str:
        for stage, markers in self.STAGE_MARKERS:
            if any((root / marker).exists() for marker in markers):
                return stage
        return "results"

    def _read_text(self, target: Path) -> str:
        try:
            raw = target.read_bytes()
        except OSError:
            return ""

        text = raw[: self.MAX_TEXT_BYTES].decode("utf-8", errors="replace")
        if len(raw) > self.MAX_TEXT_BYTES:
            text += "\n\n[truncated]"
        return text

    def _url(self, target: Path) -> str:
        return "/resultsmedia?path=" + quote(str(target))
