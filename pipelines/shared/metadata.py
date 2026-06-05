from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pipelines.shared.io import FileIO
from tools.logger        import Logger


class MetadataBase:
    def __init__(self, config, logger: Logger | None = None) -> None:
        self.config = config
        self.logger = logger

    @staticmethod
    def timestamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _save_json(self, payload: dict, path: Path, message: str | None = None) -> Path:
        out_path = FileIO.save_json(payload, path)

        if message and self.logger:
            self.logger.subsection(f"{message}: {out_path}")

        return out_path

    def _save_text(self, entries: dict, path: Path, message: str | None = None) -> Path:
        out_path = FileIO.save_text_metadata(entries, path)

        if message and self.logger:
            self.logger.subsection(f"{message}: {out_path}")

        return out_path
