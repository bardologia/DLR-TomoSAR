from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.data.io           import FileIO
from tools.monitoring.logger import Logger


class DatasetLayoutClipBackfiller:

    LAYOUT_NAME       = "dataset.json"
    DATA_SUBDIR       = "data"
    METADATA_SUBDIR   = "meta"
    CONFIG_STATE_NAME = "config_state.json"
    CLIP_KEY          = "max_amplitude_clip"

    def __init__(self, root: Path, logger: Logger) -> None:
        self.root   = Path(root)
        self.logger = logger

    def _layouts(self) -> list[Path]:
        return sorted(self.root.rglob(f"{self.DATA_SUBDIR}/{self.LAYOUT_NAME}"))

    def _backfill_one(self, layout_path: Path) -> bool:
        layout = FileIO.load_json(layout_path)
        if self.CLIP_KEY in layout:
            return False

        clip = self._recover_clip(layout_path.parent.parent)

        layout[self.CLIP_KEY] = clip
        FileIO.save_json(layout, layout_path, indent=2)

        self.logger.info(f"{layout_path}  <-  {self.CLIP_KEY}={clip}")
        return True

    def _recover_clip(self, run_dir: Path) -> float:
        state_path = run_dir / self.METADATA_SUBDIR / self.CONFIG_STATE_NAME
        if not state_path.is_file():
            raise FileNotFoundError(f"No {self.METADATA_SUBDIR}/{self.CONFIG_STATE_NAME} under {run_dir}; cannot recover {self.CLIP_KEY}.")

        state = FileIO.load_json(state_path)
        return float(state["tomogram_config"][self.CLIP_KEY])

    def run(self) -> None:
        self.logger.section("Backfilling dataset-layout amplitude clip")

        layouts = self._layouts()
        self.logger.subsection(f"Found {len(layouts)} dataset layouts under {self.root}")

        updated = already = failed = 0
        for layout_path in layouts:
            try:
                if self._backfill_one(layout_path):
                    updated += 1
                else:
                    already += 1
            except (FileNotFoundError, KeyError, ValueError) as error:
                failed += 1
                self.logger.error(f"{layout_path}: {error}")

        self.logger.section(f"[Done] {updated} updated, {already} already complete, {failed} unrecoverable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill max_amplitude_clip into preprocessing dataset.json layouts from config_state.json")
    parser.add_argument("--root", default="/ste/rnd/User/vice_vi/Dataset")
    args = parser.parse_args()

    logger = Logger(log_dir="logs", name="backfill_dataset_layout_clip")
    DatasetLayoutClipBackfiller(Path(args.root), logger).run()
    logger.close()


if __name__ == "__main__":
    main()
