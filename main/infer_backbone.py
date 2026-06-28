from __future__ import annotations

from pathlib import Path


def main() -> None:
    from _inference_entry import InferenceEntry, RunType
    InferenceEntry(entry_script=Path(__file__).resolve(), run_type=RunType.BACKBONE).run()


if __name__ == "__main__":
    main()
