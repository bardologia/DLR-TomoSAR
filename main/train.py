from __future__ import annotations

from pathlib import Path

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.gpu(expandable_segments=True)
    from pipelines.training_launcher import TrainingLauncher
    TrainingLauncher(entry_script=Path(__file__).resolve()).run()

if __name__ == "__main__":
    main()
