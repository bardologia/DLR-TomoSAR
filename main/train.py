from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import EnvironmentPinner


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int, default=0)
    args, _ = parser.parse_known_args()

    EnvironmentPinner.gpu(args.gpu, expandable_segments=True)

    from pipelines.shared.training_launcher import TrainingLauncher
    TrainingLauncher(entry_script=Path(__file__).resolve()).run()

if __name__ == "__main__":
    main()
