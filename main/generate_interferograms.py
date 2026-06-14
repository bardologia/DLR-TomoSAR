from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.threads()

    parser = argparse.ArgumentParser(description="Build the interferometric stack via pyrat (runs under the pyrat env, e.g. stetools)")
    parser.add_argument("--spec", required=True, help="Path to the interferogram job spec JSON")
    args = parser.parse_args()

    from pipelines.processing_pipeline.interferogram import InterferogramGenerator
    from tools.monitoring.logger import Logger

    spec_path = Path(args.spec)

    with Logger(log_dir=str(spec_path.parent), name="interferograms") as logger:
        InterferogramGenerator.from_spec_file(spec_path, logger).run()


if __name__ == "__main__":
    main()
