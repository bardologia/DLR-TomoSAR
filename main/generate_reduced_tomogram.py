from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import EnvironmentPinner


def main() -> None:
    EnvironmentPinner.threads()

    parser = argparse.ArgumentParser(description="Generate a reduced Capon tomogram via pyrat (runs under the pyrat env, e.g. stetools)")
    parser.add_argument("--spec", required=True, help="Path to the reduced-tomogram job spec JSON")
    args = parser.parse_args()

    from pipelines.inference_pipeline.reduced_generation import ReducedTomogramGenerator
    from tools.logger import Logger

    spec_path = Path(args.spec)

    with Logger(log_dir=str(spec_path.parent), name="reduced_tomogram") as logger:
        ReducedTomogramGenerator.from_spec_file(spec_path, logger).run()


if __name__ == "__main__":
    main()
