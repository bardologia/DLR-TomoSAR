from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _pin_environment() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int, default=0)
    args, _ = parser.parse_known_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MKL_NUM_THREADS"]      = "4"
    os.environ["NUMEXPR_NUM_THREADS"]  = "4"
    os.environ["OMP_NUM_THREADS"]      = "4"


def main() -> None:
    _pin_environment()

    from configuration.inference_config import SingleInferenceConfig
    from pipelines.inference_pipeline.pipeline import InferencePipeline
    from tools.config_cli import ConfigCli
    from tools.logger import Logger

    config = ConfigCli(SingleInferenceConfig(), description="Single inference run").apply()

    logger = Logger(log_dir=str(Path(config.run_directory) / "logs"), name="single_infer")

    pipeline    = InferencePipeline(replace(config.inference, run_directory=Path(config.run_directory), output_subdir=None))
    report_path = pipeline.run()

    logger.info(f"Inference report written to: {report_path}")
    logger.close()


if __name__ == "__main__":
    main()
