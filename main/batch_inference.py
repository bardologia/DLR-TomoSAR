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

    from configuration.inference_config import BatchInferenceConfig
    from pipelines.inference_pipeline.pipeline import InferencePipeline
    from tools.config_cli import ConfigCli
    from tools.logger import Logger

    config   = ConfigCli(BatchInferenceConfig(), description="Batch inference over run directories").apply()
    logs_dir = Path(config.logs_dir)

    run_dirs = sorted(
        [d for d in logs_dir.iterdir() if d.is_dir()]
        if not config.run_filter
        else [logs_dir / name for name in config.run_filter]
    )

    logger = Logger(log_dir=str(logs_dir), name="batch_inference")

    logger.section("Batch inference")
    logger.kv_table({
        "Runs"     : len(run_dirs),
        "Logs dir" : str(logs_dir),
    }, title="Configuration")

    for run_dir in run_dirs:
        logger.subsection(run_dir.name)

        pipeline    = InferencePipeline(replace(config.inference, run_directory=run_dir, output_subdir=None))
        report_path = pipeline.run()

        logger.info(f"{run_dir.name}  :  {report_path}")

    logger.close()


if __name__ == "__main__":
    main()
