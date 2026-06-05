from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _pin_environment() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int, default=0)
    args, _ = parser.parse_known_args()

    os.environ["CUDA_VISIBLE_DEVICES"]    = str(args.gpu)
    os.environ["MKL_NUM_THREADS"]         = "4"
    os.environ["NUMEXPR_NUM_THREADS"]     = "4"
    os.environ["OMP_NUM_THREADS"]         = "4"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def main() -> None:
    _pin_environment()

    from configuration.single_train_config import SingleTrainConfig
    from pipelines.training_pipeline.pipeline import SingleTrainRunner
    from tools.config_cli import ConfigCli

    config = ConfigCli(SingleTrainConfig(), description="Single training run").apply()
    SingleTrainRunner(config).run()


if __name__ == "__main__":
    main()
