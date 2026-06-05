from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*pynvml.*", category=FutureWarning)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class EnvironmentPinner:
    THREAD_LIMITS = {
        "MKL_NUM_THREADS"     : "4",
        "NUMEXPR_NUM_THREADS" : "4",
        "OMP_NUM_THREADS"     : "4",
    }

    @classmethod
    def threads(cls) -> None:
        for key, value in cls.THREAD_LIMITS.items():
            os.environ[key] = value

    @classmethod
    def gpu(cls, gpu_id: int | None = None, expandable_segments: bool = False) -> None:
        if gpu_id is None:
            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument("--gpu", type=int, default=0)
            args, _ = parser.parse_known_args()
            gpu_id  = args.gpu

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cls.threads()

        if expandable_segments:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
