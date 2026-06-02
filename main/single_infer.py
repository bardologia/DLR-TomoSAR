from __future__ import annotations

import os
os.environ["MKL_NUM_THREADS"]     = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"]     = "4"

N_GPUS            = 1

_VISIBLE = ", ".join(str(i) for i in range(N_GPUS))
os.environ["CUDA_VISIBLE_DEVICES"] = _VISIBLE

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from configuration.inference_config        import InferenceConfig
from pipelines.inference_pipeline.pipeline import InferencePipeline


run_directory = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/logs/test/resunet_w-pHub1")

def main() -> None:
    cfg = InferenceConfig(
        run_directory      = run_directory,
        output_subdir      = None,
        device             = "cuda" if torch.cuda.is_available() else "cpu",
        use_ema            = True,
        checkpoint_name    = "best_model.pt",
        split              = "test",
        batch_size         = None,
        num_workers        = 4,
        stitch_window      = "hann",
        save_cubes         = True,
        n_best_profiles    = 12,
        n_worst_profiles   = 12,
        n_random_profiles  = 12,
        n_range_slices     = 5,
        n_azimuth_slices   = 5,
        n_elevation_slices = 5,
        gif_axes           = ["elevation", "range", "azimuth"],
        gif_fps            = 12,
        gif_max_frames     = 150,
        cpu_workers        = 16,
    )

    pipeline    = InferencePipeline(cfg)
    report_path = pipeline.run()
    print(f"\n[OK] Inference report written to: {report_path}")


if __name__ == "__main__":
    main()
