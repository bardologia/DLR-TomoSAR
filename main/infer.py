from __future__ import annotations

import os
os.environ["MKL_NUM_THREADS"]     = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"]     = "4"

import sys
from dataclasses import dataclass, field
from pathlib     import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from configuration.inference_config        import InferenceConfig
from pipelines.inference_pipeline.pipeline import InferencePipeline


@dataclass
class RunInferenceConfig:
    run_directory      : Path = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/logs/unet_v1")
    output_subdir      : str  = ""
    split              : str  = "test"
    device             : str  = "cuda" if torch.cuda.is_available() else "cpu"

    use_ema            : bool = True
    checkpoint_name    : str  = "best_model.pt"

    batch_size         : int | None = None
    num_workers        : int        = 4

    stitch_window      : str  = "hann"
    save_cubes         : bool = True

    n_best_profiles    : int  = 12
    n_worst_profiles   : int  = 12
    n_random_profiles  : int  = 12

    n_range_slices     : int  = 5
    n_azimuth_slices   : int  = 5
    n_elevation_slices : int  = 5

    gif_axes           : list[str]  = field(default_factory=lambda: ["elevation", "range", "azimuth"])
    gif_fps            : int  = 12
    gif_max_frames     : int  = 150


def main() -> None:
    run = RunInferenceConfig()

    cfg = InferenceConfig(
        run_directory      = run.run_directory,
        output_subdir      = run.output_subdir or None,
        device             = run.device,
        use_ema            = run.use_ema,
        checkpoint_name    = run.checkpoint_name,
        split              = run.split,
        batch_size         = run.batch_size,
        num_workers        = run.num_workers,
        stitch_window      = run.stitch_window,
        save_cubes         = run.save_cubes,
        n_best_profiles    = run.n_best_profiles,
        n_worst_profiles   = run.n_worst_profiles,
        n_random_profiles  = run.n_random_profiles,
        n_range_slices     = run.n_range_slices,
        n_azimuth_slices   = run.n_azimuth_slices,
        n_elevation_slices = run.n_elevation_slices,
        gif_axes           = run.gif_axes,
        gif_fps            = run.gif_fps,
        gif_max_frames     = run.gif_max_frames,
    )

    pipeline    = InferencePipeline(cfg)
    report_path = pipeline.run()
    print(f"\n[OK] Inference report written to: {report_path}")


if __name__ == "__main__":
    main()
