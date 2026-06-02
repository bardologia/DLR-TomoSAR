from __future__ import annotations

import os
os.environ["MKL_NUM_THREADS"]     = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"]     = "4"

n_gpus = 1
os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(str(i) for i in range(n_gpus))

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from configuration.inference_config        import InferenceConfig
from pipelines.inference_pipeline.pipeline import InferencePipeline
from tools.logger                          import Logger


logs_dir = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/logs/test")

run_filter: list[str] = []
save_cubes: bool = False

run_dirs = sorted(
    [d for d in logs_dir.iterdir() if d.is_dir()]
    if not run_filter
    else [logs_dir / name for name in run_filter]
)

def main() -> None:
    logger = Logger(log_dir=str(logs_dir), name="batch_inference")

    logger.section("Batch inference")
    logger.kv_table({
        "Runs":     len(run_dirs),
        "Logs dir": str(logs_dir),
    }, title="Configuration")

    for run_dir in run_dirs:
        logger.subsection(run_dir.name)

        cfg = InferenceConfig(
            run_directory      = run_dir,
            output_subdir      = None,
            device             = "cuda",
            use_ema            = True,
            checkpoint_name    = "best_model.pt",
            split              = "test",
            batch_size         = None,
            num_workers        = 4,
            stitch_window      = "hann",
            save_cubes         = save_cubes,
            n_best_profiles    = 12,
            n_worst_profiles   = 12,
            n_random_profiles  = 12,
            n_range_slices     = 5,
            n_azimuth_slices   = 5,
            n_elevation_slices = 5,
            gif_axes           = [""],
            gif_fps            = 12,
            gif_max_frames     = 150,
            cpu_workers        = 16,
        )

        pipeline    = InferencePipeline(cfg)
        report_path = pipeline.run()
        logger.info(f"{run_dir.name}  :  {report_path}")

    logger.close()


if __name__ == "__main__":
    main()
