"""
Run training for n_gaussians in [2, 3, 4, 5], one per GPU (0-3) in parallel.
Logs are saved under DLR-TomoSAR/logs/gaussians_sweep/<run_name>.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GAUSSIANS      = [2, 3, 4, 5]           # one job per value
GPUS           = [0, 1, 2, 3]           # must match len(GAUSSIANS)
SWEEP_LOGDIR   = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/logs/gaussians_sweep")
# Use the env's Python directly to avoid conda run spawning duplicate processes
PYTHON_BIN     = Path("/home/vice_vi/.conda/envs/dlr-cu12/bin/python")
REPO_ROOT      = Path(__file__).resolve().parent.parent

assert len(GAUSSIANS) == len(GPUS), "Need exactly one GPU per gaussian value."

# ---------------------------------------------------------------------------
# Per-run training script (written to a temp file and executed)
# ---------------------------------------------------------------------------
TRAIN_TEMPLATE = textwrap.dedent("""\
    from __future__ import annotations
    import os, json, sys
    os.environ["CUDA_VISIBLE_DEVICES"]    = "{gpu_id}"
    os.environ["MKL_NUM_THREADS"]         = "4"
    os.environ["NUMEXPR_NUM_THREADS"]     = "4"
    os.environ["OMP_NUM_THREADS"]         = "4"

    from pathlib import Path
    sys.path.insert(0, "{repo_root}")

    from configuration.dataset_config import (
        DatasetCreationConfiguration, InputConfig, PatchConfiguration,
        Representation, SplitRegions, InputNormalizationMode, OutputNormalizationMode,
    )
    from tools.crop_region import CropRegion
    from configuration.training_config import (
        EarlyStoppingConfig, EMAConfig, GaussianConfig, IOConfig, LossConfig,
        OptimizerConfig, SchedulerConfig, TrainerConfig, TrainingConfigInner, WarmupConfig,
    )
    from pipelines.training_pipeline.pipeline import TrainingPipeline

    n_gaussians  = {n_gaussians}
    model_name   = "unet"
    dataset_path = Path("/ste/rnd/User/vice_vi/Dataset/base_dataset")
    logdir       = "{logdir}"

    params_subdir = f"params_Ng{{n_gaussians}}_adapt"
    params_path   = dataset_path / "params" / params_subdir / f"parameters_Ng{{n_gaussians}}_adapt.npy"

    with open(dataset_path / "data" / "dataset.json", "r", encoding="utf-8") as f:
        layout = json.load(f)
    global_crop = CropRegion(*layout["global_crop"])

    split_regions = SplitRegions(
        train = CropRegion(1000,  9120,  global_crop.range_start, global_crop.range_end),
        val   = CropRegion(9120,  12400, global_crop.range_start, global_crop.range_end),
        test  = CropRegion(12400, 16000, global_crop.range_start, global_crop.range_end),
    )

    dataset_config = DatasetCreationConfiguration(
        preprocessing_run_directory = dataset_path,
        parameters_path             = params_path,
        split_regions               = split_regions,
        patch                       = PatchConfiguration(size=(64, 64), stride=32, use_reflective_padding=True),
        input_config = InputConfig(
            use_primary        = True,  primary_representation        = Representation.MAG_ONLY,
            use_secondaries    = True,  secondaries_representation    = Representation.MAG_ONLY,
            use_interferograms = True,  interferograms_representation  = Representation.ANGLE_ONLY,
        ),
        batch_size               = 256,
        num_workers              = 8,
        shuffle_train            = True,
        pin_memory               = False,
        input_normalization_mode  = InputNormalizationMode.GROUPED,
        output_normalization_mode = OutputNormalizationMode.GROUPED,
    )

    trainer_config = TrainerConfig(
        gaussian       = GaussianConfig.from_dataset(dataset_path, params_subdir=params_subdir),
        early_stopping = EarlyStoppingConfig(patience=15, min_delta=0.0001, restore_best=True),
        warmup         = WarmupConfig(warmup_steps=50, warmup_start_factor=0.1, warmup_enabled=True),
        scheduler      = SchedulerConfig(epochs=150, eta_min=1e-6),
        ema            = EMAConfig(use_ema=False, ema_decay=0.999),
        optimizer      = OptimizerConfig(lr=3e-4, betas=(0.9, 0.999), eps=1e-8),
        io             = IOConfig(logdir=logdir),
        training = TrainingConfigInner(
            device                      = "gpu",
            epochs                      = 150,
            validation_frequency        = 5,
            use_amp                     = False,
            gradient_accumulation_steps = 1,
            max_grad_norm               = None,
            verbose                     = True,
            overfit_enabled             = False,
            deep_validation             = False,
            eval_train_split            = False,
        ),
        loss = LossConfig(
            use_ssim_curve    = False,
            weight_ssim_curve = 1.0,
            ssim_window_size  = 11,
            ssim_sigma        = 1.5,
            ssim_data_range   = 1.0,
            ssim_k1           = 0.01,
            ssim_k2           = 0.03,
            ssim_axis         = "elevation",
            use_mse_curve     = True,
            weight_mse_curve  = 1.0,
        ),
    )

    if __name__ == "__main__":
        pipeline = TrainingPipeline(
            trainer_config = trainer_config,
            dataset_config = dataset_config,
            model_name     = model_name,
            seed           = 0,
        )
        pipeline.run()
""")


def launch_jobs() -> None:
    SWEEP_LOGDIR.mkdir(parents=True, exist_ok=True)

    processes: list[tuple[int, subprocess.Popen, Path]] = []

    for ng, gpu in zip(GAUSSIANS, GPUS):
        run_name  = f"Ng{ng}_adapt_gpu{gpu}"
        run_logdir = SWEEP_LOGDIR / run_name
        run_logdir.mkdir(parents=True, exist_ok=True)

        # Write a self-contained training script for this run
        script_path = run_logdir / "train_run.py"
        script_path.write_text(
            TRAIN_TEMPLATE.format(
                gpu_id    = gpu,
                repo_root = REPO_ROOT,
                n_gaussians = ng,
                logdir    = str(run_logdir),
            )
        )

        # Guard: skip if a previous run already exists in this folder
        existing = list(run_logdir.glob("run_*"))
        if existing:
            print(f"[launcher] SKIP Ng={ng} — previous run(s) found: {[e.name for e in existing]}")
            continue

        stdout_file = open(run_logdir / "stdout.log", "w")
        stderr_file = open(run_logdir / "stderr.log", "w")

        cmd = [str(PYTHON_BIN), str(script_path)]

        print(f"[launcher] Starting Ng={ng} on GPU {gpu}  →  {run_logdir}")
        proc = subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file)
        processes.append((ng, proc, run_logdir))

    print(f"\n[launcher] All {len(processes)} jobs launched. Waiting for completion...\n")

    exit_codes: dict[int, int] = {}
    for ng, proc, run_logdir in processes:
        ret = proc.wait()
        exit_codes[ng] = ret
        status = "✓ OK" if ret == 0 else f"✗ FAILED (exit {ret})"
        print(f"[launcher] Ng={ng}  {status}  —  logs: {run_logdir}")

    failed = [ng for ng, code in exit_codes.items() if code != 0]
    if failed:
        print(f"\n[launcher] {len(failed)} job(s) failed: Ng={failed}")
        sys.exit(1)
    else:
        print("\n[launcher] All jobs completed successfully.")


if __name__ == "__main__":
    launch_jobs()
