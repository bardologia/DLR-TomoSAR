from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

dataset_path = Path("/ste/rnd/User/vice_vi/Dataset/clean_dataset")
params_path  = Path("/ste/rnd/User/vice_vi/Dataset/clean_dataset/params/params_sig_k5/parameters_sig_k5.npy")
log_base_dir = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/logs/benchmark")

gpus                = [0, 1, 2, 3]
epochs              = 200
batch_size          = 256
num_workers         = 4
warmup_steps        = 200
eta_min             = 1e-6
early_stop_patience = 30

skip_models: set[str] = set()

run_tag: str | None = None


def _scheduler(tag: str) -> None:
    from models import CONFIG_REGISTRY
    from tools.logger import Logger

    run_dir = log_base_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(log_dir=str(run_dir), name="benchmark")

    all_models    = list(CONFIG_REGISTRY.keys())
    models_to_run = [m for m in all_models if m not in skip_models]

    logger.section("Benchmark")
    logger.kv_table({
        "Run tag":    tag,
        "Models":     len(models_to_run),
        "GPUs":       gpus,
        "Epochs":     epochs,
        "Batch size": batch_size,
        "Log dir":    str(run_dir),
    }, title="Configuration")

    queue    = list(models_to_run)
    gpu_pool = list(gpus)
    running : list[tuple] = []
    results : list[dict]  = []
    summary_path = run_dir / "benchmark_results.json"

    def _reap_finished() -> None:
        still_running = []
        for proc, model_name, gpu_id, log_path in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((proc, model_name, gpu_id, log_path))
            else:
                status = "DONE" if ret == 0 else "FAILED"
                if ret == 0:
                    logger.info(f"✓  [GPU {gpu_id}] {model_name}  →  {status}")
                else:
                    logger.error(f"✗  [GPU {gpu_id}] {model_name}  →  {status}  (exit {ret})")
                results.append({
                    "model"   : model_name,
                    "status"  : status,
                    "gpu"     : gpu_id,
                    "logdir"  : str(run_dir / model_name),
                    "log_file": str(log_path),
                    "error"   : None if ret == 0 else f"exit code {ret} — see {log_path}",
                })
                gpu_pool.append(gpu_id)
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
        running.clear()
        running.extend(still_running)

    logger.section("Running")
    while queue or running:
        _reap_finished()
        while queue and gpu_pool:
            model_name = queue.pop(0)
            gpu_id     = gpu_pool.pop(0)
            log_path   = run_dir / model_name / "worker.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, __file__,
                "--worker",
                "--model",   model_name,
                "--gpu",     str(gpu_id),
                "--run-tag", tag,
            ]
            logger.info(f"▶  [GPU {gpu_id}] launching {model_name}")
            log_fh = open(log_path, "w")
            proc   = subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh)
            running.append((proc, model_name, gpu_id, log_path))
        time.sleep(5)

    done   = [r for r in results if r["status"] == "DONE"]
    failed = [r for r in results if r["status"] == "FAILED"]

    logger.section("Summary")
    logger.kv_table({
        "Total":  len(results),
        "Done":   len(done),
        "Failed": len(failed),
    }, title=f"{len(done)}/{len(results)} finished")

    if done:
        logger.subsection("Done")
        for r in done:
            logger.info(f"✓  [GPU {r['gpu']}] {r['model']}")

    if failed:
        logger.subsection("Failed")
        for r in failed:
            logger.error(f"✗  [GPU {r['gpu']}] {r['model']}  →  {r['log_file']}")

    logger.info(f"Results saved to: {summary_path}")
    logger.close()

    if failed:
        sys.exit(1)


def _worker(model_name: str, gpu_id: int, tag: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"]    = str(gpu_id)
    os.environ["MKL_NUM_THREADS"]         = "4"
    os.environ["NUMEXPR_NUM_THREADS"]     = "4"
    os.environ["OMP_NUM_THREADS"]         = "4"

    from configuration.dataset_config import (
        DatasetConfiguration, InputConfig, PatchConfiguration,
        Representation, SplitRegions,
    )
    from tools.crop_region      import CropRegion
    from tools.loss_scale_probe import LossScaleProbeConfig
    from tools.logger           import Logger
    from configuration.training_config import (
        LossCurriculumConfig, EarlyStoppingConfig, EMAConfig, GaussianConfig,
        GradientClipperConfig, IOConfig, LossConfig, OptimizerConfig,
        OverfitConfig, SchedulerConfig, TrainerConfig, TrainingConfigInner, WarmupConfig,
    )
    from models import CONFIG_REGISTRY
    from pipelines.training_pipeline.pipeline import TrainingPipeline

    run_dir      = log_base_dir / tag
    model_logdir = str(run_dir / model_name)

    logger = Logger(log_dir=model_logdir, name=f"worker_{model_name}")
    logger.section(f"GPU {gpu_id}  —  {model_name}")

    with open(dataset_path / "data" / "dataset.json", "r", encoding="utf-8") as f:
        layout = json.load(f)
    global_crop = CropRegion(*layout["global_crop"])

    split_regions = SplitRegions(
        train = CropRegion(1000,  9120,  global_crop.range_start, global_crop.range_end),
        val   = CropRegion(9120,  12400, global_crop.range_start, global_crop.range_end),
        test  = CropRegion(12400, 16000, global_crop.range_start, global_crop.range_end),
    )

    dataset_config = DatasetConfiguration(
        preprocessing_run_directory = dataset_path,
        parameters_path             = params_path,
        split_regions               = split_regions,
        patch        = PatchConfiguration(size=(64, 64), stride=32, use_reflective_padding=True),
        input_config = InputConfig(
            use_primary        = True,  primary_representation        = Representation.MAG_ONLY,
            use_secondaries    = True,  secondaries_representation    = Representation.MAG_ONLY,
            use_interferograms = True,  interferograms_representation = Representation.ANGLE_ONLY,
        ),
        batch_size    = batch_size,
        num_workers   = num_workers,
        shuffle_train = True,
        pin_memory    = True,
    )

    trainer_config = TrainerConfig(
        gaussian       = GaussianConfig.from_dataset(dataset_path, n_gaussians=5),
        early_stopping = EarlyStoppingConfig(patience=early_stop_patience, min_delta=0.0001, restore_best=True),
        warmup         = WarmupConfig(warmup_steps=warmup_steps, warmup_start_factor=0.1, warmup_enabled=True, warmup_mode="linear"),
        scheduler      = SchedulerConfig(type="cosine_annealing", epochs=epochs, eta_min=eta_min),
        ema              = EMAConfig(use_ema=False, ema_decay=0.999),
        optimizer        = OptimizerConfig(betas=(0.9, 0.999), eps=1e-8),
        gradient_clipper = GradientClipperConfig(clip_mode="fixed", max_grad_norm=1.0),
        io               = IOConfig(logdir=model_logdir),
        training = TrainingConfigInner(
            device="gpu", epochs=epochs, validation_frequency=1,
            use_amp=False, gradient_accumulation_steps=1,
            max_grad_norm=None, verbose=True,
        ),
        overfit    = OverfitConfig(enabled=False),
        curriculum = LossCurriculumConfig(
            enabled  = False,
            warmup   = LossConfig(use_param_l1=True, weight_param_l1=1.0, param_weights=(1.0, 1.0, 1.0)),
            complete = LossConfig(use_param_l1=True, weight_param_l1=1.0),
        ),
    )

    model_config = CONFIG_REGISTRY[model_name]()

    pipeline = TrainingPipeline(
        trainer_config = trainer_config,
        dataset_config = dataset_config,
        model_name     = model_name,
        model_config   = model_config,
        seed           = 0,
        run_name       = f"benchmark_{model_name}",
    )

    pipeline.run(probe_config=LossScaleProbeConfig(
        enabled=False, n_batches=100, reference="param_l1",
        exit_after=True, enabled_losses={},
    ))

    logger.info(f"✓  {model_name}  →  DONE")
    logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker",  action="store_true")
    parser.add_argument("--model",   type=str, default=None)
    parser.add_argument("--gpu",     type=int, default=0)
    parser.add_argument("--run-tag", type=str, default=None)
    args, _ = parser.parse_known_args()

    if args.worker:
        if args.model is None:
            sys.exit("ERROR: --worker requires --model")
        _worker(
            model_name = args.model,
            gpu_id     = args.gpu,
            tag        = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    else:
        _scheduler(tag=run_tag or datetime.now().strftime("%Y%m%d_%H%M%S"))


if __name__ == "__main__":
    main()
